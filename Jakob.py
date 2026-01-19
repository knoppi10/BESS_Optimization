"""
BESS Arbitrage Simulation - Korrigierte Version

WICHTIGE VERBESSERUNGEN (basierend auf wissenschaftlicher Literatur):

1. Wirkungsgrade implementiert:
   - SOC-Dynamik: SOC_{t+1} = SOC_t + dt * (η_ch * c_t - d_t / η_dis)
   - Separate Lade-/Entladevariablen (c_t, d_t) für korrekte Effizienzmodellierung

2. Zeitauflösung (dt) konsistent:
   - Automatische Erkennung der Datenfrequenz (Stunden vs. 15-Minuten)
   - dt wird konsistent in allen Berechnungen verwendet (Erlös, Kosten, Zyklen)

3. Initial-/End-SOC korrekt modelliert:
   - Initial SOC als Zustand (nicht als Entscheidung)
   - End-SOC Constraint explizit auf SOC-Zustand (nicht Summe der Entscheidungen)

4. Dimensionskonsistenz:
   - Erlös: dt * p_t * (d_t - c_t) [€]
   - Price Impact: alpha * dt * (d_t - c_t)^2 [€]
   - Hurdle: hurdle_rate * dt * (c_t + d_t) [€]

5. Literaturkonforme Parameter:
   - Rundreiseeffizienz: 0.88-0.94 (Standard: 0.92)
   - Degradationskosten: 20-40 €/MWh Durchsatz
   - Price Impact: Szenarioparameter (nicht kalibriert)
"""

import pandas as pd
import numpy as np
import osqp
from scipy import sparse
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
from typing import Optional
import warnings
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
warnings.filterwarnings('ignore')

# ==========================================
# 1. KONFIGURATION
# ==========================================
@dataclass
class SimulationConfig:
    """
    Konfiguration für BESS-Simulation mit wissenschaftlich fundierten Parametern.

    Literaturwerte:
    - Rundreiseeffizienz: 0.88-0.94 (typisch ~0.90)
    - C-Rate für Arbitrage: 0.5-1C typisch, 0.25C (4h-Speicher) konservativ
    - Degradationskosten: 20-40 €/MWh Durchsatz (abgeleitet aus CAPEX/2*Zyklen)
    - Price Impact: Kein Standardwert → Szenarioparameter, explizit als Modellannahme

    Hinweis Effizienz:
    - `roundtrip_efficiency` ist die gewünschte Rundreiseeffizienz η_rt.
    - Daraus werden symmetrische Einzelwirkungsgrade gesetzt: η_ch = η_dis = sqrt(η_rt).
    """

    # Ziel-Rundreiseeffizienz (η_rt). Einzelwirkungsgrade werden in __post_init__ gesetzt.
    roundtrip_efficiency: float = 0.92
    efficiency_charge: Optional[float] = None
    efficiency_discharge: Optional[float] = None

    c_rate: float = 0.25  # C-Rate (0.25 = 4-Stunden-Batterie)
    hurdle_rate: float = 30.0  # €/MWh Durchsatz (Degradations-/Opportunitätskosten)
    slope: float = 0.01  # Price Impact Parameter alpha [€/(MW^2*h)] (weil impact = alpha * dt[h] * (net_power[MW])^2)
    enforce_end_soc_zero: bool = True  # Zyklisch: End-SOC = Start-SOC
    debug: bool = False

    def __post_init__(self):
        # Plausibilitaet: entweder beide Einzelwirkungsgrade setzen, oder Rundreiseeffizienz verwenden
        if self.efficiency_charge is None and self.efficiency_discharge is None:
            # symmetrische Einzelwirkungsgrade so, dass eta_ch * eta_dis = eta_rt
            if not (0.0 < self.roundtrip_efficiency <= 1.0):
                raise ValueError("roundtrip_efficiency muss in (0, 1] liegen.")
            eta = float(np.sqrt(self.roundtrip_efficiency))
            self.efficiency_charge = eta
            self.efficiency_discharge = eta
        elif self.efficiency_charge is not None and self.efficiency_discharge is not None:
            # beide explizit gesetzt -> Rundreiseeffizienz ableiten
            if not (0.0 < float(self.efficiency_charge) <= 1.0) or not (0.0 < float(self.efficiency_discharge) <= 1.0):
                raise ValueError("efficiency_charge und efficiency_discharge muessen in (0, 1] liegen.")
            self.roundtrip_efficiency = float(self.efficiency_charge) * float(self.efficiency_discharge)
        else:
            raise ValueError("Bitte entweder beide Einzelwirkungsgrade setzen oder keine (dann wird roundtrip_efficiency verwendet).")

# ==========================================
# 2. DATEN LADEN
# ==========================================
def load_market_data(filename='market_data_2019_2025.csv'):
    """Lädt die echten Daten."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)

    if os.path.exists(filepath):
        print(f"Lade echte Daten aus {filepath}...")
        df = pd.read_csv(filepath)
        # Whitespace aus Spaltennamen entfernen
        df.columns = df.columns.str.strip()
        # Timestamp parsen
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df = df.set_index('timestamp')
        else:
            first_col = df.columns[0]
            try:
                df[first_col] = pd.to_datetime(df[first_col], utc=True)
                df = df.set_index(first_col)
                df.index.name = 'timestamp'
            except Exception:
                raise ValueError(f"Keine gültige 'timestamp' Spalte in '{filepath}' gefunden.")
        
        try:
            df.index = pd.DatetimeIndex(df.index)
        except Exception:
            raise ValueError("Index konnte nicht in DatetimeIndex umgewandelt werden.")
        
        if df.index.tz is None:
            df.index = df.index.tz_localize('Europe/Berlin')
        else:
            df.index = df.index.tz_convert('Europe/Berlin')
        
        return df
    else:
        raise FileNotFoundError(f"Datei '{filepath}' nicht gefunden! Bitte führe zuerst data_fetch.py aus.")

# ==========================================
# 3. OSQP OPTIMIZER (Quadratische Programmierung)
# ==========================================
class BessOptimizerQuadratic:
    def __init__(self, capacity_mwh, config: SimulationConfig, dt_hours=1.0):
        """
        Quadratisches Optimierungsproblem mit OSQP - korrigierte Version
        
        Args:
            capacity_mwh: Batteriekapazität in MWh
            config: SimulationConfig mit Parametern
            dt_hours: Zeitschritt in Stunden (1.0 für stündlich, 0.25 für 15-Minuten)
        """
        self.T = None  # Wird dynamisch gesetzt
        self.capacity = capacity_mwh
        self.power = capacity_mwh * config.c_rate
        self.cfg = config
        self.dt = dt_hours  # Zeitauflösung in Stunden
        
        # Wirkungsgrade
        self.eta_ch = config.efficiency_charge
        self.eta_dis = config.efficiency_discharge
        
        # Matrizen werden in solve_year gebaut (da T dynamisch ist)

    def solve_year(self, prices, enforce_end_soc_zero=False, initial_soc=0.0):
        """
        Löse für ein einzelnes Jahr mit korrigierter SOC-Dynamik und Effizienzen
        
        Optimierungsvariablen: x = [c_0, ..., c_{T-1}, d_0, ..., d_{T-1}, soc_0, ..., soc_T]
        wobei:
        - c_t = Ladeleistung (MW)
        - d_t = Entladeleistung (MW)
        - soc_t = SOC-Zustand (MWh)
        
        SOC-Dynamik als Constraint: soc_{t+1} = soc_t + dt * (η_ch * c_t - d_t / η_dis)
        Dies reduziert die Constraint-Matrix von O(T²) auf O(T) - skalierbar für 15-Min-Daten!
        """
        T = len(prices)
        self.T = T
        # Robustness: verhindere numerische Drift außerhalb der SOC-Grenzen
        if self.capacity > 0:
            initial_soc = float(np.clip(initial_soc, 0.0, self.capacity))
        else:
            initial_soc = 0.0
        alpha = self.cfg.slope
        power = self.power
        dt = self.dt
        
        # ===== OPTIMIERUNGSVARIABLEN: 2*T (c_t, d_t) + T+1 (soc_t) =====
        # Variablenreihenfolge: [c_0..c_{T-1}, d_0..d_{T-1}, soc_0..soc_T]
        n_vars = 2 * T + (T + 1)
        
        # ===== ZIELFUNKTION: min (1/2) x^T P x + q^T x =====
        # Erlös: dt * p_t * (d_t - c_t)  [im Minimierungsproblem: c_t kostet, d_t bringt Erlös]
        # Price Impact: alpha * dt * (d_t - c_t)^2  [auf Netto-Leistung]
        # Hurdle: hurdle_rate * dt * (c_t + d_t)  [Durchsatz-Kosten]
        
        # Hesse-Matrix P: quadratischer Term für Price Impact auf Netto-Leistung
        # Variablenreihenfolge: [c_0..c_{T-1}, d_0..d_{T-1}, soc_0..soc_T]
        # c_t ist bei Index t, d_t bei Index T+t, soc_t bei Index 2*T+t
        # Ziel: alpha * dt * (d_t - c_t)^2
        P_rows = []
        P_cols = []
        P_data = []
        for t in range(T):
            c_idx = t
            d_idx = T + t
            # c_t^2 Term
            P_rows.append(c_idx)
            P_cols.append(c_idx)
            P_data.append(2.0 * alpha * dt)
            # d_t^2 Term
            P_rows.append(d_idx)
            P_cols.append(d_idx)
            P_data.append(2.0 * alpha * dt)
            # -2*c_t*d_t Kreuzterm (symmetrisch)
            P_rows.append(c_idx)
            P_cols.append(d_idx)
            P_data.append(-2.0 * alpha * dt)
            P_rows.append(d_idx)
            P_cols.append(c_idx)
            P_data.append(-2.0 * alpha * dt)
        # SOC-Variablen haben keinen quadratischen Term
        self.P = sparse.csc_matrix((P_data, (P_rows, P_cols)), shape=(n_vars, n_vars))
        
        # Linearer Term q: Preise und Hurdle
        # Minimierung: c_t (Laden) kostet → positive Kosten, d_t (Entladen) bringt Erlös → negative Kosten
        q = np.zeros(n_vars)
        h = self.cfg.hurdle_rate
        for t in range(T):
            c_idx = t
            d_idx = T + t
            # c_t: laden kostet Preis + Hurdle (positive Kosten im Minimierungsproblem)
            q[c_idx] = dt * prices[t] + h * dt
            # d_t: verkaufen bringt Erlös, kostet Hurdle (negative Kosten = Erlös im Minimierungsproblem)
            q[d_idx] = -dt * prices[t] + h * dt
        # SOC-Variablen haben keinen linearen Term
        
        # ===== CONSTRAINTS =====
        # Variablenreihenfolge: [c_0..c_{T-1}, d_0..d_{T-1}, soc_0..soc_T]
        # 1. Power-Beschränkungen: 0 <= c_t <= power, 0 <= d_t <= power
        # 2. SOC-Beschränkungen: 0 <= soc_t <= capacity
        # 3. SOC-Dynamik: soc_{t+1} = soc_t + dt * (η_ch * c_t - d_t / η_dis) für t=0..T-1
        # 4. Initial SOC: soc_0 = initial_soc
        # 5. End SOC: soc_T = soc_0 (zyklisch, wenn enforce_end_soc_zero)
        
        # Baue Constraints effizient auf (O(T) statt O(T²)) - COO Format
        constraint_rows = []
        constraint_cols = []
        constraint_data = []
        constraint_l = []
        constraint_u = []
        constraint_idx = 0
        
        # 1. Power-Constraints: 0 <= c_t <= power, 0 <= d_t <= power
        for t in range(T):
            c_idx = t
            d_idx = T + t
            # c_t >= 0, <= power
            constraint_rows.append(constraint_idx)
            constraint_cols.append(c_idx)
            constraint_data.append(1.0)
            constraint_l.append(0.0)
            constraint_u.append(power)
            constraint_idx += 1
            # d_t >= 0, <= power
            constraint_rows.append(constraint_idx)
            constraint_cols.append(d_idx)
            constraint_data.append(1.0)
            constraint_l.append(0.0)
            constraint_u.append(power)
            constraint_idx += 1
        
        # 2. SOC-Bounds: 0 <= soc_t <= capacity
        for t in range(T + 1):
            soc_idx = 2 * T + t
            constraint_rows.append(constraint_idx)
            constraint_cols.append(soc_idx)
            constraint_data.append(1.0)
            constraint_l.append(0.0)
            constraint_u.append(self.capacity)
            constraint_idx += 1
        
        # 3. SOC-Dynamik: soc_{t+1} = soc_t + dt * (η_ch * c_t - d_t / η_dis)
        # Umgeformt: soc_{t+1} - soc_t - dt*η_ch*c_t + dt/η_dis*d_t = 0
        for t in range(T):
            c_idx = t
            d_idx = T + t
            soc_t_idx = 2 * T + t
            soc_tp1_idx = 2 * T + t + 1
            
            constraint_rows.append(constraint_idx)
            constraint_cols.append(soc_tp1_idx)
            constraint_data.append(1.0)  # soc_{t+1}
            constraint_rows.append(constraint_idx)
            constraint_cols.append(soc_t_idx)
            constraint_data.append(-1.0)  # -soc_t
            constraint_rows.append(constraint_idx)
            constraint_cols.append(c_idx)
            constraint_data.append(-self.eta_ch * dt)  # -dt*η_ch*c_t
            constraint_rows.append(constraint_idx)
            constraint_cols.append(d_idx)
            constraint_data.append(dt / self.eta_dis)  # dt/η_dis*d_t
            
            constraint_l.append(0.0)
            constraint_u.append(0.0)  # Gleichheitsconstraint
            constraint_idx += 1
        
        # 4. Initial SOC: soc_0 = initial_soc
        soc_0_idx = 2 * T
        constraint_rows.append(constraint_idx)
        constraint_cols.append(soc_0_idx)
        constraint_data.append(1.0)
        constraint_l.append(initial_soc)
        constraint_u.append(initial_soc)
        constraint_idx += 1
        
        # 5. End SOC: soc_T = soc_0 (zyklisch) - effizient als COO aufbauen
        if enforce_end_soc_zero:
            soc_T_idx = 2 * T + T
            constraint_rows.append(constraint_idx)
            constraint_cols.append(soc_T_idx)
            constraint_data.append(1.0)  # soc_T
            constraint_rows.append(constraint_idx)
            constraint_cols.append(soc_0_idx)
            constraint_data.append(-1.0)  # -soc_0
            constraint_l.append(0.0)
            constraint_u.append(0.0)  # Gleichheitsconstraint
            constraint_idx += 1
        
        # Baue Constraint-Matrix aus COO-Format (effizient)
        A = sparse.csc_matrix((constraint_data, (constraint_rows, constraint_cols)), 
                              shape=(constraint_idx, n_vars))
        l = np.array(constraint_l)
        u = np.array(constraint_u)
        
        try:
            solver = osqp.OSQP()
            solver.setup(
                P=self.P, q=q, A=A, l=l, u=u,
                verbose=False,
                # Robustere Konvergenz fuer grosse Jahresprobleme
                eps_abs=1e-3,
                eps_rel=1e-3,
                max_iter=100000,
                polish=True,
                adaptive_rho=True,
                rho=0.1,
                scaling=10,
                warm_start=True,
            )
            # OSQP löst das Problem (keine Progress Bar möglich, da OSQP keine Callbacks unterstützt)
            res = solver.solve()
            
            # Prüfe Konvergenz
            if res.info.status_val != 1:  # 1 = OSQP_SOLVED
                tqdm.write(f"  Warnung: Optimierung nicht optimal gelöst (Status: {res.info.status})")
        except Exception as e:
            tqdm.write(f"  Fehler: {e}")
            return np.zeros(T), np.zeros(T), np.zeros(T + 1), 0.0, 0.0, 0.0
        
        # Results extrahieren
        x_opt = res.x if res.x is not None else np.zeros(n_vars)
        # Variablenreihenfolge: [c_0..c_{T-1}, d_0..d_{T-1}, soc_0..soc_T]
        charge = x_opt[0:T]  # c_t
        discharge = x_opt[T:2*T]  # d_t
        soc = x_opt[2*T:2*T+T+1]  # soc_t (bereits aus Optimierung)
        
        # Erlösberechnung mit dt
        # Netto-Leistung: d_t - c_t (positiv = Entladung/Verkauf)
        net_power = discharge - charge
        cashflow = dt * net_power * prices
        # Price Impact: auf Netto-Leistung (Markteinfluss durch Handelsvolumen)
        impact = alpha * dt * (net_power ** 2)
        # Hurdle: Durchsatz (Laden + Entladen)
        ops = self.cfg.hurdle_rate * dt * (charge + discharge)
        revenue = float(np.sum(cashflow - impact - ops))
        
        # Zyklen: Durchsatz / (2 * Kapazität)
        throughput = dt * np.sum(charge + discharge)
        cycles = float(throughput / (2.0 * self.capacity)) if self.capacity > 0 else 0.0
        final_soc = float(soc[-1])
        
        return discharge, charge, soc, revenue, cycles, final_soc


# ==========================================
# 4. WORKER-FUNKTION FÜR MULTIPROCESSING
# ==========================================
def optimize_scenario_worker(args):
    """
    Worker-Funktion für Multiprocessing: Rechnet ein komplettes Szenario durch.
    
    Args:
        args: Tuple von (scenario_name, capacity, year_data, years, cfg_dict, dt_hours)
        - year_data: Dictionary mit Jahr -> DataFrame (pandas DataFrames sind picklable)
        - cfg_dict: SimulationConfig als Dictionary
    
    Returns:
        Dictionary mit Ergebnissen des Szenarios
    """
    scenario_name, capacity, year_data, years, cfg_dict, dt_hours = args
    
    # Rekonstruiere Config aus Dictionary
    cfg = SimulationConfig(**cfg_dict)
    
    # Optimizer für dieses Szenario
    opt = BessOptimizerQuadratic(capacity, cfg, dt_hours=dt_hours)
    
    all_decisions = []
    all_timestamps = []
    current_soc = 0.0  # Kontinuierlicher SoC über alle Jahre
    total_revenue = 0.0
    total_cycles = 0.0
    
    # Jahre sequentiell durchrechnen (SOC-Übergang muss kontinuierlich sein)
    available_years = [y for y in years if y in year_data]
    for year in available_years:
        year_df = year_data[year]
        prices = year_df['price_da'].values

        # End-SOC-Constraint nur anwenden, wenn konfiguriert (typisch: nur am Ende des gesamten Horizonts)
        is_last_year = (year == years[-1])
        enforce_end = bool(cfg.enforce_end_soc_zero) and is_last_year

        # Robustness: SOC-Drift begrenzen
        current_soc = float(np.clip(current_soc, 0.0, capacity)) if capacity > 0 else 0.0

        # Löse Jahr
        discharge, charge, soc, revenue, cycles, final_soc = opt.solve_year(
            prices,
            enforce_end_soc_zero=enforce_end,
            initial_soc=current_soc
        )

        # Speichern
        net_discharge = discharge - charge
        all_decisions.extend(net_discharge)
        all_timestamps.extend(year_df.index)

        total_revenue += revenue
        total_cycles += cycles
        current_soc = final_soc
    
    return {
        'scenario_name': scenario_name,
        'capacity': capacity,
        'all_decisions': all_decisions,
        'all_timestamps': all_timestamps,
        'total_revenue': total_revenue,
        'total_cycles': total_cycles,
        'final_soc': current_soc
    }


# ==========================================
# 5. HAUPTSIMULATION (PARALLELISIERT)
# ==========================================
def run_simulation():
    # 1. Daten laden
    df = load_market_data()
    
    # 2. Zeitauflösung erkennen (robust gegen DST und Datenlücken)
    if len(df) > 1:
        # Verwende Median der Zeitdifferenzen (robust gegen Ausreißer und DST)
        time_diffs = df.index.to_series().diff().dropna()
        if len(time_diffs) > 0:
            median_diff = time_diffs.median()
            dt_hours = median_diff.total_seconds() / 3600.0
            print(f"Erkannte Zeitauflösung: {dt_hours} Stunden (Median: {median_diff})")
        else:
            dt_hours = 1.0
            print("Warnung: Konnte Zeitauflösung nicht erkennen, verwende 1.0 Stunden")
    else:
        dt_hours = 1.0
        print("Warnung: Konnte Zeitauflösung nicht erkennen, verwende 1.0 Stunden")
    
    # 3. Config verwenden (aus Dataclass am Anfang definiert)
    cfg = SimulationConfig()
    
    # Sanity check: Effizienzen muessen gesetzt sein (Variante A: aus roundtrip_efficiency abgeleitet)
    if cfg.efficiency_charge is None or cfg.efficiency_discharge is None:
        raise RuntimeError("Effizienzwerte sind None. Erwartet: Ableitung aus roundtrip_efficiency in __post_init__.")
    
    if cfg.debug:
        print("DEBUG eta_ch, eta_dis, eta_rt:", cfg.efficiency_charge, cfg.efficiency_discharge, cfg.roundtrip_efficiency)
    
    print(
        f"Config: C-Rate={cfg.c_rate}, Slope={cfg.slope}, "
        f"Effizienz Lade={cfg.efficiency_charge:.4f}, Entlade={cfg.efficiency_discharge:.4f}, "
        f"Rundreise={cfg.roundtrip_efficiency:.4f}"
    )
    
    # 4. In einzelne Jahre chunken (2019-2024)
    years = [2019, 2020, 2021, 2022, 2023, 2024]
    year_data = {}
    
    print("Chunke Daten in einzelne Jahre...")
    for year in years:
        mask = df.index.year == year
        year_df = df[mask]
        if len(year_df) > 0:
            year_data[year] = year_df
            print(f"  {year}: {len(year_df)} Zeitschritte")
    
    # 4. Szenarien
    scenarios = {
        'Ex': 5,
        'S': 10,
        'M': 50,
        'L': 100,
        'XL': 1000
    }
    
    # 5. Parallele Optimierung aller Szenarien
    aggregated_results = {}
    revenue_summary = []
    
    print(f"\nStarte Multi-Jahr Simulation (2019-2024, 5 Szenarien) - PARALLELISIERT...")
    print(f"Verwende {mp.cpu_count()} CPU-Kerne")
    print(f"{'Szenario':<10} | {'Gesamterlös (€)':<20} | {'Zyklen':<15}")
    print("-" * 50)
    
    # Vorbereitung der Arguments für Worker
    scenario_items = list(scenarios.items())
    
    # Config als Dictionary (serialisierbar - dataclass ist nicht direkt picklable)
    cfg_dict = {
        'roundtrip_efficiency': cfg.roundtrip_efficiency,
        'efficiency_charge': None,
        'efficiency_discharge': None,
        'c_rate': cfg.c_rate,
        'hurdle_rate': cfg.hurdle_rate,
        'slope': cfg.slope,
        'enforce_end_soc_zero': cfg.enforce_end_soc_zero
    }
    
    # year_data kann direkt übergeben werden (pandas DataFrames sind picklable)
    pool_args = [
        (scenario_name, capacity, year_data, years, cfg_dict, dt_hours)
        for scenario_name, capacity in scenario_items
    ]
    
    # Parallele Ausführung mit Progress Bar
    num_workers = min(len(scenario_items), mp.cpu_count())
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Alle Tasks einreichen
        future_to_scenario = {
            executor.submit(optimize_scenario_worker, args): args[0] 
            for args in pool_args
        }
        
        # Progress Bar für abgeschlossene Szenarien
        with tqdm(total=len(scenario_items), desc="Szenarien", unit="Szenario") as pbar:
            for future in as_completed(future_to_scenario):
                scenario_name = future_to_scenario[future]
                try:
                    result = future.result()
                    
                    # Ergebnisse speichern
                    aggregated_results[result['scenario_name']] = result['all_decisions']
                    revenue_summary.append({
                        'Szenario': result['scenario_name'],
                        'Kapazität (MWh)': result['capacity'],
                        'Gesamterlös (€)': result['total_revenue'],
                        'Zyklen': result['total_cycles'],
                        'Finale_SoC': result['final_soc']
                    })
                    
                    # Ergebnis ausgeben
                    tqdm.write(f"{result['scenario_name']:<10} | "
                             f"{result['total_revenue']:>18,.0f}   | "
                             f"{result['total_cycles']:>13.2f}")
                    
                except Exception as e:
                    tqdm.write(f"Fehler bei Szenario {scenario_name}: {e}")
                finally:
                    pbar.update(1)
    
    # Sortiere revenue_summary nach Szenario-Reihenfolge
    scenario_order = {name: idx for idx, (name, _) in enumerate(scenario_items)}
    revenue_summary.sort(key=lambda x: scenario_order.get(x['Szenario'], 999))
    
    # 6. CSV mit stündlichen Entscheidungen speichern
    # Timestamps aus dem ersten Szenario nehmen (alle haben dieselben)
    print("\nSpeichere stündliche Entscheidungen...", end=' ', flush=True)
    # Timestamps müssen aus year_data rekonstruiert werden
    all_timestamps = []
    for year in sorted(year_data.keys()):
        all_timestamps.extend(year_data[year].index)
    
    output_df = pd.DataFrame({
        'timestamp': all_timestamps,
        **{scenario: aggregated_results[scenario] for scenario in scenarios.keys()}
    })
    output_df.set_index('timestamp', inplace=True)
    
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulation_decisions_multiyear.csv')
    output_df.to_csv(csv_path)
    print(f"✓")
    print(f"Datei: {csv_path}")
    
    # 7. Arbitrage-Tabelle speichern
    print("Speichere Arbitrage-Zusammenfassung...", end=' ', flush=True)
    revenue_df = pd.DataFrame(revenue_summary)
    table_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'arbitrage_summary_multiyear.csv')
    revenue_df.to_csv(table_path, index=False)
    print(f"✓")
    print(f"Datei: {table_path}")
    
    # 8. Tabelle in Konsole ausgeben
    print("\n" + "="*75)
    print("MULTI-JAHR ARBITRAGE-ZUSAMMENFASSUNG (2019-2024)")
    print("="*75)
    print(revenue_df.to_string(index=False))
    print("="*75)
    
    # 9. Plots
    plot_results_multiyear(revenue_summary, aggregated_results, all_timestamps, year_data, cfg)

def plot_results_multiyear(results, aggregated_results, timestamps, year_data, cfg):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot 1: Revenue by capacity
    scenarios = [r['Szenario'] for r in results]
    revenues = [r['Gesamterlös (€)'] / 1e6 for r in results]
    
    axes[0].bar(scenarios, revenues, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_title('Gesamterlös nach Batteriegröße (Multi-Jahr 2019-2024)')
    axes[0].set_xlabel('Szenario')
    axes[0].set_ylabel('Erlös (Mio. €)')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Full timeline (alle 6 Jahre)
    # Jahre haben unterschiedliche Längen, daher concatenate statt array
    prices = np.concatenate([year_data[year]['price_da'].values for year in sorted(year_data.keys())])
    
    ax2a = axes[1]
    ax2b = ax2a.twinx()
    
    ax2a.plot(prices, color='grey', linewidth=1.5, label='Marktpreis', zorder=5, alpha=0.8)
    
    # Plot für jedes Szenario
    colors = {'Ex': 'blue', 'S': 'green', 'M': 'orange', 'L': 'red', 'XL': 'purple'}
    for scenario, color in colors.items():
        if scenario in aggregated_results:
            ax2b.plot(aggregated_results[scenario], alpha=0.4, label=scenario, color=color, linewidth=0.8)
    
    ax2a.set_xlabel('Stunde (6 Jahre: 2019-2024)')
    ax2a.set_ylabel('Marktpreis (€/MWh)', color='grey')
    ax2b.set_ylabel('Netto-Entladung (MW)')
    ax2a.set_title('6-Jahres Timeline: Marktpreis vs. Handelsverhalten')
    ax2a.tick_params(axis='y', labelcolor='grey')
    ax2a.grid(True, alpha=0.3)
    
    lines1, labels1 = ax2a.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2a.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()
