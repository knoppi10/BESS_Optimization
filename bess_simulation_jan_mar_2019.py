import pandas as pd
import numpy as np
import osqp
from scipy import sparse
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. KONFIGURATION
# ==========================================
@dataclass
class SimulationConfig:
    efficiency_charge: float = 0.92
    efficiency_discharge: float = 0.92
    c_rate: float = 0.2 # HIER ÄNDERN: C-Rate
    hurdle_rate: float = 5.0  # €/MWh Opportunitätskosten
    slope: float = 0.05  # HIER ÄNDERN: Slope (Price Impact)
    enforce_end_soc_zero: bool = True

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
        df.columns = df.columns.str.strip()
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
    def __init__(self, prices, capacity_mwh, config: SimulationConfig):
        self.prices = prices.values.copy()
        self.T = len(prices)
        self.capacity = capacity_mwh
        self.power = capacity_mwh * config.c_rate
        self.cfg = config

    def solve_osqp(self):
        T = self.T
        alpha = self.cfg.slope
        power = self.capacity * self.cfg.c_rate
        
        P = sparse.diags([2.0 * alpha] * T, format='csc')
        q = -self.prices.copy()
        
        row = np.repeat(np.arange(T), np.arange(1, T + 1))
        col = np.concatenate([np.arange(i + 1) for i in range(T)])
        data = np.ones(len(row))
        L = sparse.csc_matrix((data, (row, col)), shape=(T, T))
        
        I = sparse.eye(T, format='csc')
        A = sparse.vstack([-L, L, I], format='csc')

        l = np.hstack([-np.inf * np.ones(T), np.zeros(T), -power * np.ones(T)])
        u = np.hstack([np.zeros(T), self.capacity * np.ones(T), power * np.ones(T)])
        
        if self.cfg.enforce_end_soc_zero:
            A_end = sparse.csc_matrix(np.ones((1, T)))
            A = sparse.vstack([A, A_end], format='csc')
            l = np.hstack([l, 0.0])
            u = np.hstack([u, 0.0])
        
        try:
            solver = osqp.OSQP()
            solver.setup(P=P, q=q, A=A, l=l, u=u, verbose=False, eps_abs=1e-3, eps_rel=1e-3, max_iter=4000)
            res = solver.solve()
        except Exception as e:
            print(f"  Fehler: {e}")
            return np.zeros(T), np.zeros(T), np.zeros(T + 1), 0.0, 0.0
        
        x_opt = res.x if res.x is not None else np.zeros(T)
        soc = np.concatenate([[0], np.cumsum(x_opt)])
        
        cashflow = x_opt * self.prices
        impact = alpha * (x_opt ** 2)
        ops = np.abs(x_opt) * self.cfg.hurdle_rate
        revenue = float(np.sum(cashflow - impact - ops))
        
        discharge = np.maximum(x_opt, 0)
        charge = np.maximum(-x_opt, 0)
        cycles = float(np.sum(np.abs(x_opt)) / (2.0 * self.capacity)) if self.capacity > 0 else 0.0
        
        return discharge, charge, soc, revenue, cycles

# ==========================================
# 4. REFAKTOISIERTE SIMULATIONSFUNKTIONEN
# ==========================================

def run_single_simulation(cfg: SimulationConfig, df: pd.DataFrame, verbose=True):
    """
    Führt die Simulation für eine einzelne Konfiguration und alle Kapazitätsszenarien durch.
    """
    if verbose:
        print(f"\nConfig: C-Rate={cfg.c_rate:.2f}, Slope={cfg.slope:.2f}, Hurdle Rate={cfg.hurdle_rate:.2f} €/MWh")
    
    scenarios = {'Ex': 5, 'S': 10, 'M': 50, 'L': 100, 'XL': 1000}
    revenue_summary = []
    
    if verbose:
        print(f"{ 'Szenario':<10} | {'Kapazität (MWh)':<20} | {'Erlös (€)':<15} | {'Zyklen':<10} | {'€/MWh Bat':<15}")
        print("-" * 85)
    
    for scenario_name, capacity in scenarios.items():
        opt = BessOptimizerQuadratic(df['price_da'], capacity, cfg)
        _, _, _, revenue, cycles = opt.solve_osqp()
        
        erloes_pro_mwh = revenue / capacity if capacity > 0 else 0.0
        
        revenue_summary.append({
            'Szenario': scenario_name,
            'Kapazität (MWh)': capacity,
            'Erlös (€)': revenue,
            'Zyklen': cycles,
            '€/MWh Bat': erloes_pro_mwh
        })
        
        if verbose:
            print(f"{scenario_name:<10} | {capacity:<20} | {revenue:>13,.0f}   | {cycles:>8.2f}   | {erloes_pro_mwh:>13,.0f}")
        
        del opt
        
    return revenue_summary

def run_parameter_study():
    """
    Führt eine Parameterstudie durch, indem C-Rate, Slope und Hurdle Rate variiert werden.
    Vergleicht die Ergebnisse mit einer Baseline-Konfiguration.
    """
    # 1. Daten laden und vorbereiten
    df = load_market_data()
    df = df[(df.index.year == 2019) & (df.index.month >= 1) & (df.index.month <= 3)]
    print(f"Gefilterte Daten: {df.index[0]} bis {df.index[-1]} ({len(df)} Stunden)")

    # 2. Parameter-Variationen definieren
    base_cfg = SimulationConfig()
    
    configs_to_test = [
        ("Baseline", base_cfg),
        ("C-Rate +0.05", SimulationConfig(c_rate=base_cfg.c_rate + 0.05, slope=base_cfg.slope, hurdle_rate=base_cfg.hurdle_rate)),
        ("C-Rate +0.10", SimulationConfig(c_rate=base_cfg.c_rate + 0.10, slope=base_cfg.slope, hurdle_rate=base_cfg.hurdle_rate)),
        ("Slope +0.05", SimulationConfig(c_rate=base_cfg.c_rate, slope=base_cfg.slope + 0.05, hurdle_rate=base_cfg.hurdle_rate)),
        ("Slope +0.10", SimulationConfig(c_rate=base_cfg.c_rate, slope=base_cfg.slope + 0.10, hurdle_rate=base_cfg.hurdle_rate)),
        ("Hurdle +0.05", SimulationConfig(c_rate=base_cfg.c_rate, slope=base_cfg.slope, hurdle_rate=base_cfg.hurdle_rate + 0.05)),
        ("Hurdle +0.10", SimulationConfig(c_rate=base_cfg.c_rate, slope=base_cfg.slope, hurdle_rate=base_cfg.hurdle_rate + 0.10)),
    ]
    
    all_study_results = []
    
    print("\nStarte Parameterstudie...")
    for study_name, cfg in configs_to_test:
        print(f"\n===== LAUFE STUDIE: {study_name.upper()} =====")
        # Für die Studie selbst weniger Output in der Konsole
        revenue_summary = run_single_simulation(cfg, df, verbose=False) 
        
        # Repräsentatives Szenario 'L' (100 MWh) für den Vergleich heranziehen
        l_scenario_result = next((item for item in revenue_summary if item['Szenario'] == 'L'), None)
        l_scenario_revenue = l_scenario_result['Erlös (€)'] if l_scenario_result else 0.0

        all_study_results.append({
            'Studie': study_name,
            'C-Rate': cfg.c_rate,
            'Slope': cfg.slope,
            'Hurdle Rate': cfg.hurdle_rate,
            'Erlös (Szenario L, €)': l_scenario_revenue,
        })

    # 3. Ergebnisse zusammenfassen und ausgeben
    results_df = pd.DataFrame(all_study_results)
    
    baseline_revenue = results_df.loc[results_df['Studie'] == 'Baseline', 'Erlös (Szenario L, €)'].iloc[0]
    if baseline_revenue != 0:
        results_df['Impact vs Baseline (%)'] = ((results_df['Erlös (Szenario L, €)'] - baseline_revenue) / baseline_revenue) * 100
    else:
        results_df['Impact vs Baseline (%)'] = 0.0

    print("\n\n" + "="*80)
    print("ERGEBNISSE DER PARAMETERSTUDIE (Vergleich mit Szenario 'L', 100 MWh)")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)

    # 4. Ergebnisse in CSV speichern
    study_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parameter_study_summary.csv')
    results_df.to_csv(study_path, index=False, float_format='%.2f')
    print(f"\nStudien-Ergebnisse gespeichert in: {study_path}")

def plot_results(results, results_dict, df, cfg):
    """Platzhalter für Plots, wird in der Studie nicht direkt verwendet."""
    pass # In dieser Version nicht aktiv genutzt, um die Ausgabe übersichtlich zu halten.

if __name__ == "__main__":
    run_parameter_study()