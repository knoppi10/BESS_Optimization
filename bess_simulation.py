# @CELIA BESS Simulation Script
# This script simulates the operation of a Battery Energy Storage System (BESS)
# OTIMIZER muss noch angepasst werden
# Ergebnis soll ein Vektor mit stündlichen Entscheidungen sein


import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy.special import expit
import os

# ==========================================
# 1. KONFIGURATION
# ==========================================
@dataclass
class SimulationConfig:
    # Batterie-Parameter
    efficiency_charge: float = 0.92      # ca. 85% Roundtrip Effizienz (Wurzel aus 0.85)
    efficiency_discharge: float = 0.92
    c_rate: float = 0.25                 # 4-Stunden-Batterie (Leistung = 0.25 * Kapazität)
    hurdle_rate: float = 5.0             # €/MWh Opportunitätskosten/Verschleiß
    
    # Fundamentalmodell Parameter (Residuallast -> Slope)
    # Wir bilden die "Merit Order" als S-Kurve (Sigmoid) nach.
    slope_min: float = 0.001   # €/MW (Flacher Bereich: Viel Wind/Solar, geringer Preiseinfluss)
    slope_max: float = 0.05    # €/MW (Steiler Bereich: Gaskraftwerke/Knappheit, hoher Preiseinfluss)
    
    # Wendepunkt der Kurve in MW
    # Ab welcher Residuallast fangen die Preise an stark zu steigen?
    # Für DE ca. 40 GW Residuallast
    residual_inflection_point: float = 40000 
    
    # Steilheit des Anstiegs der S-Kurve
    residual_sensitivity: float = 0.0001
    # Enforce end-of-horizon SOC = 0 to prevent artificial end-of-horizon arbitrage
    enforce_end_soc_zero: bool = True

# ==========================================
# 2. HELPER & DATEN
# ==========================================
def load_market_data(filename='market_data_2019_2025.csv'):
    """
    Lädt die echten Daten. Bricht ab, falls Datei nicht gefunden wird.
    """
    # Pfad relativ zum Skript auflösen
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)

    if os.path.exists(filepath):
        print(f"Lade echte Daten aus {filepath}...")
        # Read CSV then normalize column names and parse timestamp robustly
        df = pd.read_csv(filepath)
        # Strip whitespace from column names (some CSVs contain padded headers)
        df.columns = df.columns.str.strip()
        # Parse timestamp column (robust to header variations)
        if 'timestamp' in df.columns:
            # Parse as UTC to handle mixed offsets, then convert
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df = df.set_index('timestamp')
        else:
            # Fallback: try first column as timestamp
            first_col = df.columns[0]
            try:
                df[first_col] = pd.to_datetime(df[first_col], utc=True)
                df = df.set_index(first_col)
                df.index.name = 'timestamp'
            except Exception:
                raise ValueError(f"Keine gültige 'timestamp' Spalte in '{filepath}' gefunden.")
        # Ensure DatetimeIndex and convert to Europe/Berlin
        try:
            df.index = pd.DatetimeIndex(df.index)
        except Exception:
            raise ValueError("Index konnte nicht in DatetimeIndex umgewandelt werden.")
        # Wenn tz-naiv, lokalisiere nach Europe/Berlin; sonst konvertiere
        if df.index.tz is None:
            df.index = df.index.tz_localize('Europe/Berlin')
        else:
            df.index = df.index.tz_convert('Europe/Berlin')
        return df
    else:
        # KEINE Dummy-Daten mehr -> Harter Fehler
        raise FileNotFoundError(f"Datei '{filepath}' nicht gefunden! Bitte führe zuerst data_fetch.py aus.")

def _sigmoid_func(x, slope_min, slope_max, inflection, sensitivity):
    """Die Sigmoid-Funktion, die wir fitten wollen. Nutzt numerisch stabile 'expit'."""
    z = sensitivity * (x - inflection)
    sigmoid = expit(z)
    return slope_min + (slope_max - slope_min) * sigmoid

def get_season(month):
    """Ordnet einen Monat einer Jahreszeit zu."""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Frühling'
    elif month in [6, 7, 8]:
        return 'Sommer'
    else: # 9, 10, 11
        return 'Herbst'

def get_seasonal_slope_parameters(df: pd.DataFrame):
    """Ermittelt für jede Jahreszeit die Parameter der Slope-Kurve (aggregiert über alle Jahre)."""
    print("Ermittle saisonale Slope-Parameter aus den Daten...")
    df['season'] = df.index.month.map(get_season)
    seasonal_params = {}
    
    for season in ['Winter', 'Frühling', 'Sommer', 'Herbst']:
        season_df = df[df['season'] == season].copy()
        
        # Binning der Daten
        min_load, max_load = season_df['residual_load'].min(), season_df['residual_load'].max()
        bins = np.arange(min_load, max_load + 2000, 2000)
        season_df['load_bin'] = pd.cut(season_df['residual_load'], bins=bins, right=False)
        
        empirical_points = []
        for bin_interval, group in season_df.groupby('load_bin'):
            if len(group) < 30: continue
            regression = linregress(x=group['residual_load'], y=group['price_da'])
            empirical_points.append({'residual_load': bin_interval.mid, 'slope': regression.slope})
        
        if not empirical_points: continue
        
        empirical_df = pd.DataFrame(empirical_points)
        
        # Constrain parameters to reasonable ranges to avoid pathological fits
        lower_bounds = [0.0, 0.0, 0.0, 1e-12]
        upper_bounds = [0.1, 1.0, 1e6, 1.0]
        try:
            popt, _ = curve_fit(_sigmoid_func, empirical_df['residual_load'], empirical_df['slope'], 
                                p0=[0.001, 0.05, 40000, 0.0001], bounds=(lower_bounds, upper_bounds), maxfev=10000)
            # Validate fitted params
            if not np.isfinite(popt).all() or popt[1] <= 0 or popt[3] <= 0:
                print(f"  - Warnung: Ungültiger Fit für {season}, verworfen")
                seasonal_params[season] = None
            else:
                seasonal_params[season] = popt
                print(f"  - Saison {season}: Inflection ~{popt[2]:.0f} MW, Max Slope {popt[1]:.4f}")
        except RuntimeError:
            print(f"  - Warnung: Fit fehlgeschlagen für {season}")
            seasonal_params[season] = None
            
    return seasonal_params

def calculate_seasonal_slopes(residual_load: pd.Series, seasonal_params: dict):
    """
    Berechnet die stündlichen Slopes basierend auf den saisonalen Parametern.
    """
    slopes = pd.Series(index=residual_load.index, dtype=float)
    seasons_map = residual_load.index.month.map(get_season)
    
    for season, params in seasonal_params.items():
        if params is None: continue # Skip if fitting failed
        
        season_mask = (seasons_map == season)
        if not season_mask.any(): continue
        
        x = residual_load[season_mask].values
        slopes.loc[season_mask] = _sigmoid_func(x, *params)

    # Lücken füllen
    if slopes.isna().any():
        slopes = slopes.ffill().bfill()

    # Prevent negative or pathologically large slopes (which break convexity assumptions)
    slopes = slopes.clip(lower=0.0, upper=1.0)

    return slopes

# ==========================================
# 3. OPTIMIERUNGS-ENGINE (CVXPY)
# ==========================================
class BessOptimizer:
    def __init__(self, prices, capacity_mwh, config: SimulationConfig, hourly_slopes):
        self.prices = prices.values
        self.T = len(prices)
        self.capacity = capacity_mwh
        self.power = capacity_mwh * config.c_rate
        self.cfg = config
        # Ensure numpy arrays and correct shapes for CVXPY operations
        self.slopes = np.asarray(hourly_slopes)
        if self.slopes.shape[0] != self.T:
            raise ValueError(f"hourly_slopes length ({self.slopes.shape[0]}) does not match number of timesteps ({self.T})")
        
        # Speicher für Ergebnisse
        self.charge = None
        self.discharge = None
        self.soc = None
        self.revenue = 0
        self.cycles = 0

    def _get_common_constraints(self, c, d, s):
        """Standard BESS Constraints für beide Modelle"""
        constraints = [s[0] == 0] # Start leer
        for t in range(self.T):
            # Speicherbilanz: SOC_new = SOC_old + Charge*Eff - Discharge/Eff
            constraints.append(s[t+1] == s[t] + c[t]*self.cfg.efficiency_charge - d[t]/self.cfg.efficiency_discharge)
            # Limits (Leistung und Kapazität)
            constraints.append(c[t] >= 0); constraints.append(c[t] <= self.power)
            constraints.append(d[t] >= 0); constraints.append(d[t] <= self.power)
            constraints.append(s[t+1] >= 0); constraints.append(s[t+1] <= self.capacity)
        # Optional: erzwinge Endzustand = 0 zur Vermeidung von Arbitrage über das Ende des Horizons
        if self.cfg.enforce_end_soc_zero:
            constraints.append(s[self.T] == 0)
        return constraints

    def solve_exogenous(self):
        """
        Price Taker Modell (Lineare Programmierung)
        Nimmt historische Preise als gegeben an (ignoriert Markteinfluss).
        """
        c = cp.Variable(self.T)
        d = cp.Variable(self.T)
        s = cp.Variable(self.T + 1)
        
        # Zielfunktion: Maximiere (Entladen - Laden)*Preis - HurdleCosts
        revenue = cp.sum(cp.multiply(d - c, self.prices))
        costs = cp.sum((d + c) * self.cfg.hurdle_rate)
        
        prob = cp.Problem(cp.Maximize(revenue - costs), self._get_common_constraints(c, d, s))
        prob.solve(solver=cp.ECOS) # ECOS ist schnell und robust für LP
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Warnung: Solver gab Status {prob.status} zurück (Exogenes Modell)")

        self._save_results(c, d, s)
        self._calc_financials(self.prices) # Erlös auf Basis der historischen Preise

    def solve_endogenous(self):
        """
        Price Maker Modell (Quadratische Programmierung)
        Antizipiert Preisänderung: P_real = P_hist + Slope * Net_Load
        Das Modell "weiß", dass aggressives Handeln den Preis verschlechtert.
        """
        c = cp.Variable(self.T)
        d = cp.Variable(self.T)
        s = cp.Variable(self.T + 1)
        
        # Basis-Umsatz (linear)
        base_rev = cp.sum(cp.multiply(d - c, self.prices))
        
        # Preis-Impact-Strafe (Quadratisch)
        # Formel: Slope[t] * (Discharge[t] - Charge[t])^2
        net_discharge = d - c
        # cp.multiply erlaubt elementweise Multiplikation mit dem stündlichen Slope-Vektor (Fundamentalmodell)
        impact_penalty = cp.sum(cp.multiply(self.slopes, cp.square(net_discharge)))
        
        costs = cp.sum((d + c) * self.cfg.hurdle_rate)
        
        # Wir maximieren (Umsatz - Strafe - Kosten)
        # Das ist ein konkaves Problem (leicht lösbar), da wir den quadratischen Term abziehen
        prob = cp.Problem(cp.Maximize(base_rev - impact_penalty - costs), self._get_common_constraints(c, d, s))
        try:
            prob.solve(solver=cp.OSQP) # OSQP ist spezialisiert auf Quadratische Programme
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                print(f"Warnung: Solver gab Status {prob.status} zurück (Endogenes Modell)")
        except Exception as e:
            print(f"Fehler beim Lösen des endogenen Problems: {e}")
            # Leave results empty and return to allow the simulation to continue
            self.charge = np.zeros(self.T)
            self.discharge = np.zeros(self.T)
            self.soc = np.zeros(self.T + 1)
            self.revenue = float('nan')
            self.cycles = 0
            return
        
        self._save_results(c, d, s)
        
        # WICHTIG: Erlösberechnung mit den NEUEN, durch die Batterie veränderten Preisen (Feedback Loop)
        net_flow = self.charge - self.discharge # Positiv = Laden (Mehr Nachfrage) -> Preis steigt
        final_prices = self.prices + (net_flow * self.slopes)
        self._calc_financials(final_prices)

    def _save_results(self, c, d, s):
        self.charge = c.value
        self.discharge = d.value
        self.soc = s.value
        # Rauschen filtern (Solver Präzision)
        if self.charge is not None:
            self.charge[self.charge < 1e-4] = 0
            self.discharge[self.discharge < 1e-4] = 0

    def _calc_financials(self, calc_prices):
        cashflow = (self.discharge - self.charge) * calc_prices
        ops_cost = (self.discharge + self.charge) * self.cfg.hurdle_rate
        self.revenue = np.sum(cashflow - ops_cost)
        self.cycles = np.sum(self.discharge) / self.capacity if self.capacity > 0 else 0

# ==========================================
# 4. MAIN SIMULATION LOOP
# ==========================================
def run_simulation():
    # 1. Daten laden
    df = load_market_data()
    
    # 2. Config & Slope Berechnung
    cfg = SimulationConfig(c_rate=0.25) # 4h Batterie
    
    # Sicherstellen, dass Residuallast da ist (bei Dummy Daten automatisch, bei CSV prüfen)
    if 'residual_load' not in df.columns:
        print("Berechne Residuallast aus Load & Generation...")
        # Fallback, falls Spalten anders heißen oder fehlen
        if 'load' in df.columns and 'generation_renewable' in df.columns:
            df['residual_load'] = df['load'] - df['generation_renewable']
        else:
            raise ValueError("Fehlende Spalten: 'load' oder 'generation_renewable' nicht in den Daten gefunden.")
        
    # Saisonale Slope-Parameter fitten und stündliche Slopes berechnen
    seasonal_params = get_seasonal_slope_parameters(df)
    hourly_slopes = calculate_seasonal_slopes(df['residual_load'], seasonal_params)
    
    # 3. Szenarien (Batteriegrößen in MWh)
    capacities = [0, 100, 500, 1000, 2000, 5000]
    results = []
    
    print(f"\nStarte Simulation über {len(df)} Stunden...")
    print(f"{'Größe (MWh)':<15} | {'Exogen (€)':<15} | {'Endogen (€)':<15} | {'Delta (%)':<10}")
    print("-" * 65)
    
    plot_data = {} # Zum Speichern für Plots

    for cap in capacities:
        if cap == 0: continue
        
        # A. Exogen (Price Taker)
        opt_exo = BessOptimizer(df['price_da'], cap, cfg, hourly_slopes)
        opt_exo.solve_exogenous()
        
        # B. Endogen (Price Maker)
        opt_endo = BessOptimizer(df['price_da'], cap, cfg, hourly_slopes)
        opt_endo.solve_endogenous()
        
        # Ergebnisse speichern
        rev_ex = opt_exo.revenue
        rev_en = opt_endo.revenue
        delta = (rev_en - rev_ex) / rev_ex * 100 if rev_ex != 0 else 0
        
        results.append({
            'capacity_mwh': cap,
            'revenue_exogenous': rev_ex,
            'revenue_endogenous': rev_en,
            'delta_percent': delta,
            'cycles_exogenous': opt_exo.cycles,
            'cycles_endogenous': opt_endo.cycles
        })
        print(f"{cap:<15} | {rev_ex:,.0f}         | {rev_en:,.0f}         | {delta:.2f}%")
        
        # Daten für Plot speichern (vom größten Szenario)
        if cap == max(capacities):
            # Erste Woche (168h) slice für Zoom-In
            s = slice(0, 168)
            plot_data['prices'] = df['price_da'].values[s]
            plot_data['residual'] = df['residual_load'].values[s]
            plot_data['slopes'] = hourly_slopes[s]
            plot_data['ops_exo'] = (opt_exo.discharge - opt_exo.charge)[s]
            plot_data['ops_endo'] = (opt_endo.discharge - opt_endo.charge)[s]

    # Ergebnisse als CSV speichern
    results_df = pd.DataFrame(results)
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulation_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nErgebnisse wurden gespeichert in: {output_path}")

    # 4. Visualisierung
    plot_results(results, plot_data, cfg)

def plot_results(results, plot_data, cfg):
    # Wir reduzieren die Plots auf die wesentlichen Ergebnisse
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 3]})
    
    # Plot 1: Revenue Decay (Kannibalisierung)
    caps = [r['capacity_mwh'] for r in results]
    ex = [r['revenue_exogenous']/1e6 for r in results] # in Mio €
    en = [r['revenue_endogenous']/1e6 for r in results]
    
    ax1.plot(caps, ex, 'o--', label='Exogen (Price Taker)', color='grey', alpha=0.7)
    ax1.plot(caps, en, 'o-', label='Endogen (Price Maker)', color='#1f77b4', linewidth=2.5)
    ax1.set_title('Kannibalisierungseffekt: Umsatzrückgang bei steigender Kapazität')
    ax1.set_ylabel('Gesamterlös (Mio. €)')
    ax1.set_xlabel('Installierte Kapazität (MWh)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Operations (Vergleich Strategie)
    ax2.plot(plot_data['prices'], color='grey', alpha=0.3, label='Marktpreis (Historisch)')
    ax2_twin = ax2.twinx()
    
    # Exogen handelt aggressiv (Ignoriert Preisimpact)
    ax2_twin.fill_between(range(168), plot_data['ops_exo'], color='grey', alpha=0.3, label='Exogen (Aggressiv)')
    # Endogen handelt strategisch (Glättet Spitzen um Preis nicht zu zerstören)
    ax2_twin.plot(plot_data['ops_endo'], color='green', linewidth=2, label='Endogen (Strategisch)')
    
    ax2.set_ylabel('Preis (€/MWh)')
    ax2_twin.set_ylabel('Netto-Entladung (MW)')
    ax2.set_title('Handelsverhalten Woche 1: Strategische Zurückhaltung bei hohem Volumen')
    
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()