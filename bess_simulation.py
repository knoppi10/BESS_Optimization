import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from dataclasses import dataclass
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

# ==========================================
# 2. HELPER & DATEN
# ==========================================
def load_market_data(filepath='market_data_2019_2025.csv'):
    """
    Lädt die echten Daten. Falls nicht vorhanden, generiert es Dummy-Daten
    für Testzwecke (damit der Code immer läuft).
    """
    if os.path.exists(filepath):
        print(f"Lade echte Daten aus {filepath}...")
        df = pd.read_csv(filepath, parse_dates=['timestamp']).set_index('timestamp')
        return df
    else:
        print(f"WARNUNG: Datei '{filepath}' nicht gefunden. Generiere synthetische Testdaten...")
        return _generate_dummy_data()

def _generate_dummy_data():
    """Erzeugt realistische Zeitreihen für Preis, Last und Erneuerbare (Fallback)"""
    dates = pd.date_range(start='2025-01-01', end='2025-12-31 23:00', freq='h')
    n = len(dates)
    
    # Saisonales Muster + Tagesmuster + Zufall
    t = np.arange(n)
    
    # Last (Winter hoch, Tag hoch)
    load_base = 60000 + 10000 * np.cos(2*np.pi*t/(24*365)) # Jahr
    load_day = 10000 * np.cos(2*np.pi*t/24) # Tag
    load = load_base + load_day + np.random.normal(0, 2000, n)
    
    # Erneuerbare (Wind stochastisch, Solar tagsüber)
    wind = 15000 + 10000 * np.sin(2*np.pi*t/(24*10)) + np.random.normal(0, 5000, n)
    wind = np.maximum(wind, 0)
    
    solar = np.zeros(n)
    hours = dates.hour
    is_day = (hours >= 6) & (hours <= 20)
    solar[is_day] = 30000 * np.sin((hours[is_day]-6)/(14)*np.pi) * np.random.uniform(0.5, 1.0, np.sum(is_day))
    
    renewable = wind + solar
    residual = load - renewable
    
    # Preis korreliert exponentiell mit Residuallast
    price = 20 + 0.001 * residual + 2e-9 * np.exp(residual/2500)
    
    return pd.DataFrame({
        'price_da': price,
        'load': load,
        'generation_renewable': renewable,
        'residual_load': residual
    }, index=dates)

def calculate_fundamental_slopes(residual_load, cfg: SimulationConfig):
    """
    Kernfunktion des Fundamentalmodells:
    Wandelt Residuallast (MW) in Preissensitivität/Slope (€/MW) um.
    """
    x = residual_load.values
    
    # Logistische Funktion (Sigmoid)
    # Normiert x um den Wendepunkt.
    # Ergebnis ist ein Wert zwischen 0 und 1, der angibt, wie "angespannt" das Netz ist.
    sigmoid = 1 / (1 + np.exp(-cfg.residual_sensitivity * (x - cfg.residual_inflection_point)))
    
    # Skalieren auf den Bereich [min, max]
    slopes = cfg.slope_min + (cfg.slope_max - cfg.slope_min) * sigmoid
    
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
        self.slopes = hourly_slopes
        
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
        revenue = (d - c) @ self.prices
        costs = (d + c) * self.cfg.hurdle_rate
        
        prob = cp.Problem(cp.Maximize(revenue - costs), self._get_common_constraints(c, d, s))
        prob.solve(solver=cp.ECOS) # ECOS ist schnell und robust für LP
        
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
        base_rev = (d - c) @ self.prices
        
        # Preis-Impact-Strafe (Quadratisch)
        # Formel: Slope[t] * (Discharge[t] - Charge[t])^2
        net_discharge = d - c
        # cp.multiply erlaubt elementweise Multiplikation mit dem stündlichen Slope-Vektor (Fundamentalmodell)
        impact_penalty = cp.sum(cp.multiply(self.slopes, cp.square(net_discharge)))
        
        costs = (d + c) * self.cfg.hurdle_rate
        
        # Wir maximieren (Umsatz - Strafe - Kosten)
        # Das ist ein konkaves Problem (leicht lösbar), da wir den quadratischen Term abziehen
        prob = cp.Problem(cp.Maximize(base_rev - impact_penalty - costs), self._get_common_constraints(c, d, s))
        prob.solve(solver=cp.OSQP) # OSQP ist spezialisiert auf Quadratische Programme
        
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
            print("ACHTUNG: Keine Last/Erzeugungsdaten gefunden. Nutze Preis-Heuristik für Residuallast.")
            df['residual_load'] = df['price_da'] * 500 # Sehr grober Proxy
        
    # Slope berechnen (Fundamentalmodell)
    hourly_slopes = calculate_fundamental_slopes(df['residual_load'], cfg)
    
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
        
        results.append({'cap': cap, 'ex': rev_ex, 'en': rev_en})
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

    # 4. Visualisierung
    plot_results(results, plot_data, cfg)

def plot_results(results, plot_data, cfg):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))
    
    # Plot 1: Revenue Decay (Kannibalisierung)
    caps = [r['cap'] for r in results]
    ex = [r['ex']/1e6 for r in results] # in Mio €
    en = [r['en']/1e6 for r in results]
    
    ax1.plot(caps, ex, 'o--', label='Exogen (Price Taker)', color='grey')
    ax1.plot(caps, en, 'o-', label='Endogen (Price Maker)', color='#1f77b4', linewidth=2)
    ax1.set_title('Kannibalisierungseffekt: Umsatzrückgang bei steigender Kapazität')
    ax1.set_ylabel('Jahreserlös (Mio. €)')
    ax1.set_xlabel('Installierte Kapazität (MWh)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Fundamentalmodell (Residual Load vs Slope)
    # Wir sortieren die Daten für einen sauberen Linienplot der S-Kurve
    res_sorted = np.sort(plot_data['residual'])
    # Sigmoid Funktion neu berechnen für glatte Linie im Plot
    x_smooth = np.linspace(min(plot_data['residual']), max(plot_data['residual']), 100)
    sig_smooth = 1 / (1 + np.exp(-cfg.residual_sensitivity * (x_smooth - cfg.residual_inflection_point)))
    slope_smooth = cfg.slope_min + (cfg.slope_max - cfg.slope_min) * sig_smooth
    
    ax2.plot(x_smooth, slope_smooth, color='red', linewidth=2, label='Slope-Funktion (Modell)')
    ax2.scatter(plot_data['residual'], plot_data['slopes'], alpha=0.3, s=10, color='black', label='Stundenwerte (Woche 1)')
    ax2.set_title('Fundamentalmodell: Ableitung der Preissensitivität (Slope) aus Residuallast')
    ax2.set_xlabel('Residuallast (MW) [Last - Wind - Solar]')
    ax2.set_ylabel('Slope (€/MW Impact)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Operations (Vergleich Strategie)
    ax3.plot(plot_data['prices'], color='grey', alpha=0.3, label='Marktpreis (Historisch)')
    ax3_twin = ax3.twinx()
    
    # Exogen handelt aggressiv (Ignoriert Preisimpact)
    ax3_twin.fill_between(range(168), plot_data['ops_exo'], color='grey', alpha=0.3, label='Exogen (Aggressiv)')
    # Endogen handelt strategisch (Glättet Spitzen um Preis nicht zu zerstören)
    ax3_twin.plot(plot_data['ops_endo'], color='green', linewidth=2, label='Endogen (Strategisch)')
    
    ax3.set_ylabel('Preis (€/MWh)')
    ax3_twin.set_ylabel('Netto-Entladung (MW)')
    ax3.set_title('Handelsverhalten Woche 1: Strategische Zurückhaltung bei hohem Volumen')
    
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()