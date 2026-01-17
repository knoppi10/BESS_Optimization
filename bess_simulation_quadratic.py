import pandas as pd
import numpy as np
import osqp
from scipy import sparse
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. KONFIGURATION
# ==========================================
@dataclass
class SimulationConfig:
    efficiency_charge: float = 0.92
    efficiency_discharge: float = 0.92
    c_rate: float = 0.25  # HIER ÄNDERN: C-Rate
    hurdle_rate: float = 5.0  # €/MWh Opportunitätskosten
    slope: float = 0.01  # HIER ÄNDERN: Slope (Price Impact)
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
    def __init__(self, prices, capacity_mwh, config: SimulationConfig):
        """Quadratisches Optimierungsproblem mit OSQP"""
        self.prices = prices.values.copy()
        self.T = len(prices)
        self.capacity = capacity_mwh
        self.power = capacity_mwh * config.c_rate
        self.cfg = config

    def solve_osqp(self):
        """Löse mit OSQP - Optimierte CSC-Variante"""
        from scipy.sparse import tril
        
        T = self.T
        alpha = self.cfg.slope
        power = self.capacity * self.cfg.c_rate  # ← DYNAMISCH berechnen!
        
        # Hesse-Matrix (CSC)
        P = sparse.diags([2.0 * alpha] * T, format='csc')
        # NUR Preise - Hurdle Rate kann nicht in QP korrekt modelliert werden (nicht-konvex)
        q = -self.prices.copy()
        
        L_full = np.tril(np.ones((T, T)))
        L = sparse.csc_matrix(L_full)
        
        I = sparse.eye(T, format='csc')
        
        I = sparse.eye(T, format='csc')
        
        # Stack constraints (nur 3 statt 4 - redundante -I entfernt)
        A = sparse.vstack([-L, L, I], format='csc')

        l = np.hstack([
            -np.inf * np.ones(T),
            np.zeros(T),
            -power * np.ones(T)
        ])

        u = np.hstack([
            np.zeros(T),
            self.capacity * np.ones(T),
            power * np.ones(T)
])
        
        if self.cfg.enforce_end_soc_zero:
            A_end = sparse.csc_matrix(np.ones((1, T)))
            A = sparse.vstack([A, A_end], format='csc')
            l = np.hstack([l, 0.0])
            u = np.hstack([u, 0.0])
        
        try:
            solver = osqp.OSQP()
            solver.setup(P=P, q=q, A=A, l=l, u=u, verbose=False, 
                         eps_abs=1e-2, eps_rel=1e-2, max_iter=4000)
            res = solver.solve()
        except Exception as e:
            print(f"  Fehler: {e}")
            return np.zeros(T), np.zeros(T), np.zeros(T + 1), 0.0, 0.0
        
        # Results
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
# 3B. WORKER-FUNKTION FÜR MULTIPROCESSING
# ==========================================
def optimize_scenario(args):
    """Optimiert ein einzelnes Szenario - wird von Pool aufgerufen"""
    scenario_name, capacity, prices, cfg = args
    opt = BessOptimizerQuadratic(prices, capacity, cfg)
    discharge, charge, soc, revenue, cycles = opt.solve_osqp()
    net_discharge = discharge - charge
    
    return {
        'scenario_name': scenario_name,
        'net_discharge': net_discharge,
        'revenue': revenue,
        'cycles': cycles,
        'capacity': capacity
    }


# ==========================================
def run_simulation():
    # 1. Daten laden
    df = load_market_data()
    
    # Auf ein Jahr reduzieren (speicherschonend)
    df = df.iloc[:8760]  # Ein Jahr = ~8760 Stunden
    
    # 2. Config verwenden (aus Dataclass am Anfang definiert)
    cfg = SimulationConfig()
    print(f"Config: C-Rate={cfg.c_rate}, Slope={cfg.slope}")
    
    # 3. Sicherstellen, dass Residuallast da ist
    if 'residual_load' not in df.columns:
        print("Berechne Residuallast...")
        if 'load' in df.columns and 'generation_renewable' in df.columns:
            df['residual_load'] = df['load'] - df['generation_renewable']
        else:
            raise ValueError("Fehlende Spalten: 'load' oder 'generation_renewable'")
    
    # 4. Szenarien (verschiedene Batteriekapazitäten)
    scenarios = {
        'Ex': 5,
        'S': 10,
        'M': 50,
        'L': 100,
        'XL': 1000
    }
    
    # 5. Parallele Optimierung mit Pool
    print(f"\nStarte quadratische Simulation über {len(df)} Stunden...")
    print(f"{'Szenario':<10} | {'Kapazität (MWh)':<20} | {'Erlös (€)':<15} | {'Zyklen':<10}")
    print("-" * 65)
    
    # Vorbereitung der Arguments für Pool
    pool_args = [
        (scenario_name, capacity, df['price_da'], cfg)
        for scenario_name, capacity in scenarios.items()
    ]
    
    results_dict = {}
    revenue_summary = []
    
    # Pool mit Anzahl der CPU-Kerne ausführen
    with Pool() as pool:
        pool_results = pool.map(optimize_scenario, pool_args)
    
    # Ergebnisse verarbeiten
    for result in pool_results:
        scenario_name = result['scenario_name']
        results_dict[scenario_name] = result['net_discharge']
        
        revenue_summary.append({
            'Szenario': scenario_name,
            'Kapazität (MWh)': result['capacity'],
            'Erlös (€)': result['revenue'],
            'Zyklen': result['cycles']
        })
        
        print(f"{scenario_name:<10} | {result['capacity']:<20} | {result['revenue']:>13,.0f}   | {result['cycles']:>8.2f}")
    
    # 6. CSV mit stündlichen Entscheidungen speichern
    print("\nSpeichere stündliche Entscheidungen...", end=' ', flush=True)
    output_df = pd.DataFrame(results_dict)
    output_df.index = df.index
    
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulation_decisions_quadratic.csv')
    output_df.to_csv(csv_path)
    print(f"✓")
    print(f"Datei: {csv_path}")
    
    # 7. Arbitrage-Tabelle speichern
    print("Speichere Arbitrage-Zusammenfassung...", end=' ', flush=True)
    revenue_df = pd.DataFrame(revenue_summary)
    table_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'arbitrage_summary_quadratic.csv')
    revenue_df.to_csv(table_path, index=False)
    print(f"✓")
    print(f"Datei: {table_path}")
    
    # 8. Tabelle in Konsole ausgeben
    print("\n" + "="*75)
    print("ARBITRAGE-ZUSAMMENFASSUNG (Slope: 10%, c_rate: 0.2)")
    print("="*75)
    print(revenue_df.to_string(index=False))
    print("="*75)
    
    # 9. Plots
    plot_results(revenue_summary, results_dict, df, cfg)

def plot_results(results, results_dict, df, cfg):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Revenue by capacity
    scenarios = [r['Szenario'] for r in results]
    revenues = [r['Erlös (€)'] / 1e6 for r in results]
    
    axes[0].bar(scenarios, revenues, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_title('Gesamterlös nach Batteriegröße (Konstanter Slope: 10%)')
    axes[0].set_xlabel('Szenario')
    axes[0].set_ylabel('Erlös (Mio. €)')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Net discharge comparison (erste Woche)
    s = slice(0, min(8760, len(df)))
    prices = df['price_da'].values[s]
    
    ax2a = axes[1]
    ax2b = ax2a.twinx()
    
    ax2a.plot(prices, color='grey', linewidth=2, label='Marktpreis', zorder=5)
    
    # Plot für jedes Szenario (unterschiedliche Farben)
    colors = {'Ex': 'blue', 'S': 'green', 'M': 'orange', 'L': 'red', 'XL': 'purple'}
    for scenario, color in colors.items():
        if scenario in results_dict:
            data = results_dict[scenario][s]
            if hasattr(data, 'values'):
                data = data.values
            ax2b.plot(data, alpha=0.5, label=scenario, color=color, linewidth=1)
    
    ax2a.set_xlabel('Stunde (Woche 1)')
    ax2a.set_ylabel('Marktpreis (€/MWh)', color='grey')
    ax2b.set_ylabel('Netto-Entladung (MW)')
    ax2a.set_title('Woche 1: Marktpreis vs. Handelsverhalten')
    ax2a.tick_params(axis='y', labelcolor='grey')
    ax2a.grid(True, alpha=0.3)
    
    lines1, labels1 = ax2a.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2a.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()
