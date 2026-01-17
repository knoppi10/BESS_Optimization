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
    def __init__(self, capacity_mwh, config: SimulationConfig):
        """Quadratisches Optimierungsproblem mit OSQP - Matrizen werden einmal gebaut"""
        self.T = 8760  # Stunden pro Jahr
        self.capacity = capacity_mwh
        self.power = capacity_mwh * config.c_rate
        self.cfg = config
        
        # ===== MATRIZEN EINMAL BAUEN (werden wiederverwendet) =====
        T = self.T
        alpha = self.cfg.slope
        
        # Hesse-Matrix (CSC)
        self.P = sparse.diags([2.0 * alpha] * T, format='csc')
        
        # SoC-Matrix L: Lower triangular (direkt sparse aufbauen!)
        row = np.repeat(np.arange(T), np.arange(1, T + 1))
        col = np.concatenate([np.arange(i + 1) for i in range(T)])
        data = np.ones(len(row))
        L = sparse.csc_matrix((data, (row, col)), shape=(T, T))
        
        I = sparse.eye(T, format='csc')
        
        # Stack constraints (ohne SoC end constraint - wird später hinzugefügt)
        self.A_base = sparse.vstack([-L, L, I], format='csc')
        
        self.l_base = np.hstack([
            -np.inf * np.ones(T),
            np.zeros(T),
            -self.power * np.ones(T)
        ])
        
        self.u_base = np.hstack([
            np.zeros(T),
            self.capacity * np.ones(T),
            self.power * np.ones(T)
        ])

    def solve_year(self, prices, enforce_end_soc_zero=False, initial_soc=0.0):
        """Löse für ein einzelnes Jahr mit aktualisierten Preisen"""
        T = self.T
        alpha = self.cfg.slope
        power = self.capacity * self.cfg.c_rate  # ← DYNAMISCH berechnen!
        
        # Preise als Vektor
        q = -prices.copy()
        
        # Constraints kopieren
        A = self.A_base.copy()
        l = self.l_base.copy()
        u = self.u_base.copy()
        
        # Power-Constraints dynamisch anpassen
        l[2*T:3*T] = -power * np.ones(T)
        u[2*T:3*T] = power * np.ones(T)
        
        # End-SoC Constraint: Optional
        if enforce_end_soc_zero:
            A_end = sparse.csc_matrix(np.ones((1, T)))
            A = sparse.vstack([A, A_end], format='csc')
            l = np.hstack([l, 0.0])
            u = np.hstack([u, 0.0])
        
        # Initial-SoC Constraint: Erste Stunde muss = initial_soc sein
        A_init = sparse.csc_matrix(np.zeros((1, T)))
        A_init[0, 0] = 1.0
        A = sparse.vstack([A, A_init], format='csc')
        l = np.hstack([l, initial_soc])
        u = np.hstack([u, initial_soc])
        
        try:
            solver = osqp.OSQP()
            solver.setup(P=self.P, q=q, A=A, l=l, u=u, verbose=False, 
                         eps_abs=1e-2, eps_rel=1e-2, max_iter=4000)
            res = solver.solve()
        except Exception as e:
            print(f"  Fehler: {e}")
            return np.zeros(T), np.zeros(T), np.zeros(T + 1), 0.0, 0.0, 0.0
        
        # Results
        x_opt = res.x if res.x is not None else np.zeros(T)
        soc = np.concatenate([[initial_soc], initial_soc + np.cumsum(x_opt)])
        
        cashflow = x_opt * prices
        impact = alpha * (x_opt ** 2)
        ops = np.abs(x_opt) * self.cfg.hurdle_rate
        revenue = float(np.sum(cashflow - impact - ops))
        
        discharge = np.maximum(x_opt, 0)
        charge = np.maximum(-x_opt, 0)
        cycles = float(np.sum(np.abs(x_opt)) / (2.0 * self.capacity)) if self.capacity > 0 else 0.0
        final_soc = float(soc[-1])
        
        return discharge, charge, soc, revenue, cycles, final_soc


# ==========================================
# 4. HAUPTSIMULATION (SEQUENTIELL)
# ==========================================
def run_simulation():
    # 1. Daten laden
    df = load_market_data()
    
    # 2. Config verwenden (aus Dataclass am Anfang definiert)
    cfg = SimulationConfig()
    print(f"Config: C-Rate={cfg.c_rate}, Slope={cfg.slope}")
    
    # 3. In einzelne Jahre chunken (2019-2024)
    years = [2019, 2020, 2021, 2022, 2023, 2024]
    year_data = {}
    
    print("Chunke Daten in einzelne Jahre...")
    for year in years:
        mask = df.index.year == year
        year_df = df[mask]
        if len(year_df) > 0:
            year_data[year] = year_df
            print(f"  {year}: {len(year_df)} Stunden")
    
    # 4. Szenarien
    scenarios = {
        'Ex': 5,
        'S': 10,
        'M': 50,
        'L': 100,
        'XL': 1000
    }
    
    # 5. Für jedes Szenario: Matrizen einmal bauen, dann Jahre durchrechnen
    aggregated_results = {}
    revenue_summary = []
    
    print(f"\nStarte Multi-Jahr Simulation (2019-2024, 5 Szenarien)...")
    print(f"{'Szenario':<10} | {'Gesamterlös (€)':<20} | {'Zyklen':<15}")
    print("-" * 50)
    
    for scenario_name, capacity in scenarios.items():
        print(f"{scenario_name:<10} | ", end='', flush=True)
        
        # Optimizer mit Matrizen für dieses Szenario
        opt = BessOptimizerQuadratic(capacity, cfg)
        
        all_decisions = []
        all_timestamps = []
        current_soc = 0.0  # Kontinuierlicher SoC über alle Jahre
        total_revenue = 0.0
        total_cycles = 0.0
        
        # Durch alle Jahre
        for year in years:
            if year not in year_data:
                continue
            
            year_df = year_data[year]
            prices = year_df['price_da'].values
            
            # Ist das letzte Jahr? Dann enforce_end_soc_zero
            is_last_year = (year == years[-1])
            
            # Löse Jahr
            discharge, charge, soc, revenue, cycles, final_soc = opt.solve_year(
                prices, 
                enforce_end_soc_zero=is_last_year,
                initial_soc=current_soc
            )
            
            # Speichern
            net_discharge = discharge - charge
            all_decisions.extend(net_discharge)
            all_timestamps.extend(year_df.index)
            
            total_revenue += revenue
            total_cycles += cycles
            current_soc = final_soc
        
        # Speichern für CSV
        aggregated_results[scenario_name] = all_decisions
        revenue_summary.append({
            'Szenario': scenario_name,
            'Kapazität (MWh)': capacity,
            'Gesamterlös (€)': total_revenue,
            'Zyklen': total_cycles,
            'Finale_SoC': current_soc
        })
        
        print(f"{total_revenue:>18,.0f}   | {total_cycles:>13.2f}")
        
        del opt
    
    # 6. CSV mit stündlichen Entscheidungen speichern
    print("\nSpeichere stündliche Entscheidungen...", end=' ', flush=True)
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
    prices = np.array([year_data[year]['price_da'].values for year in sorted(year_data.keys())])
    prices = np.concatenate(prices)
    
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
