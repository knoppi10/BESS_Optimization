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
    efficiency_charge: float = 0.9592
    efficiency_discharge: float = 0.9592
    c_rate: float = 0.25  # HIER ÄNDERN: C-Rate
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
        """Quadratisches Optimierungsproblem mit OSQP. Matrizen werden JIT pro Jahr gebaut."""
        self.capacity = capacity_mwh
        self.cfg = config
        # Die Matrizen werden nicht mehr im Konstruktor gebaut, um variable Jahreslängen zu erlauben.

    def solve_year(self, prices, enforce_end_soc_zero=False, initial_soc=0.0):
        """Löst das Optimierungsproblem für ein einzelnes Jahr mit variabler Länge."""
        
        T = len(prices)
        if T == 0:
            # Wenn keine Preisdaten vorhanden sind, gibt es nichts zu tun.
            return np.array([]), np.array([]), np.array([initial_soc]), 0.0, 0.0, initial_soc

        alpha = self.cfg.slope
        power = self.capacity * self.cfg.c_rate
        
        # === Matrizen und Vektoren dynamisch für die Jahreslänge T bauen ===
        
        # 1. Hesse-Matrix (P) für die quadratische Kostenfunktion und Preis-Vektor (q)
        P = sparse.diags([2.0 * alpha] * T, format='csc')
        q = -prices.copy()
        
        # 2. Constraint-Matrizen (A)
        # L-Matrix für die kumulative Summe des SoC
        row = np.repeat(np.arange(T), np.arange(1, T + 1))
        col = np.concatenate([np.arange(i + 1) for i in range(T)])
        data = np.ones(len(row))
        L = sparse.csc_matrix((data, (row, col)), shape=(T, T))
        
        # Einheitsmatrix I für die Power-Grenzen
        I = sparse.eye(T, format='csc')
        
        # A-Matrix: Stapelt die SoC- und Power-Constraints
        A = sparse.vstack([L, I], format='csc')

        # 3. Lower (l) und Upper (u) Bounds für die Constraints
        # SoC-Grenzen: 0 <= initial_soc + cumsum(x) <= capacity
        # Dies wird umgeformt zu: -initial_soc <= cumsum(x) <= capacity - initial_soc
        l_soc = -initial_soc * np.ones(T)
        u_soc = (self.capacity - initial_soc) * np.ones(T)
        
        # Power-Grenzen: -power <= x <= power
        l_power = -power * np.ones(T)
        u_power = power * np.ones(T)
        
        # Bounds-Vektoren zusammensetzen
        l = np.hstack([l_soc, l_power])
        u = np.hstack([u_soc, u_power])

        # Optional: Erzwinge, dass der SoC am Ende des Zeitraums 0 ist.
        if enforce_end_soc_zero:
            # Damit der absolute SoC am Ende 0 ist, muss gelten: initial_soc + sum(x) = 0
            # Also muss die Summe aller Entscheidungen `sum(x)` gleich `-initial_soc` sein.
            A_end = sparse.csc_matrix(np.ones((1, T)))
            A = sparse.vstack([A, A_end], format='csc')
            
            target_sum = -initial_soc
            l = np.hstack([l, target_sum])
            u = np.hstack([u, target_sum])
        
        # === OSQP Solver Setup und Lösung ===
        try:
            solver = osqp.OSQP()
            solver.setup(P=P, q=q, A=A, l=l, u=u, verbose=False, 
                         eps_abs=1e-2, eps_rel=1e-2, max_iter=4000)
            res = solver.solve()
            
            if res.info.status != 'solved':
                raise ValueError(f"OSQP konnte nicht lösen. Status: {res.info.status}")

        except Exception as e:
            print(f"  Fehler bei der OSQP-Lösung: {e}")
            return np.zeros(T), np.zeros(T), np.full(T + 1, initial_soc), 0.0, 0.0, initial_soc
        
        # === Ergebnisse verarbeiten ===
        x_opt = res.x
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
    prices = np.concatenate([year_data[year]['price_da'].values for year in sorted(year_data.keys())])
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
