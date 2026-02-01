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

# 0. CONSTANTS
MARKET_DATA_FILE = 'market_data_2019_2025.csv'
YEARS = [2019, 2020, 2021, 2022, 2023, 2024]


# 1. CONFIGURATION
@dataclass
class SimulationConfig:

    # Target round-trip efficiency (η_rt). Single efficiencies are set in __post_init__.
    roundtrip_efficiency: float = 0.85
    efficiency_charge: Optional[float] = None
    efficiency_discharge: Optional[float] = None

    c_rate: float = 0.25
    hurdle_rate: float = 5.0  # €/MWh
    slope: float = 0.01  # α [€/(MW²·h)]
    enforce_end_soc_zero: bool = True
    debug: bool = False

    def __post_init__(self):
        if self.efficiency_charge is None and self.efficiency_discharge is None:
            if not (0.0 < self.roundtrip_efficiency <= 1.0):
                raise ValueError("roundtrip_efficiency must be in (0, 1].")
            eta = float(np.sqrt(self.roundtrip_efficiency))
            self.efficiency_charge = eta
            self.efficiency_discharge = eta
        elif self.efficiency_charge is not None and self.efficiency_discharge is not None:
            if not (0.0 < float(self.efficiency_charge) <= 1.0) or not (0.0 < float(self.efficiency_discharge) <= 1.0):
                raise ValueError("efficiency_charge and efficiency_discharge must be in (0, 1].")
            self.roundtrip_efficiency = float(self.efficiency_charge) * float(self.efficiency_discharge)
        else:
            raise ValueError("Set both single efficiencies or none.")


# Scenarios: {Name: (Capacity_MWh, slope)}
DEFAULT_SLOPE = SimulationConfig().slope

SCENARIOS = {
    '4':        (4, 0.0),        # Price-taker baseline
    '10':       (10, DEFAULT_SLOPE),
    '100':      (100, DEFAULT_SLOPE),
    '1000':     (1000, DEFAULT_SLOPE),
    '10000':    (10000, DEFAULT_SLOPE),
    '10_pt':    (10, 0.0),       # Price-taker versions
    '100_pt':   (100, 0.0),
    '1000_pt':  (1000, 0.0),
    '10000_pt': (10000, 0.0),
}

MAIN_CAPACITIES = ['10', '100', '1000', '10000']
PRICE_TAKER_CAPACITY = 4


# 2. DATA LOADING
def load_market_data(filename=None):
    """Load market data from CSV."""
    if filename is None:
        filename = MARKET_DATA_FILE
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)

    if os.path.exists(filepath):
        print(f"Loading market data from {filepath}...")
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
                raise ValueError(f"No valid timestamp column in '{filepath}'.")
        
        try:
            df.index = pd.DatetimeIndex(df.index)
        except Exception:
            raise ValueError("Could not convert index to DatetimeIndex.")
        
        if df.index.tz is None:
            df.index = df.index.tz_localize('Europe/Berlin')
        else:
            df.index = df.index.tz_convert('Europe/Berlin')
        
        return df
    else:
        raise FileNotFoundError(f"File '{filepath}' not found! Please run 01_data_fetch.py first.")


# 3. OSQP OPTIMIZER
class BessOptimizerQuadratic:
    def __init__(self, capacity_mwh, config: SimulationConfig, dt_hours=1.0):
        self.T = None
        self.capacity = capacity_mwh
        self.power = capacity_mwh * config.c_rate
        self.cfg = config
        self.dt = dt_hours
        self.eta_ch = config.efficiency_charge
        self.eta_dis = config.efficiency_discharge

    def solve_year(self, prices, enforce_end_soc_zero=False, initial_soc=0.0):
        """Solve QP for one year. Returns (discharge, charge, soc, revenue, cycles, final_soc)."""
        T = len(prices)
        self.T = T
        if self.capacity > 0:
            initial_soc = float(np.clip(initial_soc, 0.0, self.capacity))
        else:
            initial_soc = 0.0
        alpha = self.cfg.slope
        power = self.power
        dt = self.dt
        
        # Variables: [c_0..c_{T-1}, d_0..d_{T-1}, soc_0..soc_T]
        n_vars = 2 * T + (T + 1)
        
        # Hessian P for price impact term: α*dt*(d_t - c_t)²
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
            # cross term
            P_rows.append(c_idx)
            P_cols.append(d_idx)
            P_data.append(-2.0 * alpha * dt)
            P_rows.append(d_idx)
            P_cols.append(c_idx)
            P_data.append(-2.0 * alpha * dt)
        self.P = sparse.csc_matrix((P_data, (P_rows, P_cols)), shape=(n_vars, n_vars))
        
        # Linear term q
        q = np.zeros(n_vars)
        h = self.cfg.hurdle_rate
        for t in range(T):
            c_idx = t
            d_idx = T + t
            q[c_idx] = dt * prices[t] + h * dt   # charge cost
            q[d_idx] = -dt * prices[t] + h * dt  # discharge revenue
        
        # Constraints
        constraint_rows = []
        constraint_cols = []
        constraint_data = []
        constraint_l = []
        constraint_u = []
        constraint_idx = 0
        
        # Power bounds
        for t in range(T):
            c_idx = t
            d_idx = T + t
            constraint_rows.append(constraint_idx)
            constraint_cols.append(c_idx)
            constraint_data.append(1.0)
            constraint_l.append(0.0)
            constraint_u.append(power)
            constraint_idx += 1
            constraint_rows.append(constraint_idx)
            constraint_cols.append(d_idx)
            constraint_data.append(1.0)
            constraint_l.append(0.0)
            constraint_u.append(power)
            constraint_idx += 1
        
        # SOC bounds
        for t in range(T + 1):
            soc_idx = 2 * T + t
            constraint_rows.append(constraint_idx)
            constraint_cols.append(soc_idx)
            constraint_data.append(1.0)
            constraint_l.append(0.0)
            constraint_u.append(self.capacity)
            constraint_idx += 1
        
        # SOC dynamics: soc_{t+1} = soc_t + dt*(η_ch*c_t - d_t/η_dis)
        for t in range(T):
            c_idx = t
            d_idx = T + t
            soc_t_idx = 2 * T + t
            soc_tp1_idx = 2 * T + t + 1
            
            constraint_rows.append(constraint_idx)
            constraint_cols.append(soc_tp1_idx)
            constraint_data.append(1.0)
            constraint_rows.append(constraint_idx)
            constraint_cols.append(soc_t_idx)
            constraint_data.append(-1.0)
            constraint_rows.append(constraint_idx)
            constraint_cols.append(c_idx)
            constraint_data.append(-self.eta_ch * dt)
            constraint_rows.append(constraint_idx)
            constraint_cols.append(d_idx)
            constraint_data.append(dt / self.eta_dis)
            
            constraint_l.append(0.0)
            constraint_u.append(0.0)
            constraint_idx += 1
        
        # Initial SOC
        soc_0_idx = 2 * T
        constraint_rows.append(constraint_idx)
        constraint_cols.append(soc_0_idx)
        constraint_data.append(1.0)
        constraint_l.append(initial_soc)
        constraint_u.append(initial_soc)
        constraint_idx += 1
        
        # End SOC = 0
        if enforce_end_soc_zero:
            soc_T_idx = 2 * T + T
            constraint_rows.append(constraint_idx)
            constraint_cols.append(soc_T_idx)
            constraint_data.append(1.0)
            constraint_l.append(0.0)
            constraint_u.append(0.0)
            constraint_idx += 1
        
        A = sparse.csc_matrix((constraint_data, (constraint_rows, constraint_cols)), 
                              shape=(constraint_idx, n_vars))
        l = np.array(constraint_l)
        u = np.array(constraint_u)
        
        try:
            solver = osqp.OSQP()
            solver.setup(
                P=self.P, q=q, A=A, l=l, u=u,
                verbose=False,
                eps_abs=1e-3, eps_rel=1e-3, max_iter=100000,
                polish=True, adaptive_rho=True, rho=0.1, scaling=10, warm_start=True,
            )
            res = solver.solve()
            if res.info.status_val != 1:
                tqdm.write(f"  Warning: Status {res.info.status}")
        except Exception as e:
            tqdm.write(f"  Error: {e}")
            return np.zeros(T), np.zeros(T), np.zeros(T + 1), 0.0, 0.0, 0.0
        
        x_opt = res.x if res.x is not None else np.zeros(n_vars)
        charge = x_opt[0:T]
        discharge = x_opt[T:2*T]
        soc = x_opt[2*T:2*T+T+1]
        
        # Revenue
        net_power = discharge - charge
        cashflow = dt * net_power * prices
        impact = alpha * dt * (net_power ** 2)
        ops = self.cfg.hurdle_rate * dt * (charge + discharge)
        revenue = float(np.sum(cashflow - impact - ops))
        
        throughput = dt * np.sum(charge + discharge)
        cycles = float(throughput / (2.0 * self.capacity)) if self.capacity > 0 else 0.0
        final_soc = float(soc[-1])
        
        return discharge, charge, soc, revenue, cycles, final_soc


# 4. WORKER
def optimize_scenario_worker(args):
    """Multiprocessing worker for one scenario."""
    scenario_name, capacity, year_data, years, cfg_dict, dt_hours = args
    cfg = SimulationConfig(**cfg_dict)
    opt = BessOptimizerQuadratic(capacity, cfg, dt_hours=dt_hours)
    
    all_decisions = []
    all_timestamps = []
    current_soc = 0.0
    total_revenue = 0.0
    total_cycles = 0.0
    
    available_years = [y for y in years if y in year_data]
    for year in available_years:
        year_df = year_data[year]
        prices = year_df['price_da'].values
        is_last_year = (year == years[-1])
        enforce_end = bool(cfg.enforce_end_soc_zero) and is_last_year
        current_soc = float(np.clip(current_soc, 0.0, capacity)) if capacity > 0 else 0.0

        discharge, charge, soc, revenue, cycles, final_soc = opt.solve_year(
            prices, enforce_end_soc_zero=enforce_end, initial_soc=current_soc
        )

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



# 5. MAIN
def run_simulation():
    df = load_market_data()
    
    # Detect time resolution
    if len(df) > 1:
        time_diffs = df.index.to_series().diff().dropna()
        if len(time_diffs) > 0:
            median_diff = time_diffs.median()
            dt_hours = median_diff.total_seconds() / 3600.0
            print(f"Detected time resolution: {dt_hours} hours")
        else:
            dt_hours = 1.0
    else:
        dt_hours = 1.0
    
    cfg = SimulationConfig()
    if cfg.efficiency_charge is None or cfg.efficiency_discharge is None:
        raise RuntimeError("Efficiency values are None.")
    
    print(f"Config: C-Rate={cfg.c_rate}, Slope={cfg.slope}, η_rt={cfg.roundtrip_efficiency:.4f}")
    
    # Split by year
    year_data = {}
    print("Splitting data by year...")
    for year in YEARS:
        year_df = df[df.index.year == year]
        if len(year_df) > 0:
            year_data[year] = year_df
            print(f"  {year}: {len(year_df)} timesteps")
    
    print(f"\nRunning {len(SCENARIOS)} scenarios ({YEARS[0]}-{YEARS[-1]})...")
    print(f"{'Scenario':<10} | {'Total Revenue (€)':<20} | {'Revenue/MWh':<14} | {'Cycles':<10}")
    print("-" * 65)
    
    aggregated_results = {}
    revenue_summary = []
    scenario_items = list(SCENARIOS.items())
    
    pool_args = []
    for scenario_name, (capacity, slope) in scenario_items:
        cfg_dict = {
            'roundtrip_efficiency': cfg.roundtrip_efficiency,
            'efficiency_charge': None,
            'efficiency_discharge': None,
            'c_rate': cfg.c_rate,
            'hurdle_rate': cfg.hurdle_rate,
            'slope': slope,
            'enforce_end_soc_zero': cfg.enforce_end_soc_zero
        }
        pool_args.append((scenario_name, capacity, year_data, YEARS, cfg_dict, dt_hours))
    
    num_workers = min(len(scenario_items), mp.cpu_count())
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_scenario = {
            executor.submit(optimize_scenario_worker, args): args[0] 
            for args in pool_args
        }
        with tqdm(total=len(scenario_items), desc="Scenarios", unit="Scenario") as pbar:
            for future in as_completed(future_to_scenario):
                scenario_name = future_to_scenario[future]
                try:
                    result = future.result()
                    if result:
                        aggregated_results[result['scenario_name']] = result['all_decisions']
                        revenue_per_mwh = result['total_revenue'] / result['capacity'] if result['capacity'] > 0 else 0
                        revenue_summary.append({
                            'Szenario': result['scenario_name'],
                            'Kapazität (MWh)': result['capacity'],
                            'Gesamterlös (€)': result['total_revenue'],
                            'Erlös/MWh (€/MWh)': revenue_per_mwh,
                            'Zyklen': result['total_cycles'],
                            'Finale_SoC': result['final_soc']
                        })
                        
                        # Output result
                        tqdm.write(f"{result['scenario_name']:<10} | "
                                 f"{result['total_revenue']:>18,.0f}   | "
                                 f"{revenue_per_mwh:>12,.0f}   | "
                                 f"{result['total_cycles']:>10.2f}")
                    
                except Exception as e:
                    tqdm.write(f"Error in scenario {scenario_name}: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    pbar.update(1)
    
    scenario_order = {name: idx for idx, (name, _) in enumerate(scenario_items)}
    revenue_summary.sort(key=lambda x: scenario_order.get(x['Szenario'], 999))
    
    # Save results
    print("\nSaving results...", end=' ', flush=True)
    all_timestamps = []
    for year in sorted(year_data.keys()):
        all_timestamps.extend(year_data[year].index)
    
    output_df = pd.DataFrame({
        'timestamp': all_timestamps,
        **{scenario: aggregated_results[scenario] for scenario in SCENARIOS.keys()}
    })
    output_df.set_index('timestamp', inplace=True)
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulation_decisions_with_pt.csv')
    output_df.to_csv(csv_path)
    
    revenue_df = pd.DataFrame(revenue_summary)
    table_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'arbitrage_summary_with_pt.csv')
    revenue_df.to_csv(table_path, index=False)
    print("✓")
    
    print("\n" + "="*75)
    print(f"ARBITRAGE SUMMARY ({YEARS[0]}-{YEARS[-1]})")
    print("="*75)
    print(revenue_df.to_string(index=False))
    print("="*75)
    
    plot_results_multiyear(revenue_summary, aggregated_results, all_timestamps, year_data, cfg)

def plot_results_multiyear(results, aggregated_results, timestamps, year_data, cfg):
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    
    # Only show January 2019
    jan_2019 = year_data[2019]
    jan_mask = jan_2019.index.month == 1
    jan_data = jan_2019[jan_mask]
    jan_prices = jan_data['price_da'].values
    jan_hours = len(jan_prices)
    
    # Timestamps for x-axis
    jan_timestamps = jan_data.index
    
    ax2a = ax
    ax2b = ax2a.twinx()
    
    ax2a.plot(range(jan_hours), jan_prices, color='grey', linewidth=1.5, label='$p_t$ (Day-Ahead Price)', zorder=5, alpha=0.8)
    
    # Plot for each scenario (only January 2019)
    colors = {'Ex': 'blue', 'S': 'green', 'M': 'orange', 'L': 'red', 'XL': 'purple'}
    for scenario, color in colors.items():
        if scenario in aggregated_results:
            # Only the first jan_hours values (January 2019)
            scenario_data = aggregated_results[scenario][:jan_hours]
            ax2b.plot(range(jan_hours), scenario_data, alpha=0.6, label=f'{scenario}: $d_t - c_t$', color=color, linewidth=1.0)
    
    ax2a.set_xlabel('Hour $t$ (January 2019)')
    ax2a.set_ylabel('Day-Ahead Price $p_t$ (€/MWh)', color='grey')
    ax2b.set_ylabel('Net Power $d_t - c_t$ (MW)')
    ax2a.set_title('January 2019: Day-Ahead Price vs. Trading Decisions')
    ax2a.tick_params(axis='y', labelcolor='grey')
    ax2a.grid(True, alpha=0.3)
    
    lines1, labels1 = ax2a.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2a.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()    
