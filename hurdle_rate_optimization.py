"""
Hurdle Rate Optimization for BESS Arbitrage
============================================

Finds the optimal hurdle rate considering:
1. Calendar aging (max 15 years lifetime)
2. Cycle aging (max 5,000 cycles)  
3. Economies of Scale (CAPEX decreases with battery size)

Net Profit Calculation:
    Net Profit = Market Revenue - Total Costs
    
    Total Costs = CAPEX (amortized over actual usage)
    
    If cycles_used < max_cycles AND years < 15:
        → Calendar aging dominates: Effective cost = CAPEX / years
    If cycles_used >= max_cycles:
        → Cycle aging dominates: Effective cost = CAPEX / max_cycles × cycles_used

Key Insight:
    The hurdle rate r in the optimizer represents marginal degradation costs.
    r_optimal = CAPEX_per_MWh / (2 × effective_cycles)
    
    where effective_cycles = min(max_cycles, cycles_per_year × calendar_life)

Assumptions (2024/2025 values):
- Max Calendar Life: 15 years
- Max Cycle Life: 5,000 full cycles
- 1 cycle = 1 charge + 1 discharge = 2× capacity throughput
- CAPEX varies by size (Economies of Scale)
"""

import pandas as pd
import numpy as np
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import OSQP and sparse matrix tools
import osqp
from scipy import sparse

warnings.filterwarnings('ignore')

# ==========================================
# BATTERY COST ASSUMPTIONS (2024/2025)
# ==========================================
# Manually configurable parameters - adjust these values as needed!

MAX_CALENDAR_YEARS = 15  # Maximum calendar lifetime (years)
MAX_CYCLE_LIFE = 5_000   # Maximum full cycles (1 cycle = 1 charge + 1 discharge)

# ==========================================
# SCENARIOS CONFIG - EDIT VALUES HERE!
# ==========================================
# Each scenario has:
#   - capacity: Battery capacity in MWh
#   - slope: Price impact parameter α (€/(MW²·h)), 0 = Price Taker
#   - capex_per_mwh: CAPEX in €/MWh (adjust for Economies of Scale)
#
# Typical CAPEX ranges (2024/2025):
#   - Small (<10 MWh):     180,000 - 220,000 €/MWh
#   - Medium (10-100 MWh): 140,000 - 180,000 €/MWh  
#   - Large (100-1000 MWh): 110,000 - 140,000 €/MWh
#   - Very Large (>1 GWh):  90,000 - 120,000 €/MWh

SCENARIOS = {
    'Ex': {
        'capacity': 4,           # MWh
        'slope': 0.0,            # Price Taker (no market impact)
        'capex_per_mwh': 200_000 # €/MWh - Small battery, high specific cost
    },
    'S': {
        'capacity': 10,          # MWh
        'slope': 0.05,           # €/(MW²·h)
        'capex_per_mwh': 190_000 # €/MWh
    },
    'M': {
        'capacity': 100,         # MWh
        'slope': 0.05,           # €/(MW²·h)
        'capex_per_mwh': 180_000 # €/MWh - Utility scale
    },
    'L': {
        'capacity': 1000,        # MWh
        'slope': 0.05,           # €/(MW²·h)
        'capex_per_mwh': 170_000 # €/MWh - Large utility
    },
    'XL': {
        'capacity': 10000,       # MWh
        'slope': 0.05,           # €/(MW²·h)
        'capex_per_mwh': 160_000 # €/MWh - Very large, economies of scale
    },
}

# Calculate total CAPEX for each scenario (auto-computed from above)
for name, params in SCENARIOS.items():
    params['capex_total'] = params['capex_per_mwh'] * params['capacity']

# ==========================================
# MARKET DATA
# ==========================================
MARKET_DATA_FILE = 'market_data_2019_2025.csv'
YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
SIMULATION_YEARS = len(YEARS)  # 6 years of data

# ==========================================
# HURDLE RATE GRID SEARCH PARAMETERS  
# ==========================================
HURDLE_RATES_TO_TEST = np.arange(0.0, 55.0, 2.5)  # 0, 2.5, 5.0, ..., 52.5 €/MWh


# ==========================================
# CONFIGURATION
# ==========================================
@dataclass
class SimulationConfig:
    """Configuration for BESS simulation."""
    roundtrip_efficiency: float = 0.85
    efficiency_charge: Optional[float] = None
    efficiency_discharge: Optional[float] = None
    c_rate: float = 0.25
    hurdle_rate: float = 7.0
    slope: float = 0.05
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
                raise ValueError("efficiency values must be in (0, 1].")
            self.roundtrip_efficiency = float(self.efficiency_charge) * float(self.efficiency_discharge)
        else:
            raise ValueError("Please set both efficiency values or neither.")


# ==========================================
# OPTIMIZER (from Jakob.py)
# ==========================================
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
        T = len(prices)
        self.T = T
        if self.capacity > 0:
            initial_soc = float(np.clip(initial_soc, 0.0, self.capacity))
        else:
            initial_soc = 0.0
        alpha = self.cfg.slope
        power = self.power
        dt = self.dt

        n_vars = 2 * T + (T + 1)

        # Hesse matrix P for price impact
        P_rows, P_cols, P_data = [], [], []
        for t in range(T):
            c_idx = t
            d_idx = T + t
            P_rows.append(c_idx); P_cols.append(c_idx); P_data.append(2.0 * alpha * dt)
            P_rows.append(d_idx); P_cols.append(d_idx); P_data.append(2.0 * alpha * dt)
            P_rows.append(c_idx); P_cols.append(d_idx); P_data.append(-2.0 * alpha * dt)
            P_rows.append(d_idx); P_cols.append(c_idx); P_data.append(-2.0 * alpha * dt)
        self.P = sparse.csc_matrix((P_data, (P_rows, P_cols)), shape=(n_vars, n_vars))

        # Linear term q
        q = np.zeros(n_vars)
        h = self.cfg.hurdle_rate
        for t in range(T):
            c_idx = t
            d_idx = T + t
            q[c_idx] = dt * prices[t] + h * dt
            q[d_idx] = -dt * prices[t] + h * dt

        # Constraints
        constraint_rows, constraint_cols, constraint_data = [], [], []
        constraint_l, constraint_u = [], []
        constraint_idx = 0

        # Power constraints
        for t in range(T):
            c_idx = t
            d_idx = T + t
            constraint_rows.append(constraint_idx); constraint_cols.append(c_idx)
            constraint_data.append(1.0); constraint_l.append(0.0); constraint_u.append(power)
            constraint_idx += 1
            constraint_rows.append(constraint_idx); constraint_cols.append(d_idx)
            constraint_data.append(1.0); constraint_l.append(0.0); constraint_u.append(power)
            constraint_idx += 1

        # SOC bounds
        for t in range(T + 1):
            soc_idx = 2 * T + t
            constraint_rows.append(constraint_idx); constraint_cols.append(soc_idx)
            constraint_data.append(1.0); constraint_l.append(0.0); constraint_u.append(self.capacity)
            constraint_idx += 1

        # SOC dynamics
        for t in range(T):
            c_idx = t
            d_idx = T + t
            soc_t_idx = 2 * T + t
            soc_tp1_idx = 2 * T + t + 1
            constraint_rows.append(constraint_idx); constraint_cols.append(soc_tp1_idx); constraint_data.append(1.0)
            constraint_rows.append(constraint_idx); constraint_cols.append(soc_t_idx); constraint_data.append(-1.0)
            constraint_rows.append(constraint_idx); constraint_cols.append(c_idx); constraint_data.append(-self.eta_ch * dt)
            constraint_rows.append(constraint_idx); constraint_cols.append(d_idx); constraint_data.append(dt / self.eta_dis)
            constraint_l.append(0.0); constraint_u.append(0.0)
            constraint_idx += 1

        # Initial SOC
        soc_0_idx = 2 * T
        constraint_rows.append(constraint_idx); constraint_cols.append(soc_0_idx)
        constraint_data.append(1.0); constraint_l.append(initial_soc); constraint_u.append(initial_soc)
        constraint_idx += 1

        # End SOC
        if enforce_end_soc_zero:
            soc_T_idx = 2 * T + T
            constraint_rows.append(constraint_idx); constraint_cols.append(soc_T_idx)
            constraint_data.append(1.0); constraint_l.append(0.0); constraint_u.append(0.0)
            constraint_idx += 1

        A = sparse.csc_matrix((constraint_data, (constraint_rows, constraint_cols)), shape=(constraint_idx, n_vars))
        l = np.array(constraint_l)
        u = np.array(constraint_u)

        try:
            solver = osqp.OSQP()
            solver.setup(P=self.P, q=q, A=A, l=l, u=u, verbose=False,
                        eps_abs=1e-3, eps_rel=1e-3, max_iter=100000,
                        polish=True, adaptive_rho=True, rho=0.1, scaling=10, warm_start=True)
            res = solver.solve()
            if res.info.status_val != 1:
                return np.zeros(T), np.zeros(T), np.zeros(T + 1), 0.0, 0.0, 0.0
        except Exception:
            return np.zeros(T), np.zeros(T), np.zeros(T + 1), 0.0, 0.0, 0.0

        x_opt = res.x if res.x is not None else np.zeros(n_vars)
        charge = x_opt[0:T]
        discharge = x_opt[T:2*T]
        soc = x_opt[2*T:2*T+T+1]

        net_power = discharge - charge
        cashflow = dt * net_power * prices
        impact = alpha * dt * (net_power ** 2)
        ops = self.cfg.hurdle_rate * dt * (charge + discharge)
        revenue = float(np.sum(cashflow - impact - ops))

        throughput = dt * np.sum(charge + discharge)
        cycles = float(throughput / (2.0 * self.capacity)) if self.capacity > 0 else 0.0
        final_soc = float(soc[-1])

        return discharge, charge, soc, revenue, cycles, final_soc


# ==========================================
# DATA LOADING
# ==========================================
def load_market_data(filename=None):
    """Load market data from CSV."""
    if filename is None:
        filename = MARKET_DATA_FILE
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File '{filepath}' not found!")

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.set_index('timestamp')
    else:
        first_col = df.columns[0]
        df[first_col] = pd.to_datetime(df[first_col], utc=True)
        df = df.set_index(first_col)
        df.index.name = 'timestamp'

    df.index = pd.DatetimeIndex(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize('Europe/Berlin')
    else:
        df.index = df.index.tz_convert('Europe/Berlin')

    return df


# ==========================================
# COST CALCULATION WITH CALENDAR AGING
# ==========================================
def calculate_effective_costs(
    cycles_in_simulation: float,
    simulation_years: int,
    capex_per_mwh: float,
    capacity_mwh: float,
    max_calendar_years: int = MAX_CALENDAR_YEARS,
    max_cycles: int = MAX_CYCLE_LIFE
) -> Dict:
    """
    Calculate effective degradation costs considering both calendar and cycle aging.
    
    The battery degrades by whichever limit is reached first:
    - Calendar: max_calendar_years (typically 15 years)
    - Cycles: max_cycles (typically 5,000)
    
    Returns:
        Dict with cost breakdown and effective hurdle rate
    """
    # Extrapolate to full calendar lifetime
    cycles_per_year = cycles_in_simulation / simulation_years
    cycles_in_calendar_life = cycles_per_year * max_calendar_years
    
    # Determine which limit is binding
    if cycles_in_calendar_life >= max_cycles:
        # Cycle limit reached before calendar limit
        effective_cycles = max_cycles
        years_until_cycle_limit = max_cycles / cycles_per_year if cycles_per_year > 0 else float('inf')
        limiting_factor = 'cycle'
        effective_years = min(years_until_cycle_limit, max_calendar_years)
    else:
        # Calendar limit reached before cycle limit
        effective_cycles = cycles_in_calendar_life
        limiting_factor = 'calendar'
        effective_years = max_calendar_years
    
    # Total CAPEX
    total_capex = capex_per_mwh * capacity_mwh
    
    # Cost per cycle (effective)
    cost_per_cycle = total_capex / effective_cycles if effective_cycles > 0 else float('inf')
    
    # Implied optimal hurdle rate
    # 1 cycle = 2 × capacity throughput, so marginal cost per MWh throughput:
    implied_hurdle_rate = capex_per_mwh / (2 * effective_cycles) if effective_cycles > 0 else float('inf')
    
    # Actual degradation cost for simulation period
    degradation_cost_simulation = (total_capex / effective_years) * simulation_years
    
    return {
        'cycles_per_year': cycles_per_year,
        'cycles_in_calendar_life': cycles_in_calendar_life,
        'effective_cycles': effective_cycles,
        'effective_years': effective_years,
        'limiting_factor': limiting_factor,
        'total_capex': total_capex,
        'cost_per_cycle': cost_per_cycle,
        'implied_hurdle_rate': implied_hurdle_rate,
        'degradation_cost_simulation': degradation_cost_simulation,
    }


# ==========================================
# WORKER FUNCTION FOR PARALLEL EXECUTION
# ==========================================
def simulate_scenario_hurdle_rate(args):
    """
    Worker function: Simulate one scenario with one hurdle rate.
    
    Returns dict with results including proper cost calculation.
    """
    scenario_name, scenario_params, hurdle_rate, year_data, years, dt_hours = args
    
    capacity = scenario_params['capacity']
    slope = scenario_params['slope']
    capex_per_mwh = scenario_params['capex_per_mwh']
    
    # Create config with this hurdle rate
    cfg = SimulationConfig(
        hurdle_rate=hurdle_rate,
        slope=slope,
        enforce_end_soc_zero=True
    )
    
    # Create optimizer
    opt = BessOptimizerQuadratic(capacity, cfg, dt_hours=dt_hours)
    
    current_soc = 0.0
    total_revenue = 0.0
    total_cycles = 0.0
    
    # Run simulation for all years
    for year in years:
        if year not in year_data:
            continue
        year_df = year_data[year]
        prices = year_df['price_da'].values
        
        is_last_year = (year == years[-1])
        enforce_end = cfg.enforce_end_soc_zero and is_last_year
        current_soc = float(np.clip(current_soc, 0.0, capacity)) if capacity > 0 else 0.0
        
        discharge, charge, soc, revenue, cycles, final_soc = opt.solve_year(
            prices, enforce_end_soc_zero=enforce_end, initial_soc=current_soc
        )
        
        total_revenue += revenue
        total_cycles += cycles
        current_soc = final_soc
    
    # Calculate effective costs with calendar aging
    cost_info = calculate_effective_costs(
        cycles_in_simulation=total_cycles,
        simulation_years=len(years),
        capex_per_mwh=capex_per_mwh,
        capacity_mwh=capacity
    )
    
    # Net profit = Market Revenue - Degradation Costs (amortized over simulation period)
    net_profit = total_revenue - cost_info['degradation_cost_simulation']
    
    # Revenue per MWh capacity (normalized)
    revenue_per_mwh = total_revenue / capacity if capacity > 0 else 0
    net_profit_per_mwh = net_profit / capacity if capacity > 0 else 0
    
    return {
        'scenario': scenario_name,
        'capacity': capacity,
        'capex_per_mwh': capex_per_mwh,
        'hurdle_rate': hurdle_rate,
        'market_revenue': total_revenue,
        'revenue_per_mwh': revenue_per_mwh,
        'cycles': total_cycles,
        'cycles_per_year': cost_info['cycles_per_year'],
        'effective_cycles': cost_info['effective_cycles'],
        'limiting_factor': cost_info['limiting_factor'],
        'implied_optimal_hurdle': cost_info['implied_hurdle_rate'],
        'degradation_cost': cost_info['degradation_cost_simulation'],
        'net_profit': net_profit,
        'net_profit_per_mwh': net_profit_per_mwh,
    }


# ==========================================
# MAIN OPTIMIZATION
# ==========================================
def optimize_hurdle_rates():
    """Run grid search to find optimal hurdle rate for each scenario."""
    
    print("=" * 80)
    print("HURDLE RATE OPTIMIZATION WITH CALENDAR AGING & ECONOMIES OF SCALE")
    print("=" * 80)
    
    print(f"\n📊 Battery Lifetime Assumptions:")
    print(f"   Max Calendar Life: {MAX_CALENDAR_YEARS} years")
    print(f"   Max Cycle Life: {MAX_CYCLE_LIFE:,} cycles")
    print(f"   1 Cycle = 1 charge + 1 discharge")
    
    print(f"\n💰 CAPEX with Economies of Scale:")
    print(f"   {'Scenario':<8} {'Capacity':>10} {'CAPEX/MWh':>14} {'Total CAPEX':>18}")
    print(f"   {'-'*8} {'-'*10} {'-'*14} {'-'*18}")
    for name, params in SCENARIOS.items():
        print(f"   {name:<8} {params['capacity']:>10,} MWh {params['capex_per_mwh']:>12,.0f} € {params['capex_total']:>16,.0f} €")
    
    print(f"\n🔍 Hurdle Rates to Test: {HURDLE_RATES_TO_TEST[0]:.1f} - {HURDLE_RATES_TO_TEST[-1]:.1f} €/MWh")
    print(f"   Step Size: {HURDLE_RATES_TO_TEST[1] - HURDLE_RATES_TO_TEST[0]:.1f} €/MWh")
    
    # Load data
    print("\n📂 Loading market data...")
    df = load_market_data()
    
    # Detect time resolution
    if len(df) > 1:
        time_diffs = df.index.to_series().diff().dropna()
        if len(time_diffs) > 0:
            median_diff = time_diffs.median()
            dt_hours = median_diff.total_seconds() / 3600.0
        else:
            dt_hours = 1.0
    else:
        dt_hours = 1.0
    print(f"   Time Resolution: {dt_hours} hours")
    
    # Chunk data by year
    year_data = {}
    for year in YEARS:
        mask = df.index.year == year
        year_df = df[mask]
        if len(year_df) > 0:
            year_data[year] = year_df
    print(f"   Years: {list(year_data.keys())}")
    
    # Prepare arguments for parallel execution
    # Test ALL scenarios × ALL hurdle rates
    pool_args = []
    for scenario_name, scenario_params in SCENARIOS.items():
        for hr in HURDLE_RATES_TO_TEST:
            pool_args.append((scenario_name, scenario_params, hr, year_data, YEARS, dt_hours))
    
    print(f"\n⚡ Running {len(pool_args)} simulations (parallel)...")
    
    # Run parallel grid search
    all_results = []
    
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(simulate_scenario_hurdle_rate, args): args for args in pool_args}
        
        with tqdm(total=len(pool_args), desc="Simulations", unit="sim") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    args = futures[future]
                    tqdm.write(f"Error: {args[0]}, hr={args[2]}: {e}")
                finally:
                    pbar.update(1)
    
    # Organize results by scenario
    results_by_scenario = {}
    for r in all_results:
        scenario = r['scenario']
        if scenario not in results_by_scenario:
            results_by_scenario[scenario] = []
        results_by_scenario[scenario].append(r)
    
    # Sort by hurdle rate within each scenario
    for scenario in results_by_scenario:
        results_by_scenario[scenario].sort(key=lambda x: x['hurdle_rate'])
    
    # Find optimal hurdle rate for each scenario
    optimal_by_scenario = {}
    for scenario, results in results_by_scenario.items():
        best = max(results, key=lambda x: x['net_profit'])
        optimal_by_scenario[scenario] = best
    
    # Print results
    print("\n" + "=" * 100)
    print("RESULTS: OPTIMAL HURDLE RATES BY SCENARIO")
    print("=" * 100)
    
    print(f"\n{'Scenario':<8} | {'Capacity':>10} | {'Optimal r':>10} | {'Implied r*':>10} | "
          f"{'Revenue':>14} | {'Cycles':>8} | {'Net Profit':>14} | {'Limit':>8}")
    print(f"{'-'*8} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*14} | {'-'*8} | {'-'*14} | {'-'*8}")
    
    for scenario in ['Ex', 'S', 'M', 'L', 'XL']:
        if scenario not in optimal_by_scenario:
            continue
        r = optimal_by_scenario[scenario]
        print(f"{scenario:<8} | {r['capacity']:>10,} | {r['hurdle_rate']:>9.1f} € | {r['implied_optimal_hurdle']:>9.1f} € | "
              f"{r['market_revenue']:>13,.0f} € | {r['cycles']:>8.1f} | {r['net_profit']:>13,.0f} € | {r['limiting_factor']:>8}")
    
    print("\n" + "-" * 100)
    print("r = Tested hurdle rate | r* = Implied optimal based on effective cycles")
    print("Limit: 'calendar' = battery life limited by time, 'cycle' = limited by usage")
    
    # Save all results to CSV
    results_df = pd.DataFrame(all_results)
    output_file = 'hurdle_rate_optimization_all_scenarios.csv'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_file)
    results_df.to_csv(output_path, index=False)
    print(f"\n📁 Full results saved to: {output_file}")
    
    # Save optimal results
    optimal_df = pd.DataFrame([optimal_by_scenario[s] for s in ['Ex', 'S', 'M', 'L', 'XL'] if s in optimal_by_scenario])
    optimal_file = 'hurdle_rate_optimal_by_scenario.csv'
    optimal_path = os.path.join(script_dir, optimal_file)
    optimal_df.to_csv(optimal_path, index=False)
    print(f"📁 Optimal results saved to: {optimal_file}")
    
    # Create plots
    plot_hurdle_rate_optimization(results_by_scenario, optimal_by_scenario)
    
    return results_by_scenario, optimal_by_scenario


def plot_hurdle_rate_optimization(results_by_scenario, optimal_by_scenario):
    """Create visualization of hurdle rate optimization for all scenarios."""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    colors = {'Ex': 'blue', 'S': 'green', 'M': 'orange', 'L': 'red', 'XL': 'purple'}
    
    # Plot 1: Net Profit vs Hurdle Rate (all scenarios)
    ax1 = axes[0, 0]
    for scenario in ['Ex', 'S', 'M', 'L', 'XL']:
        if scenario not in results_by_scenario:
            continue
        results = results_by_scenario[scenario]
        hrs = [r['hurdle_rate'] for r in results]
        profits = [r['net_profit_per_mwh'] for r in results]
        ax1.plot(hrs, profits, color=colors[scenario], marker='o', markersize=3, 
                linewidth=1.5, label=scenario)
        # Mark optimal
        opt = optimal_by_scenario[scenario]
        ax1.scatter([opt['hurdle_rate']], [opt['net_profit_per_mwh']], 
                   color=colors[scenario], s=100, zorder=5, edgecolor='black')
    
    ax1.set_xlabel('Hurdle Rate $r$ (€/MWh)')
    ax1.set_ylabel('Net Profit per MWh Capacity (€/MWh)')
    ax1.set_title('Net Profit vs. Hurdle Rate\n(normalized by capacity)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cycles vs Hurdle Rate
    ax2 = axes[0, 1]
    for scenario in ['Ex', 'S', 'M', 'L', 'XL']:
        if scenario not in results_by_scenario:
            continue
        results = results_by_scenario[scenario]
        hrs = [r['hurdle_rate'] for r in results]
        cycles = [r['cycles'] for r in results]
        ax2.plot(hrs, cycles, color=colors[scenario], marker='o', markersize=3,
                linewidth=1.5, label=scenario)
    
    ax2.set_xlabel('Hurdle Rate $r$ (€/MWh)')
    ax2.set_ylabel('Cycles (in simulation period)')
    ax2.set_title(f'Battery Cycles vs. Hurdle Rate\n({SIMULATION_YEARS} years)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Implied vs Tested Hurdle Rate
    ax3 = axes[0, 2]
    scenarios_list = ['Ex', 'S', 'M', 'L', 'XL']
    x_pos = range(len(scenarios_list))
    optimal_hrs = [optimal_by_scenario[s]['hurdle_rate'] for s in scenarios_list if s in optimal_by_scenario]
    implied_hrs = [optimal_by_scenario[s]['implied_optimal_hurdle'] for s in scenarios_list if s in optimal_by_scenario]
    
    width = 0.35
    ax3.bar([x - width/2 for x in x_pos], optimal_hrs, width, label='Grid Search Optimal', color='steelblue')
    ax3.bar([x + width/2 for x in x_pos], implied_hrs, width, label='Theoretical r*', color='coral')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(scenarios_list)
    ax3.set_xlabel('Scenario')
    ax3.set_ylabel('Hurdle Rate (€/MWh)')
    ax3.set_title('Optimal vs. Theoretical Hurdle Rate')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Revenue decomposition for XL
    ax4 = axes[1, 0]
    if 'XL' in results_by_scenario:
        results = results_by_scenario['XL']
        hrs = [r['hurdle_rate'] for r in results]
        revenues = [r['market_revenue'] / 1e6 for r in results]
        deg_costs = [r['degradation_cost'] / 1e6 for r in results]
        net_profits = [r['net_profit'] / 1e6 for r in results]
        
        ax4.plot(hrs, revenues, 'g-o', markersize=4, label='Market Revenue')
        ax4.plot(hrs, deg_costs, 'r-s', markersize=4, label='Degradation Cost')
        ax4.plot(hrs, net_profits, 'b-^', markersize=4, label='Net Profit')
        opt = optimal_by_scenario['XL']
        ax4.axvline(x=opt['hurdle_rate'], color='blue', linestyle='--', alpha=0.7)
        
    ax4.set_xlabel('Hurdle Rate $r$ (€/MWh)')
    ax4.set_ylabel('Amount (Million €)')
    ax4.set_title('XL Scenario: Revenue Decomposition')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Limiting factor visualization
    ax5 = axes[1, 1]
    capacities = [SCENARIOS[s]['capacity'] for s in scenarios_list]
    cycles_per_year = [optimal_by_scenario[s]['cycles_per_year'] for s in scenarios_list if s in optimal_by_scenario]
    
    ax5.bar(scenarios_list, cycles_per_year, color=[colors[s] for s in scenarios_list])
    ax5.axhline(y=MAX_CYCLE_LIFE / MAX_CALENDAR_YEARS, color='red', linestyle='--', 
                label=f'Threshold ({MAX_CYCLE_LIFE/MAX_CALENDAR_YEARS:.0f} cycles/year)')
    ax5.set_xlabel('Scenario')
    ax5.set_ylabel('Cycles per Year')
    ax5.set_title(f'Cycles per Year\n(>{MAX_CYCLE_LIFE/MAX_CALENDAR_YEARS:.0f}/year → cycle-limited)')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add annotations for limiting factor
    for i, (scenario, cpy) in enumerate(zip(scenarios_list, cycles_per_year)):
        limit = optimal_by_scenario[scenario]['limiting_factor']
        ax5.annotate(limit, (i, cpy + 10), ha='center', fontsize=8)
    
    # Plot 6: CAPEX per MWh vs Capacity
    ax6 = axes[1, 2]
    capacities = [SCENARIOS[s]['capacity'] for s in scenarios_list]
    capex_values = [SCENARIOS[s]['capex_per_mwh'] / 1000 for s in scenarios_list]
    
    ax6.semilogx(capacities, capex_values, 'ko-', markersize=8, linewidth=2)
    for i, (cap, cpx, name) in enumerate(zip(capacities, capex_values, scenarios_list)):
        ax6.annotate(name, (cap, cpx + 3), ha='center', fontsize=10)
    
    ax6.set_xlabel('Battery Capacity (MWh)')
    ax6.set_ylabel('CAPEX (k€/MWh)')
    ax6.set_title('Economies of Scale:\nCAPEX vs. Battery Size')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = 'hurdle_rate_optimization_all_scenarios.png'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(script_dir, plot_file)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Plot saved to: {plot_file}")
    
    plt.show()


if __name__ == "__main__":
    results, optimal = optimize_hurdle_rates()
