import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Import from main optimizer module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
_optimizer = importlib.import_module('02_Arbitrage_Optimizer')
SimulationConfig = _optimizer.SimulationConfig
MARKET_DATA_FILE = _optimizer.MARKET_DATA_FILE
YEARS = _optimizer.YEARS
SCENARIOS = _optimizer.SCENARIOS
load_market_data = _optimizer.load_market_data
BessOptimizerQuadratic = _optimizer.BessOptimizerQuadratic
PRICE_TAKER_CAPACITY = _optimizer.PRICE_TAKER_CAPACITY
MAIN_CAPACITIES = _optimizer.MAIN_CAPACITIES
DEFAULT_SLOPE = _optimizer.DEFAULT_SLOPE

# Matplotlib settings to match LaTeX document (clean academic style)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6
plt.rcParams['legend.fontsize'] = 6
plt.rcParams['figure.titlesize'] = 8
plt.rcParams['axes.linewidth'] = 0.4
plt.rcParams['grid.linewidth'] = 0.2
plt.rcParams['grid.alpha'] = 0.0  # No grid
plt.rcParams['lines.linewidth'] = 0.8
plt.rcParams['lines.markersize'] = 3
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = False
plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.handlelength'] = 1.5
plt.rcParams['legend.handletextpad'] = 0.4
plt.rcParams['legend.borderpad'] = 0.2
plt.rcParams['legend.labelspacing'] = 0.3
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.02

def run_single_year_scenario(capacity, slope, year_df, cfg_base, dt_hours=1.0):
    cfg = SimulationConfig(
        roundtrip_efficiency=cfg_base.roundtrip_efficiency,
        c_rate=cfg_base.c_rate,
        hurdle_rate=cfg_base.hurdle_rate,
        slope=slope,
        enforce_end_soc_zero=True
    )
    
    opt = BessOptimizerQuadratic(capacity, cfg, dt_hours=dt_hours)
    prices = year_df['price_da'].values
    
    discharge, charge, soc, revenue, cycles, final_soc = opt.solve_year(
        prices, enforce_end_soc_zero=True, initial_soc=0.0
    )
    
    return {
        'revenue': revenue,
        'cycles': cycles,
        'revenue_per_mwh': revenue / capacity if capacity > 0 else 0
    }


def load_base_results():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Prefer new file with price-taker scenarios, fall back to old file
    summary_path_new = os.path.join(script_dir, '06_arbitrage_summary_with_pt.csv')
    summary_path_old = os.path.join(script_dir, 'arbitrage_summary_multiyear.csv')
    
    if os.path.exists(summary_path_new):
        print(f"Loading results from {summary_path_new}...")
        df = pd.read_csv(summary_path_new)
        return df
    elif os.path.exists(summary_path_old):
        print(f"Loading results from {summary_path_old} (old version without _pt scenarios)...")
        df = pd.read_csv(summary_path_old)
        return df
    else:
        print("No results found. Please run 02_Arbitrage_Optimizer.py first!")
        return None


def plot_arbitrage_per_mwh(results_df):
    # Filter out price-taker scenarios (_pt suffix) and old 'Ex' scenario
    # Keep only main scenarios: 4, 10, 100, 1000, 10000 (with price impact)
    df = results_df[~results_df['Szenario'].str.contains('_pt', na=False)].copy()
    df = df[df['Szenario'] != 'Ex'].copy() if 'Ex' in df['Szenario'].values else df
    
    capacities = df['Kapazität (MWh)'].tolist()
    revenue_per_mwh = df['Erlös/MWh (€/MWh)'].tolist()
    
    # Use capacity as x-labels
    x_labels = [f"{int(c)}" for c in capacities]
    # Colors: from dark (small) to light (large)
    colors = ['#1f4e79', '#2e75b6', '#5b9bd5', '#7fb3d5', '#9dc3e6']
    
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(x_labels, revenue_per_mwh, color=colors[:len(x_labels)], edgecolor='none', linewidth=0)
    
    for bar, val in zip(bars, revenue_per_mwh):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000,
                f'{val:,.0f}', ha='center', va='bottom', fontsize=6)
    
    ax.set_ylabel('Specific Revenue (€/MWh$_{cap}$)')
    ax.set_xlabel('Battery Capacity (MWh)')
    ax.set_ylim(0, max(revenue_per_mwh) * 1.15)
    
    plt.tight_layout()
    plt.savefig('plot_1_arbitrage_per_mwh.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig


def plot_cycles(results_df):
    # Filter out price-taker scenarios (_pt suffix) and old 'Ex' scenario
    df = results_df[~results_df['Szenario'].str.contains('_pt', na=False)].copy()
    df = df[df['Szenario'] != 'Ex'].copy() if 'Ex' in df['Szenario'].values else df
    
    capacities = df['Kapazität (MWh)'].tolist()
    cycles = df['Zyklen'].tolist()
    
    x_labels = [f"{int(c)}" for c in capacities]
    # Colors: from dark (small) to light (large)
    colors = ['#1f4e79', '#2e75b6', '#5b9bd5', '#7fb3d5', '#9dc3e6']
    
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(x_labels, cycles, color=colors[:len(x_labels)], edgecolor='none')
    
    for bar, cyc in zip(bars, cycles):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{cyc:,.0f}', ha='center', va='bottom', fontsize=6)
    
    ax.set_ylabel('Full Equivalent Cycles')
    ax.set_xlabel('Battery Capacity (MWh)')
    ax.set_ylim(0, max(cycles) * 1.15)
    
    plt.tight_layout()
    plt.savefig('plot_2_cycles.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig


def plot_change_rates(results_df):
    # Filter out price-taker scenarios (_pt suffix) and old 'Ex' scenario
    df = results_df[~results_df['Szenario'].str.contains('_pt', na=False)].copy()
    df = df[df['Szenario'] != 'Ex'].copy() if 'Ex' in df['Szenario'].values else df
    
    capacities = df['Kapazität (MWh)'].tolist()
    revenue_per_mwh = df['Erlös/MWh (€/MWh)'].tolist()
    cycles = df['Zyklen'].tolist()
    
    change_revenue = []
    change_cycles = []
    labels = []
    for i in range(1, len(capacities)):
        pct_rev = ((revenue_per_mwh[i] - revenue_per_mwh[i-1]) / revenue_per_mwh[i-1]) * 100
        pct_cyc = ((cycles[i] - cycles[i-1]) / cycles[i-1]) * 100
        change_revenue.append(pct_rev)
        change_cycles.append(pct_cyc)
        labels.append(f'{int(capacities[i-1])}→{int(capacities[i])}')
    x = np.arange(len(labels))
    
    fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))
    
    # Revenue change
    colors_rev = ['#c55a5a' if v < 0 else '#5b9bd5' for v in change_revenue]
    bars1 = axes[0].bar(x, change_revenue, color=colors_rev, edgecolor='none')
    axes[0].axhline(y=0, color='#333333', linewidth=0.3)
    axes[0].set_ylabel('Δ Revenue (%)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
    for bar, val in zip(bars1, change_revenue):
        ypos = bar.get_height() + 0.3 if val >= 0 else bar.get_height() - 0.8
        axes[0].text(bar.get_x() + bar.get_width()/2, ypos,
                    f'{val:.1f}%', ha='center', va='bottom' if val >= 0 else 'top', fontsize=5)
    
    # Cycles change
    colors_cyc = ['#c55a5a' if v < 0 else '#5b9bd5' for v in change_cycles]
    bars2 = axes[1].bar(x, change_cycles, color=colors_cyc, edgecolor='none')
    axes[1].axhline(y=0, color='#333333', linewidth=0.3)
    axes[1].set_ylabel('Δ Cycles (%)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
    for bar, val in zip(bars2, change_cycles):
        ypos = bar.get_height() + 0.3 if val >= 0 else bar.get_height() - 0.8
        axes[1].text(bar.get_x() + bar.get_width()/2, ypos,
                    f'{val:.1f}%', ha='center', va='bottom' if val >= 0 else 'top', fontsize=5)
    
    plt.tight_layout()
    plt.savefig('plot_3_change_rates.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig


def plot_price_taker_comparison(results_df):
    # Filter out _pt scenarios for main comparison
    df_filtered = results_df[~results_df['Szenario'].str.contains('_pt', na=False)].copy()
    df_filtered = df_filtered[df_filtered['Szenario'] != 'Ex'].copy() if 'Ex' in df_filtered['Szenario'].values else df_filtered
    
    # We need to compute price-taker benchmark for 4 MWh
    df = load_market_data()
    cfg = SimulationConfig()
    time_diffs = df.index.to_series().diff().dropna()
    dt_hours = time_diffs.median().total_seconds() / 3600.0
    
    # Price-taker benchmark (α=0) for reference capacity
    cfg_pt = SimulationConfig(slope=0.0)
    opt_pt = BessOptimizerQuadratic(PRICE_TAKER_CAPACITY, cfg_pt, dt_hours=dt_hours)
    
    total_revenue_pt = 0.0
    current_soc = 0.0
    for year in YEARS:
        year_df = df[df.index.year == year]
        if len(year_df) == 0:
            continue
        prices = year_df['price_da'].values
        is_last = (year == YEARS[-1])
        current_soc = float(np.clip(current_soc, 0.0, PRICE_TAKER_CAPACITY))
        _, _, _, revenue, _, final_soc = opt_pt.solve_year(prices, enforce_end_soc_zero=is_last, initial_soc=current_soc)
        total_revenue_pt += revenue
        current_soc = final_soc
    
    price_taker_rev_per_mwh = total_revenue_pt / PRICE_TAKER_CAPACITY
    
    # Get endogenous results (without _pt scenarios)
    capacities = df_filtered['Kapazität (MWh)'].tolist()
    revenue_per_mwh = df_filtered['Erlös/MWh (€/MWh)'].tolist()
    
    x_labels = [f"{int(c)}" for c in capacities]
    # Colors: from dark (small) to light (large)
    colors = ['#1f4e79', '#2e75b6', '#5b9bd5', '#7fb3d5', '#9dc3e6']
    
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(x_labels, revenue_per_mwh, color=colors[:len(x_labels)], edgecolor='none')
    
    # Add price-taker reference line
    ax.axhline(y=price_taker_rev_per_mwh, color='#c55a5a', linestyle='--', linewidth=1.0,
               label=f'Price-taker benchmark (α=0, {PRICE_TAKER_CAPACITY} MWh)')
    
    for bar, val in zip(bars, revenue_per_mwh):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'{val:,.0f}', ha='center', va='bottom', fontsize=6)
    
    ax.set_ylabel('Specific Revenue (€/MWh$_{cap}$)')
    ax.set_xlabel('Battery Capacity (MWh)')
    ax.set_ylim(0, max(max(revenue_per_mwh), price_taker_rev_per_mwh) * 1.15)
    ax.legend(fontsize=6, frameon=False, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('plot_4_price_taker_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig, price_taker_rev_per_mwh


def plot_price_impact_cost(results_df):
    print("\nGenerating Price Impact Cost Analysis...")
    
    # Extract main capacities and their price-taker counterparts
    capacities = []
    revenue_with_alpha = []
    revenue_without_alpha = []
    
    for cap in MAIN_CAPACITIES:
        cap_pt = f"{cap}_pt"  # Price-taker version
        
        # Get data for scenario with α
        row_alpha = results_df[results_df['Szenario'] == cap]
        # Get data for scenario without α (price-taker)
        row_pt = results_df[results_df['Szenario'] == cap_pt]
        
        if len(row_alpha) > 0 and len(row_pt) > 0:
            capacities.append(int(cap))
            revenue_with_alpha.append(row_alpha['Erlös/MWh (€/MWh)'].values[0])
            revenue_without_alpha.append(row_pt['Erlös/MWh (€/MWh)'].values[0])
    
    if len(capacities) == 0:
        print("  Warning: No matching scenarios found. Run 02_Arbitrage_Optimizer.py first.")
        return None
    
    x = np.arange(len(capacities))
    width = 0.35
    
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))
    
    # Left plot: Absolute comparison
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, revenue_without_alpha, width, 
                    label='Price-Taker (α=0)', color='#9dc3e6', edgecolor='none')
    bars2 = ax1.bar(x + width/2, revenue_with_alpha, width, 
                    label=f'With Price Impact (α={DEFAULT_SLOPE})', color='#1f4e79', edgecolor='none')
    
    ax1.set_ylabel('Specific Revenue (€/MWh$_{cap}$)')
    ax1.set_xlabel('Battery Capacity (MWh)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(c) for c in capacities])
    ax1.legend(fontsize=5, frameon=False, loc='upper right')
    ax1.set_ylim(0, max(max(revenue_without_alpha), max(revenue_with_alpha)) * 1.15)
    
    # Add value labels
    for bar, val in zip(bars1, revenue_without_alpha):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'{val:,.0f}', ha='center', va='bottom', fontsize=5, color='#5b9bd5')
    for bar, val in zip(bars2, revenue_with_alpha):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'{val:,.0f}', ha='center', va='bottom', fontsize=5, color='#1f4e79')
    
    # Right plot: Revenue loss percentage
    ax2 = axes[1]
    revenue_loss_pct = [(pt - alpha) / pt * 100 if pt > 0 else 0 
                        for pt, alpha in zip(revenue_without_alpha, revenue_with_alpha)]
    
    colors = ['#1f4e79', '#2e75b6', '#5b9bd5', '#9dc3e6']
    bars3 = ax2.bar(x, revenue_loss_pct, width=0.6, color=colors[:len(capacities)], edgecolor='none')
    
    ax2.set_ylabel('Revenue Loss due to Price Impact (%)')
    ax2.set_xlabel('Battery Capacity (MWh)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(c) for c in capacities])
    ax2.set_ylim(0, max(revenue_loss_pct) * 1.25 if max(revenue_loss_pct) > 0 else 10)
    
    # Add percentage labels
    for bar, val in zip(bars3, revenue_loss_pct):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=6)
    
    plt.tight_layout()
    plt.savefig('plot_5b_price_impact_cost.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\n  Price Impact Cost Summary:")
    print(f"  {'Capacity':<12} {'α=0':>12} {'α>0':>12} {'Loss':>10}")
    print("  " + "-" * 48)
    for i, cap in enumerate(capacities):
        loss = revenue_without_alpha[i] - revenue_with_alpha[i]
        loss_pct = revenue_loss_pct[i]
        print(f"  {cap:>8} MWh {revenue_without_alpha[i]:>10,.0f}€ {revenue_with_alpha[i]:>10,.0f}€ {loss_pct:>8.1f}%")
    
    return fig


def plot_yearly_comparison():
    print("\nCalculating year-by-year results...")
    
    df = load_market_data()
    cfg = SimulationConfig()
    
    # Time resolution
    time_diffs = df.index.to_series().diff().dropna()
    dt_hours = time_diffs.median().total_seconds() / 3600.0
    
    # Filter: only main scenarios (without _pt suffix)
    main_scenarios = {k: v for k, v in SCENARIOS.items() if '_pt' not in k and k != 'Ex'}
    yearly_results = {scenario: {} for scenario in main_scenarios.keys()}
    
    for year in tqdm(YEARS, desc="Years"):
        year_df = df[df.index.year == year]
        if len(year_df) == 0:
            continue
            
        for scenario_name, (capacity, slope) in main_scenarios.items():
            result = run_single_year_scenario(capacity, slope, year_df, cfg, dt_hours)
            yearly_results[scenario_name][year] = result
    
    # Plot - Original version
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))
    x = np.arange(len(YEARS))
    width = 0.18
    colors = {'4': '#0d3b66', '10': '#1f4e79', '100': '#2e75b6', '1000': '#5b9bd5', '10000': '#9dc3e6'}
    
    all_revenues = []
    for i, (scenario, results) in enumerate(yearly_results.items()):
        if scenario not in colors:
            continue
        revenues = [results[y]['revenue_per_mwh'] for y in YEARS]
        all_revenues.extend(revenues)
        axes[0].bar(x + i*width, revenues, width, label=f'{scenario} MWh', color=colors[scenario], edgecolor='none')
    axes[0].set_ylabel('Revenue (€/MWh$_{cap}$)')
    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(YEARS, fontsize=6)
    axes[0].set_ylim(0, max(all_revenues) * 1.1)  # Extra space for legend
    axes[0].legend(loc='upper left', ncol=5, fontsize=5, frameon=False,
                   handlelength=0.8, handletextpad=0.3, columnspacing=0.5)
    
    all_cycles = []
    for i, (scenario, results) in enumerate(yearly_results.items()):
        if scenario not in colors:
            continue
        cycles = [results[y]['cycles'] for y in YEARS]
        all_cycles.extend(cycles)
        axes[1].bar(x + i*width, cycles, width, label=f'{scenario} MWh', color=colors[scenario], edgecolor='none')
    axes[1].set_ylabel('Cycles')
    axes[1].set_xticks(x + width * 1.6)
    axes[1].set_xticklabels(YEARS, fontsize=6)
    axes[1].set_ylim(0, max(all_cycles) * 1.1)
    
    plt.tight_layout()
    plt.savefig('plot_5_yearly_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # V2: Short year labels
    fig2, axes2 = plt.subplots(1, 2, figsize=(5.5, 2.5))
    year_labels = [f"'{str(y)[-2:]}" for y in YEARS]
    for i, (scenario, results) in enumerate(yearly_results.items()):
        if scenario not in colors:
            continue
        revenues = [results[y]['revenue_per_mwh'] for y in YEARS]
        axes2[0].bar(x + i*width, revenues, width, label=f'{scenario} MWh', color=colors[scenario], edgecolor='none')
    axes2[0].set_ylabel('Revenue (€/MWh$_{cap}$)')
    axes2[0].set_xticks(x + width * 1.5)
    axes2[0].set_xticklabels(year_labels, fontsize=6)
    axes2[0].set_ylim(0, max(all_revenues) * 1.25)  # Extra space for legend
    axes2[0].legend(loc='upper left', ncol=5, fontsize=5, frameon=False,
                   handlelength=0.8, handletextpad=0.3, columnspacing=0.5)
    for i, (scenario, results) in enumerate(yearly_results.items()):
        if scenario not in colors:
            continue
        cycles = [results[y]['cycles'] for y in YEARS]
        axes2[1].bar(x + i*width, cycles, width, color=colors[scenario], edgecolor='none')
    axes2[1].set_ylabel('Cycles')
    axes2[1].set_xticks(x + width * 1.5)
    axes2[1].set_xticklabels(year_labels, fontsize=6)
    axes2[1].set_ylim(0, max(all_cycles) * 1.15)
    plt.tight_layout()
    plt.savefig('plot_5_yearly_comparison_v2.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig, yearly_results


def run_sensitivity_analysis():
    print("\nStarting sensitivity analysis...")
    
    df = load_market_data()
    cfg_base = SimulationConfig()
    
    # Time resolution
    time_diffs = df.index.to_series().diff().dropna()
    dt_hours = time_diffs.median().total_seconds() / 3600.0
    
    # Only one year for faster computation (2022 as representative)
    test_year = 2022
    year_df = df[df.index.year == test_year]
    
    # Test scenario: M (100 MWh) as reference
    test_capacity = 100
    base_slope = cfg_base.slope
    
    # Parameter variations
    hurdle_variations = [0, 5, 7, 10, 15, 20, 30]  # €/MWh
    slope_variations = [0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15]  # €/(MW²·h)
    crate_variations = [0.125, 0.25, 0.5, 1.0, 2.0]  # C-Rate
    
    results = {
        'hurdle_rate': {'values': hurdle_variations, 'revenue': [], 'cycles': []},
        'slope': {'values': slope_variations, 'revenue': [], 'cycles': []},
        'c_rate': {'values': crate_variations, 'revenue': [], 'cycles': []}
    }
    
    # Hurdle Rate sensitivity
    print("  Testing hurdle_rate variations...")
    for h in tqdm(hurdle_variations, desc="Hurdle Rate"):
        cfg = SimulationConfig(hurdle_rate=h, slope=base_slope)
        opt = BessOptimizerQuadratic(test_capacity, cfg, dt_hours=dt_hours)
        _, _, _, revenue, cycles, _ = opt.solve_year(year_df['price_da'].values, True, 0.0)
        results['hurdle_rate']['revenue'].append(revenue / test_capacity)
        results['hurdle_rate']['cycles'].append(cycles)
    
    # Slope sensitivity
    print("  Testing slope (α) variations...")
    for s in tqdm(slope_variations, desc="Slope"):
        cfg = SimulationConfig(slope=s)
        opt = BessOptimizerQuadratic(test_capacity, cfg, dt_hours=dt_hours)
        _, _, _, revenue, cycles, _ = opt.solve_year(year_df['price_da'].values, True, 0.0)
        results['slope']['revenue'].append(revenue / test_capacity)
        results['slope']['cycles'].append(cycles)
    
    # C-Rate sensitivity
    print("  Testing c_rate variations...")
    for c in tqdm(crate_variations, desc="C-Rate"):
        cfg = SimulationConfig(c_rate=c, slope=base_slope)
        opt = BessOptimizerQuadratic(test_capacity, cfg, dt_hours=dt_hours)
        _, _, _, revenue, cycles, _ = opt.solve_year(year_df['price_da'].values, True, 0.0)
        results['c_rate']['revenue'].append(revenue / test_capacity)
        results['c_rate']['cycles'].append(cycles)
    
    return results, test_year, cfg_base


def plot_sensitivity(results, test_year, cfg_base):
    params = ['hurdle_rate', 'slope', 'c_rate']
    titles = ['$h$ (€/MWh)', '$\\alpha$ (€/MW²h)', 'C-Rate']
    colors = ['#1f4e79', '#5b9bd5', '#9dc3e6']
    
    # Original version
    fig, axes = plt.subplots(2, 3, figsize=(5.5, 3.5))
    for i, (param, title, color) in enumerate(zip(params, titles, colors)):
        axes[0, i].plot(results[param]['values'], results[param]['revenue'], 
                       'o-', color=color, linewidth=0.8, markersize=2)
        axes[0, i].set_xlabel(title, fontsize=7)
        if i == 0:
            axes[0, i].set_ylabel('Revenue (€/MWh)', fontsize=7)
        if param == 'hurdle_rate':
            base_idx = results[param]['values'].index(cfg_base.hurdle_rate)
        elif param == 'slope':
            base_idx = results[param]['values'].index(cfg_base.slope)
        else:
            base_idx = results[param]['values'].index(cfg_base.c_rate)
        axes[0, i].axvline(x=results[param]['values'][base_idx], color='#999999', 
                          linestyle='--', linewidth=0.4)
        axes[1, i].plot(results[param]['values'], results[param]['cycles'],
                       's-', color=color, linewidth=0.8, markersize=2)
        axes[1, i].set_xlabel(title, fontsize=7)
        if i == 0:
            axes[1, i].set_ylabel('Cycles', fontsize=7)
        axes[1, i].axvline(x=results[param]['values'][base_idx], color='#999999',
                          linestyle='--', linewidth=0.4)
    plt.tight_layout()
    plt.savefig('plot_6_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # V2: All subplots have y-labels
    fig2, axes2 = plt.subplots(2, 3, figsize=(5.5, 3.5))
    for i, (param, title, color) in enumerate(zip(params, titles, colors)):
        axes2[0, i].plot(results[param]['values'], results[param]['revenue'], 
                       'o-', color=color, linewidth=0.8, markersize=2)
        axes2[0, i].set_xlabel(title, fontsize=7)
        axes2[0, i].set_ylabel('Revenue (€/MWh)', fontsize=7)
        if param == 'hurdle_rate':
            base_idx = results[param]['values'].index(cfg_base.hurdle_rate)
        elif param == 'slope':
            base_idx = results[param]['values'].index(cfg_base.slope)
        else:
            base_idx = results[param]['values'].index(cfg_base.c_rate)
        axes2[0, i].axvline(x=results[param]['values'][base_idx], color='#999999', 
                          linestyle='--', linewidth=0.4)
        axes2[1, i].plot(results[param]['values'], results[param]['cycles'],
                       's-', color=color, linewidth=0.8, markersize=2)
        axes2[1, i].set_xlabel(title, fontsize=7)
        axes2[1, i].set_ylabel('Cycles', fontsize=7)
        axes2[1, i].axvline(x=results[param]['values'][base_idx], color='#999999',
                          linestyle='--', linewidth=0.4)
    plt.tight_layout()
    plt.savefig('plot_6_sensitivity_v2.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig


def run_full_sensitivity_matrix():
    print("\nCalculating complete sensitivity matrix for all scenarios...")
    
    df = load_market_data()
    cfg_base = SimulationConfig()
    
    # Time resolution
    time_diffs = df.index.to_series().diff().dropna()
    dt_hours = time_diffs.median().total_seconds() / 3600.0
    
    # Use all years
    year_data = {year: df[df.index.year == year] for year in YEARS if len(df[df.index.year == year]) > 0}
    
    # Base case parameters from config
    base_slope = cfg_base.slope
    base_crate = cfg_base.c_rate
    base_hurdle = cfg_base.hurdle_rate
    
    # Parameter variations
    variations = {
        'Baseline': {'slope_delta': 0, 'crate_delta': 0, 'hurdle_delta': 0},
        'C-Rate +0.125': {'slope_delta': 0, 'crate_delta': 0.125, 'hurdle_delta': 0},
        'C-Rate +0.25': {'slope_delta': 0, 'crate_delta': 0.25, 'hurdle_delta': 0},
        'Slope +0.02': {'slope_delta': 0.02, 'crate_delta': 0, 'hurdle_delta': 0},
        'Slope +0.04': {'slope_delta': 0.04, 'crate_delta': 0, 'hurdle_delta': 0},
        'Hurdle -2': {'slope_delta': 0, 'crate_delta': 0, 'hurdle_delta': -2},
        'Hurdle +2': {'slope_delta': 0, 'crate_delta': 0, 'hurdle_delta': 2},
    }
    
    # Store results
    results = {var_name: {} for var_name in variations.keys()}
    baseline_revenues = {}
    
    # For each scenario and each variation
    total_runs = len(variations) * len(SCENARIOS)
    with tqdm(total=total_runs, desc="Sensitivity Analysis") as pbar:
        for var_name, deltas in variations.items():
            for scenario_name, (capacity, scenario_slope) in SCENARIOS.items():
                # Für Ex (Price Taker) bleibt slope=0
                if scenario_name == 'Ex':
                    effective_slope = 0.0
                else:
                    effective_slope = base_slope + deltas['slope_delta']
                
                effective_crate = base_crate + deltas['crate_delta']
                effective_hurdle = base_hurdle + deltas['hurdle_delta']
                
                # Create config
                cfg = SimulationConfig(
                    c_rate=effective_crate,
                    hurdle_rate=effective_hurdle,
                    slope=effective_slope,
                    enforce_end_soc_zero=True
                )
                
                # Multi-year simulation
                opt = BessOptimizerQuadratic(capacity, cfg, dt_hours=dt_hours)
                total_revenue = 0.0
                current_soc = 0.0
                
                for year in YEARS:
                    if year not in year_data:
                        continue
                    year_df = year_data[year]
                    prices = year_df['price_da'].values
                    
                    is_last_year = (year == YEARS[-1])
                    enforce_end = is_last_year
                    current_soc = float(np.clip(current_soc, 0.0, capacity))
                    
                    _, _, _, revenue, _, final_soc = opt.solve_year(
                        prices, enforce_end_soc_zero=enforce_end, initial_soc=current_soc
                    )
                    total_revenue += revenue
                    current_soc = final_soc
                
                revenue_per_mwh = total_revenue / capacity if capacity > 0 else 0
                results[var_name][scenario_name] = revenue_per_mwh
                
                # Store baseline
                if var_name == 'Baseline':
                    baseline_revenues[scenario_name] = revenue_per_mwh
                
                pbar.update(1)
    
    # Calculate percentage changes
    pct_changes = {var_name: {} for var_name in variations.keys()}
    for var_name in variations.keys():
        for scenario_name in SCENARIOS.keys():
            base_rev = baseline_revenues[scenario_name]
            current_rev = results[var_name][scenario_name]
            if base_rev > 0:
                pct_change = ((current_rev - base_rev) / base_rev) * 100
            else:
                pct_change = 0.0
            pct_changes[var_name][scenario_name] = pct_change
    
    return pct_changes, results, baseline_revenues


def plot_sensitivity_matrix_table(pct_changes):
    
    fig, ax = plt.subplots(figsize=(5, 2.8), facecolor='white')
    ax.set_facecolor('white')
    ax.axis('off')
    
    scenarios = list(SCENARIOS.keys())  # ['10', '100', '1000', '10000']
    variations = ['Baseline', 'C-Rate +0.125', 'C-Rate +0.25', 
                  'Slope +0.02', 'Slope +0.04', 'Hurdle -2', 'Hurdle +2']
    
    table_data = []
    for var_name in variations:
        row = [var_name]
        for scenario in scenarios:
            pct = pct_changes[var_name][scenario]
            row.append(f'{pct:+.1f}%' if var_name != 'Baseline' else '0.0%')
        table_data.append(row)
    
    # Column headers with MWh units
    columns = ['Variation'] + [f'{s} MWh' for s in scenarios]
    
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        colWidths=[0.22] + [0.15]*len(scenarios)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.4)
    
    # Header styling
    for j in range(len(columns)):
        table[(0, j)].set_facecolor('#1f4e79')
        table[(0, j)].set_text_props(color='white', fontsize=7)
    
    for i, var_name in enumerate(variations):
        table[(i+1, 0)].set_facecolor('#f0f0f0')
        
        for j, scenario in enumerate(scenarios):
            pct = pct_changes[var_name][scenario]
            cell = table[(i+1, j+1)]
            
            if var_name == 'Baseline':
                cell.set_facecolor('#f5f5f5')
            elif pct > 5:
                cell.set_facecolor('#9dc3e6')
            elif pct > 0:
                cell.set_facecolor('#deebf7')
            elif pct < -10:
                cell.set_facecolor('#f4b4b4')
            elif pct < 0:
                cell.set_facecolor('#fbe4e4')
            else:
                cell.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig('plot_sensitivity_matrix_table.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    return fig


def plot_sensitivity_matrix(results, test_year, cfg_base):
    
    base_cases = {
        'hurdle_rate': cfg_base.hurdle_rate,
        'slope': cfg_base.slope,
        'c_rate': cfg_base.c_rate
    }
    
    base_revenues = {}
    for param in ['hurdle_rate', 'slope', 'c_rate']:
        base_idx = results[param]['values'].index(base_cases[param])
        base_revenues[param] = results[param]['revenue'][base_idx]
    
    sensitivity_data = []
    param_labels = ['$h$', '$\\alpha$', 'C-Rate']
    change_labels = ['-50%', '-20%', 'Base', '+20%', '+50%']
    
    for param in ['hurdle_rate', 'slope', 'c_rate']:
        base_val = base_cases[param]
        base_rev = base_revenues[param]
        
        multipliers = [0.5, 0.8, 1.0, 1.2, 1.5]
        row = []
        
        for mult in multipliers:
            target_val = base_val * mult
            vals = results[param]['values']
            revs = results[param]['revenue']
            
            if target_val <= min(vals):
                rev = revs[0]
            elif target_val >= max(vals):
                rev = revs[-1]
            else:
                for j in range(len(vals)-1):
                    if vals[j] <= target_val <= vals[j+1]:
                        t = (target_val - vals[j]) / (vals[j+1] - vals[j])
                        rev = revs[j] + t * (revs[j+1] - revs[j])
                        break
            
            pct_change = ((rev - base_rev) / base_rev) * 100
            row.append(pct_change)
        
        sensitivity_data.append(row)
    
    # Original version
    fig, ax = plt.subplots(figsize=(4, 2.5))
    data = np.array(sensitivity_data)
    im = ax.imshow(data, cmap='RdBu_r', aspect='auto', vmin=-30, vmax=30)
    ax.set_xticks(np.arange(len(change_labels)))
    ax.set_yticks(np.arange(len(param_labels)))
    ax.set_xticklabels(change_labels, fontsize=6)
    ax.set_yticklabels(param_labels, fontsize=6)
    for i in range(len(param_labels)):
        for j in range(len(change_labels)):
            val = data[i, j]
            color = 'white' if abs(val) > 15 else 'black'
            ax.text(j, i, f'{val:+.0f}%', ha='center', va='center', 
                   color=color, fontsize=6)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=6)
    plt.tight_layout()
    plt.savefig('plot_7_sensitivity_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # V2: With axis labels and colorbar label
    fig2, ax2 = plt.subplots(figsize=(4, 2.5))
    im2 = ax2.imshow(data, cmap='RdBu_r', aspect='auto', vmin=-30, vmax=30)
    ax2.set_xticks(np.arange(len(change_labels)))
    ax2.set_yticks(np.arange(len(param_labels)))
    ax2.set_xticklabels(change_labels, fontsize=6)
    ax2.set_yticklabels(param_labels, fontsize=6)
    ax2.set_xlabel('Parameter Variation', fontsize=7)
    ax2.set_ylabel('Parameter', fontsize=7)
    for i in range(len(param_labels)):
        for j in range(len(change_labels)):
            val = data[i, j]
            color = 'white' if abs(val) > 15 else 'black'
            ax2.text(j, i, f'{val:+.0f}%', ha='center', va='center', 
                   color=color, fontsize=6)
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.ax.tick_params(labelsize=6)
    cbar2.set_label('Δ Revenue (%)', fontsize=6)
    plt.tight_layout()
    plt.savefig('plot_7_sensitivity_matrix_v2.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig


def plot_summary_table(results_df):
    # Filter out _pt scenarios for main summary
    df_filtered = results_df[~results_df['Szenario'].str.contains('_pt', na=False)].copy()
    df_filtered = df_filtered[df_filtered['Szenario'] != 'Ex'].copy() if 'Ex' in df_filtered['Szenario'].values else df_filtered
    
    # Use capacity as identifier instead of scenario name
    table_data = []
    for _, row in df_filtered.iterrows():
        cap = int(row['Kapazität (MWh)'])
        table_data.append([
            f"{cap:,}",
            f"{row['Gesamterlös (€)']:,.0f}",
            f"{row['Erlös/MWh (€/MWh)']:,.0f}",
            f"{row['Zyklen']:,.0f}",
        ])
    
    fig, ax = plt.subplots(figsize=(4.5, 1.8))
    ax.axis('off')
    columns = ['Capacity (MWh)', 'Total Revenue (€)', '€/MWh$_{cap}$', 'Cycles']
    table = ax.table(cellText=table_data, colLabels=columns, loc='center',
                    cellLoc='center', colWidths=[0.20, 0.28, 0.20, 0.16])
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.3)
    for j in range(len(columns)):
        table[(0, j)].set_facecolor('#1f4e79')
        table[(0, j)].set_text_props(color='white', fontsize=7)
    for i in range(1, len(table_data) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    plt.tight_layout()
    plt.savefig('plot_8_summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig


def analyze_spread_evolution():
    print("\nAnalyzing spread evolution 2019-2024...")
    
    df = load_market_data()
    
    yearly_stats = []
    
    for year in YEARS:
        year_df = df[df.index.year == year].copy()
        if len(year_df) == 0:
            continue
        
        prices = year_df['price_da']
        
        # Hourly analysis (Peak = 8-20h, Off-Peak = rest)
        year_df['hour'] = year_df.index.hour
        peak_mask = (year_df['hour'] >= 8) & (year_df['hour'] < 20)
        
        peak_prices = prices[peak_mask]
        offpeak_prices = prices[~peak_mask]
        
        # Daily Min/Max Spreads
        year_df['date'] = year_df.index.date
        daily_stats = year_df.groupby('date')['price_da'].agg(['min', 'max'])
        daily_spreads = daily_stats['max'] - daily_stats['min']
        
        stats = {
            'year': year,
            'mean_price': prices.mean(),
            'std_price': prices.std(),
            'peak_mean': peak_prices.mean(),
            'offpeak_mean': offpeak_prices.mean(),
            'peak_offpeak_spread': peak_prices.mean() - offpeak_prices.mean(),
            'daily_spread_mean': daily_spreads.mean(),
            'daily_spread_median': daily_spreads.median(),
            'daily_spread_p90': daily_spreads.quantile(0.90),
            'daily_spread_max': daily_spreads.max(),
            'price_min': prices.min(),
            'price_max': prices.max(),
            'negative_hours': (prices < 0).sum(),
            'high_price_hours': (prices > 100).sum(),  # >100 €/MWh
            'extreme_spread_days': (daily_spreads > 100).sum(),  # Days with >100€ spread
        }
        yearly_stats.append(stats)
    
    return pd.DataFrame(yearly_stats)


def plot_spread_evolution():
    print("\nAnalyzing daily spreads...")
    
    df = load_market_data()
    
    # Calculate daily spreads (Max - Min) for each day
    daily_spreads_by_year = {}
    for year in YEARS:
        year_df = df[df.index.year == year]
        if len(year_df) == 0:
            continue
        daily_spreads = year_df.groupby(year_df.index.date)['price_da'].apply(lambda x: x.max() - x.min())
        daily_spreads_by_year[year] = daily_spreads.values
    
    # Boxplot
    fig, ax = plt.subplots(figsize=(4, 2.5))
    
    # Prepare data for boxplot
    data = [daily_spreads_by_year[year] for year in YEARS]
    positions = range(len(YEARS))
    
    # Boxplot with styling
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                    showfliers=True, flierprops={'marker': '.', 'markersize': 2, 'alpha': 0.3})
    
    # Apply colors
    colors = ['#1f4e79', '#2e75b6', '#5b9bd5', '#7fb3d5', '#9dc3e6', '#bdd7ee']
    for patch, color in zip(bp['boxes'], colors[:len(YEARS)]):
        patch.set_facecolor(color)
        patch.set_edgecolor('#1f4e79')
        patch.set_linewidth(0.5)
    
    for whisker in bp['whiskers']:
        whisker.set_color('#1f4e79')
        whisker.set_linewidth(0.5)
    for cap in bp['caps']:
        cap.set_color('#1f4e79')
        cap.set_linewidth(0.5)
    for median in bp['medians']:
        median.set_color('#c55a5a')
        median.set_linewidth(0.8)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(YEARS)
    ax.set_ylabel('Daily Intraday Spread (€/MWh)')
    ax.set_xlabel('Year')
    
    # Annotate median values
    medians = [np.median(d) for d in data]
    for i, med in enumerate(medians):
        ax.text(i, med + 5, f'{med:.0f}', ha='center', va='bottom', fontsize=5, color='#c55a5a')
    
    plt.tight_layout()
    plt.savefig('plot_spread_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\n  Daily Spread Statistics:")
    print(f"  {'Year':<8} {'Median':>10} {'Mean':>10} {'P90':>10}")
    print("  " + "-" * 40)
    for year, spreads in daily_spreads_by_year.items():
        print(f"  {year:<8} {np.median(spreads):>10.1f} {np.mean(spreads):>10.1f} {np.percentile(spreads, 90):>10.1f}")
    
    return fig, daily_spreads_by_year


def plot_revenue_change_comparison_2019_2024():
    print("\nComparing revenue change rates for 2019 vs. 2024...")

    df = load_market_data()
    cfg = SimulationConfig()

    time_diffs = df.index.to_series().diff().dropna()
    dt_hours = time_diffs.median().total_seconds() / 3600.0

    main_scenarios = {k: v for k, v in SCENARIOS.items() if '_pt' not in k and k != 'Ex'}
    
    years_to_compare = [2019, 2024]
    yearly_results = {year: {} for year in years_to_compare}

    for year in tqdm(years_to_compare, desc="Calculating 2019 & 2024"):
        year_df = df[df.index.year == year]
        if len(year_df) == 0:
            print(f"Warning: No data found for year {year}.")
            continue
        
        for scenario_name, (capacity, slope) in main_scenarios.items():
            result = run_single_year_scenario(capacity, slope, year_df, cfg, dt_hours)
            yearly_results[year][scenario_name] = result

    fig, axes = plt.subplots(1, 2, figsize=(5, 2.5), sharey=True)
    
    for i, year in enumerate(years_to_compare):
        if not yearly_results.get(year):
            axes[i].text(0.5, 0.5, f'No data for {year}', ha='center', va='center')
            axes[i].set_title(f'Revenue Change Rate ({year})')
            continue

        # Sort scenarios by capacity to ensure correct order
        sorted_scenarios = sorted(main_scenarios.keys(), key=lambda k: main_scenarios[k][0])
        
        capacities = [main_scenarios[s][0] for s in sorted_scenarios]
        revenue_per_mwh = [yearly_results[year][s]['revenue_per_mwh'] for s in sorted_scenarios]

        change_revenue = []
        labels = []
        for j in range(1, len(capacities)):
            if revenue_per_mwh[j-1] > 0:
                pct_rev = ((revenue_per_mwh[j] - revenue_per_mwh[j-1]) / revenue_per_mwh[j-1]) * 100
            else:
                pct_rev = 0
            change_revenue.append(pct_rev)
            labels.append(f'{int(capacities[j-1])}→{int(capacities[j])}')
        
        x = np.arange(len(labels))
        
        ax = axes[i]
        colors_rev = ['#c55a5a' if v < 0 else '#5b9bd5' for v in change_revenue]
        bars = ax.bar(x, change_revenue, color=colors_rev, edgecolor='none')
        ax.axhline(y=0, color='#333333', linewidth=0.3)
        
        if i == 0:
            ax.set_ylabel('Δ Specific Revenue (%)')
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
        ax.set_title(f'Revenue Change Rate ({year})')

        for bar, val in zip(bars, change_revenue):
            ypos = bar.get_height() + 0.3 if val >= 0 else bar.get_height() - 0.8
            ax.text(bar.get_x() + bar.get_width()/2, ypos,
                        f'{val:.1f}%', ha='center', va='bottom' if val >= 0 else 'top', fontsize=5)

    plt.tight_layout()
    plt.savefig('plot_revenue_change_comparison_2019_2024.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig


if __name__ == "__main__":
    print("=" * 60)
    print("PRESENTATION ANALYSIS - Generating All Plots")
    print("=" * 60)
    
    # Load base results
    results_df = load_base_results()
    
    if results_df is not None:
        print(f"\nLoaded {len(results_df)} scenarios from CSV")
        print(results_df.to_string())
        
        # Generate all plots
        print("\n" + "=" * 60)
        print("Generating Plots...")
        print("=" * 60)
        
        print("\n1. Arbitrage €/MWh by Battery Capacity...")
        plot_arbitrage_per_mwh(results_df)
        
        print("\n2. Cycles by Battery Capacity...")
        plot_cycles(results_df)
        
        print("\n3. Change Rates Between Capacities...")
        plot_change_rates(results_df)
        
        print("\n4. Price-Taker vs Price-Maker Comparison...")
        fig_pt, pt_benchmark = plot_price_taker_comparison(results_df)
        print(f"   Price-taker benchmark ({PRICE_TAKER_CAPACITY} MWh, α=0): {pt_benchmark:,.0f} €/MWh")
        
        print("\n5. Year-by-Year Comparison...")
        fig_yearly, yearly_results = plot_yearly_comparison()
        
        print("\n5b. Price Impact Cost Analysis...")
        plot_price_impact_cost(results_df)
        
        print("\n6-7. Sensitivity Analysis...")
        sensitivity_results, test_year, cfg_base = run_sensitivity_analysis()
        plot_sensitivity(sensitivity_results, test_year, cfg_base)
        plot_sensitivity_matrix(sensitivity_results, test_year, cfg_base)
        
        print("\n8. Full Sensitivity Matrix (all scenarios)...")
        pct_changes, full_results, baselines = run_full_sensitivity_matrix()
        plot_sensitivity_matrix_table(pct_changes)
        
        print("\n9. Summary Table...")
        plot_summary_table(results_df)
        
        print("\n10. Spread Evolution (2019-2024)...")
        fig_spread, spread_stats = plot_spread_evolution()

        print("\n11. Revenue Change Rate Comparison (2019 vs 2024)...")
        plot_revenue_change_comparison_2019_2024()
        
        print("\n" + "=" * 60)
        print("✓ All plots generated and saved!")
        print("=" * 60)
        print("\nSaved files:")
        print("  - plot_1_arbitrage_per_mwh.png")
        print("  - plot_2_cycles.png")
        print("  - plot_3_change_rates.png")
        print("  - plot_4_price_taker_comparison.png")
        print("  - plot_5_yearly_comparison.png (+ _v2)")
        print("  - plot_5b_price_impact_cost.png")
        print("  - plot_6_sensitivity.png")
        print("  - plot_7_sensitivity_matrix.png")
        print("  - plot_sensitivity_matrix_table.png")
        print("  - plot_8_summary_table.png")
        print("  - plot_spread_boxplot.png")
        print("  - plot_revenue_change_comparison_2019_2024.png")
    else:
        print("ERROR: No results found. Please run 02_Arbitrage_Optimizer.py first!")
