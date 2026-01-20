"""
Presentation Analysis: Plots für BESS Arbitrage Präsentation

Generiert alle Plots für die Präsentation:
1. Arbitrage €/MWh nach Batteriegröße
2. Cycles nach Batteriegröße  
3. Veränderungsraten zwischen Größen
4. Exogen vs Endogen Vergleich
5. Jahr-für-Jahr Vergleich
6. Sensitivitätsanalyse (Hurdle Rate, Slope, C-Rate)
7. Sensitivitäts-Matrix/Heatmap
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Import from Jakob.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Jakob import (SimulationConfig, MARKET_DATA_FILE, YEARS, SCENARIOS, 
                   load_market_data, BessOptimizerQuadratic)

# Matplotlib settings für Präsentation
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# ==========================================
# HELPER: Run single scenario for one year
# ==========================================
def run_single_year_scenario(capacity, slope, year_df, cfg_base, dt_hours=1.0):
    """Führe ein Szenario für ein Jahr aus"""
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


# ==========================================
# 1. LOAD DATA AND RUN BASE SCENARIOS
# ==========================================
def load_base_results():
    """Lade Ergebnisse aus CSV oder berechne neu"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(script_dir, 'arbitrage_summary_multiyear.csv')
    
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        return df
    else:
        print("Keine Ergebnisse gefunden. Bitte erst Jakob.py ausführen!")
        return None


# ==========================================
# 2. PLOT: Arbitrage €/MWh by Battery Size
# ==========================================
def plot_arbitrage_per_mwh(results_df):
    """Plot 1: Arbitrage Revenue per MWh Capacity"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scenarios = results_df['Szenario'].tolist()
    revenue_per_mwh = results_df['Erlös/MWh (€/MWh)'].tolist()
    capacities = results_df['Kapazität (MWh)'].tolist()
    cycles = results_df['Zyklen'].tolist()
    
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
    bars = ax.bar(scenarios, revenue_per_mwh, color=colors, edgecolor='black', linewidth=1.2)
    
    # Calculate percentage changes vs previous (smaller) battery
    pct_changes = [None]  # First scenario has no predecessor
    for i in range(1, len(revenue_per_mwh)):
        pct = ((revenue_per_mwh[i] - revenue_per_mwh[i-1]) / revenue_per_mwh[i-1]) * 100
        pct_changes.append(pct)
    
    # Values on bars
    for i, (bar, val, cap, cyc) in enumerate(zip(bars, revenue_per_mwh, capacities, cycles)):
        # Top of bar: Revenue/MWh and Capacity
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f'{val:,.0f} €/MWh\n({cap:,.0f} MWh)', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Middle of bar: Percentage change vs smaller battery
        if pct_changes[i] is not None:
            mid_height = bar.get_height() / 2
            ax.text(bar.get_x() + bar.get_width()/2, mid_height,
                    f'{pct_changes[i]:+.1f}%', 
                    ha='center', va='center', fontsize=11, fontweight='bold',
                    color='white', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))
        
        # Bottom of bar: Total cycles
        ax.text(bar.get_x() + bar.get_width()/2, 800,
                f'{cyc:,.0f}\nCycles', 
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color='white')
    
    ax.set_ylabel('Net Revenue (€ / MWh Capacity)')
    ax.set_xlabel('Scenario (Battery Size)')
    ax.set_title('Arbitrage Revenue per MWh Capacity (2019-2024)\nImpact of Battery Size on Profitability')
    ax.set_ylim(0, max(revenue_per_mwh) * 1.2)
    ax.grid(True, alpha=0.3, axis='y')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig('plot_1_arbitrage_per_mwh.png', dpi=300, bbox_inches='tight')
    plt.show()
    return fig


# ==========================================
# 3. PLOT: Cycles by Battery Size
# ==========================================
def plot_cycles(results_df):
    """Plot 2: Cycles by Battery Size"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scenarios = results_df['Szenario'].tolist()
    cycles = results_df['Zyklen'].tolist()
    capacities = results_df['Kapazität (MWh)'].tolist()
    
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
    bars = ax.bar(scenarios, cycles, color=colors, edgecolor='black', linewidth=1.2)
    
    for bar, cyc, cap in zip(bars, cycles, capacities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f'{cyc:,.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Total Cycles (6 Years)')
    ax.set_xlabel('Scenario (Battery Size)')
    ax.set_title('Battery Cycles by Scenario (2019-2024)\nPrice Impact Reduces Cycling for Large Batteries')
    ax.set_ylim(0, max(cycles) * 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('plot_2_cycles.png', dpi=300, bbox_inches='tight')
    plt.show()
    return fig


# ==========================================
# 4. PLOT: Change Rates Between Sizes
# ==========================================
def plot_change_rates(results_df):
    """Plot 3: Veränderungsraten zwischen Größen"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    scenarios = results_df['Szenario'].tolist()
    revenue_per_mwh = results_df['Erlös/MWh (€/MWh)'].tolist()
    cycles = results_df['Zyklen'].tolist()
    
    # Calculate change rates
    change_revenue = []
    change_cycles = []
    labels = []
    
    for i in range(1, len(scenarios)):
        pct_rev = ((revenue_per_mwh[i] - revenue_per_mwh[i-1]) / revenue_per_mwh[i-1]) * 100
        pct_cyc = ((cycles[i] - cycles[i-1]) / cycles[i-1]) * 100
        change_revenue.append(pct_rev)
        change_cycles.append(pct_cyc)
        labels.append(f'{scenarios[i-1]}→{scenarios[i]}')
    
    x = np.arange(len(labels))
    
    # Revenue change
    colors_rev = ['#e74c3c' if v < 0 else '#2ecc71' for v in change_revenue]
    bars1 = axes[0].bar(x, change_revenue, color=colors_rev, edgecolor='black')
    axes[0].axhline(y=0, color='black', linewidth=0.8)
    axes[0].set_ylabel('Change in €/MWh (%)')
    axes[0].set_xlabel('Scenario Transition')
    axes[0].set_title('Revenue/MWh Change Rate')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars1, change_revenue):
        ypos = bar.get_height() + 0.5 if val >= 0 else bar.get_height() - 2
        axes[0].text(bar.get_x() + bar.get_width()/2, ypos, f'{val:.1f}%', 
                    ha='center', va='bottom' if val >= 0 else 'top', fontweight='bold')
    
    # Cycles change
    colors_cyc = ['#e74c3c' if v < 0 else '#2ecc71' for v in change_cycles]
    bars2 = axes[1].bar(x, change_cycles, color=colors_cyc, edgecolor='black')
    axes[1].axhline(y=0, color='black', linewidth=0.8)
    axes[1].set_ylabel('Change in Cycles (%)')
    axes[1].set_xlabel('Scenario Transition')
    axes[1].set_title('Cycles Change Rate')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars2, change_cycles):
        ypos = bar.get_height() + 0.5 if val >= 0 else bar.get_height() - 2
        axes[1].text(bar.get_x() + bar.get_width()/2, ypos, f'{val:.1f}%',
                    ha='center', va='bottom' if val >= 0 else 'top', fontweight='bold')
    
    plt.suptitle('Change Rates Between Battery Sizes', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot_3_change_rates.png', dpi=300, bbox_inches='tight')
    plt.show()
    return fig


# ==========================================
# 5. PLOT: Exogen vs Endogen Comparison
# ==========================================
def plot_exogen_vs_endogen(results_df):
    """Plot 4: Vergleich Exogen (Price Taker) vs Endogen (Price Impact)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Ex ist Price Taker (exogen), alle anderen sind endogen
    ex_row = results_df[results_df['Szenario'] == 'Ex'].iloc[0]
    ex_revenue_per_mwh = ex_row['Erlös/MWh (€/MWh)']
    
    scenarios = results_df['Szenario'].tolist()
    revenue_per_mwh = results_df['Erlös/MWh (€/MWh)'].tolist()
    
    # Berechne % Unterschied zu Ex (Price Taker benchmark)
    pct_diff = [(r / ex_revenue_per_mwh - 1) * 100 for r in revenue_per_mwh]
    
    colors = ['#3498db' if p >= 0 else '#e74c3c' for p in pct_diff]
    bars = ax.bar(scenarios, pct_diff, color=colors, edgecolor='black', linewidth=1.2)
    
    ax.axhline(y=0, color='black', linewidth=1.5, linestyle='-', label='Ex (Price Taker = Benchmark)')
    
    for bar, val, rev in zip(bars, pct_diff, revenue_per_mwh):
        ypos = bar.get_height() + 1 if val >= 0 else bar.get_height() - 3
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f'{val:+.1f}%\n({rev:,.0f} €/MWh)', 
                ha='center', va='bottom' if val >= 0 else 'top', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Difference vs Price Taker (%)')
    ax.set_xlabel('Scenario')
    ax.set_title('Endogenous vs Exogenous Price: Revenue Impact\nEx = Price Taker (α=0), Others = Price Impact (α=0.05)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('plot_4_exogen_vs_endogen.png', dpi=300, bbox_inches='tight')
    plt.show()
    return fig


# ==========================================
# 6. PLOT: Year-by-Year Comparison
# ==========================================
def plot_yearly_comparison():
    """Plot 5: Jahr-für-Jahr Vergleich"""
    print("\nBerechne Jahr-für-Jahr Ergebnisse...")
    
    df = load_market_data()
    cfg = SimulationConfig()
    
    # Zeitauflösung
    time_diffs = df.index.to_series().diff().dropna()
    dt_hours = time_diffs.median().total_seconds() / 3600.0
    
    yearly_results = {scenario: {} for scenario in SCENARIOS.keys()}
    
    for year in tqdm(YEARS, desc="Years"):
        year_df = df[df.index.year == year]
        if len(year_df) == 0:
            continue
            
        for scenario_name, (capacity, slope) in SCENARIOS.items():
            result = run_single_year_scenario(capacity, slope, year_df, cfg, dt_hours)
            yearly_results[scenario_name][year] = result
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(YEARS))
    width = 0.15
    colors = {'Ex': '#3498db', 'S': '#2ecc71', 'M': '#f39c12', 'L': '#e74c3c', 'XL': '#9b59b6'}
    
    # Revenue per MWh by year
    for i, (scenario, results) in enumerate(yearly_results.items()):
        revenues = [results[y]['revenue_per_mwh'] for y in YEARS]
        axes[0].bar(x + i*width, revenues, width, label=scenario, color=colors[scenario], edgecolor='black')
    
    axes[0].set_ylabel('Revenue (€/MWh Capacity)')
    axes[0].set_xlabel('Year')
    axes[0].set_title('Arbitrage Revenue by Year and Scenario')
    axes[0].set_xticks(x + width * 2)
    axes[0].set_xticklabels(YEARS)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    # Cycles by year
    for i, (scenario, results) in enumerate(yearly_results.items()):
        cycles = [results[y]['cycles'] for y in YEARS]
        axes[1].bar(x + i*width, cycles, width, label=scenario, color=colors[scenario], edgecolor='black')
    
    axes[1].set_ylabel('Cycles')
    axes[1].set_xlabel('Year')
    axes[1].set_title('Battery Cycles by Year and Scenario')
    axes[1].set_xticks(x + width * 2)
    axes[1].set_xticklabels(YEARS)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Year-by-Year Comparison (2019-2024)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot_5_yearly_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig, yearly_results


# ==========================================
# 7. SENSITIVITY ANALYSIS
# ==========================================
def run_sensitivity_analysis():
    """Sensitivitätsanalyse für hurdle_rate, slope, c_rate"""
    print("\nStarte Sensitivitätsanalyse...")
    
    df = load_market_data()
    cfg_base = SimulationConfig()
    
    # Zeitauflösung
    time_diffs = df.index.to_series().diff().dropna()
    dt_hours = time_diffs.median().total_seconds() / 3600.0
    
    # Nur ein Jahr für schnellere Berechnung (2022 als repräsentativ)
    test_year = 2022
    year_df = df[df.index.year == test_year]
    
    # Test-Szenario: M (100 MWh) als Referenz
    test_capacity = 100
    base_slope = cfg_base.slope
    
    # Parameter-Variationen
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
    
    return results, test_year


def plot_sensitivity(results, test_year):
    """Plot 6: Sensitivitätsanalyse Plots"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    params = ['hurdle_rate', 'slope', 'c_rate']
    titles = ['Hurdle Rate $r$ (€/MWh)', 'Price Impact $α$ (€/MW²·h)', 'C-Rate']
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for i, (param, title, color) in enumerate(zip(params, titles, colors)):
        # Revenue plot
        axes[0, i].plot(results[param]['values'], results[param]['revenue'], 
                       'o-', color=color, linewidth=2, markersize=8)
        axes[0, i].set_xlabel(title)
        axes[0, i].set_ylabel('Revenue (€/MWh)')
        axes[0, i].set_title(f'Revenue vs {title}')
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        # Mark base case
        if param == 'hurdle_rate':
            base_idx = results[param]['values'].index(7)
        elif param == 'slope':
            base_idx = results[param]['values'].index(0.05)
        else:
            base_idx = results[param]['values'].index(0.25)
        
        axes[0, i].axvline(x=results[param]['values'][base_idx], color='grey', 
                          linestyle='--', alpha=0.7, label='Base Case')
        axes[0, i].legend()
        
        # Cycles plot
        axes[1, i].plot(results[param]['values'], results[param]['cycles'],
                       's-', color=color, linewidth=2, markersize=8)
        axes[1, i].set_xlabel(title)
        axes[1, i].set_ylabel('Cycles')
        axes[1, i].set_title(f'Cycles vs {title}')
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].axvline(x=results[param]['values'][base_idx], color='grey',
                          linestyle='--', alpha=0.7, label='Base Case')
        axes[1, i].legend()
    
    plt.suptitle(f'Sensitivity Analysis (100 MWh Battery, Year {test_year})', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot_6_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.show()
    return fig


# ==========================================
# 8. FULL SENSITIVITY MATRIX (ALL SCENARIOS)
# ==========================================
def run_full_sensitivity_matrix():
    """
    Sensitivitätsanalyse für ALLE Szenarien mit:
    - Alpha (slope): +0.02, +0.04 (von 0.05 auf 0.07, 0.09)
    - C-Rate: +0.125, +0.25 (von 0.25 auf 0.375, 0.50)
    - Hurdle Rate: -2, +2 (von 7 auf 5, 9)
    """
    print("\nBerechne vollständige Sensitivitätsmatrix für alle Szenarien...")
    
    df = load_market_data()
    cfg_base = SimulationConfig()
    
    # Zeitauflösung
    time_diffs = df.index.to_series().diff().dropna()
    dt_hours = time_diffs.median().total_seconds() / 3600.0
    
    # Alle Jahre verwenden
    year_data = {year: df[df.index.year == year] for year in YEARS if len(df[df.index.year == year]) > 0}
    
    # Base case parameters
    base_slope = 0.05
    base_crate = 0.25
    base_hurdle = 7.0
    
    # Parameter-Variationen
    variations = {
        'Baseline': {'slope_delta': 0, 'crate_delta': 0, 'hurdle_delta': 0},
        'C-Rate +0.125': {'slope_delta': 0, 'crate_delta': 0.125, 'hurdle_delta': 0},
        'C-Rate +0.25': {'slope_delta': 0, 'crate_delta': 0.25, 'hurdle_delta': 0},
        'Slope +0.02': {'slope_delta': 0.02, 'crate_delta': 0, 'hurdle_delta': 0},
        'Slope +0.04': {'slope_delta': 0.04, 'crate_delta': 0, 'hurdle_delta': 0},
        'Hurdle -2': {'slope_delta': 0, 'crate_delta': 0, 'hurdle_delta': -2},
        'Hurdle +2': {'slope_delta': 0, 'crate_delta': 0, 'hurdle_delta': 2},
    }
    
    # Ergebnisse speichern
    results = {var_name: {} for var_name in variations.keys()}
    baseline_revenues = {}
    
    # Für jedes Szenario und jede Variation
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
                
                # Config erstellen
                cfg = SimulationConfig(
                    c_rate=effective_crate,
                    hurdle_rate=effective_hurdle,
                    slope=effective_slope,
                    enforce_end_soc_zero=True
                )
                
                # Multi-Jahr Simulation
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
                
                # Baseline speichern
                if var_name == 'Baseline':
                    baseline_revenues[scenario_name] = revenue_per_mwh
                
                pbar.update(1)
    
    # Berechne prozentuale Änderungen
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
    """Plot: Sensitivitäts-Matrix als Tabelle mit weißem Hintergrund"""
    
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
    ax.set_facecolor('white')
    ax.axis('off')
    
    # Daten vorbereiten
    scenarios = ['Ex', 'S', 'M', 'L', 'XL']
    variations = ['Baseline', 'C-Rate +0.125', 'C-Rate +0.25', 
                  'Slope +0.02', 'Slope +0.04', 'Hurdle -2', 'Hurdle +2']
    
    # Tabellendaten erstellen
    table_data = []
    for var_name in variations:
        row = [var_name]
        for scenario in scenarios:
            pct = pct_changes[var_name][scenario]
            row.append(f'{pct:+.2f}%' if var_name != 'Baseline' else '0.00%')
        table_data.append(row)
    
    columns = ['Parameter\nVariation'] + scenarios
    
    # Tabelle erstellen
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        colWidths=[0.20] + [0.12]*5
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 2.2)
    
    # Header styling (dunkelblau)
    for j in range(len(columns)):
        table[(0, j)].set_facecolor('#2c3e50')
        table[(0, j)].set_text_props(color='white', fontweight='bold', fontsize=12)
    
    # Zellen einfärben basierend auf Wert
    for i, var_name in enumerate(variations):
        # Erste Spalte (Variation Name) hellgrau
        table[(i+1, 0)].set_facecolor('#f0f0f0')
        table[(i+1, 0)].set_text_props(fontweight='bold')
        
        for j, scenario in enumerate(scenarios):
            pct = pct_changes[var_name][scenario]
            cell = table[(i+1, j+1)]
            
            # Farbcodierung: grün für positiv, rot für negativ, weiß für Baseline
            if var_name == 'Baseline':
                cell.set_facecolor('#f8f9fa')
            elif pct > 5:
                cell.set_facecolor('#c8e6c9')  # Hellgrün
            elif pct > 0:
                cell.set_facecolor('#e8f5e9')  # Sehr hellgrün
            elif pct < -10:
                cell.set_facecolor('#ffcdd2')  # Hellrot
            elif pct < 0:
                cell.set_facecolor('#ffebee')  # Sehr hellrot
            else:
                cell.set_facecolor('white')
    
    ax.set_title(
        'SENSITIVITY ANALYSIS: Percentage Change in Arbitrage (€/MWh) vs Baseline\n'
        '(Base: α=0.05, C-Rate=0.25, Hurdle=7 €/MWh, Period: 2019-2024)',
        fontsize=13, fontweight='bold', pad=20, color='#2c3e50'
    )
    
    plt.tight_layout()
    plt.savefig('plot_sensitivity_matrix_table.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    return fig


# ==========================================
# 9. SENSITIVITY MATRIX / HEATMAP (OLD)
# ==========================================
def plot_sensitivity_matrix(results, test_year):
    """Plot 7: Sensitivitäts-Matrix als Heatmap"""
    
    # Berechne relative Änderungen zum Base Case
    base_cases = {
        'hurdle_rate': 7,
        'slope': 0.01,
        'c_rate': 0.25
    }
    
    # Finde Base-Case-Revenue für jede Parameter
    base_revenues = {}
    for param in ['hurdle_rate', 'slope', 'c_rate']:
        base_idx = results[param]['values'].index(base_cases[param])
        base_revenues[param] = results[param]['revenue'][base_idx]
    
    # Berechne prozentuale Änderung für ±20% und ±50% Änderung
    sensitivity_data = []
    param_labels = ['Hurdle Rate $r$', 'Price Impact $α$', 'C-Rate']
    change_labels = ['-50%', '-20%', 'Base', '+20%', '+50%']
    
    for param, label in zip(['hurdle_rate', 'slope', 'c_rate'], param_labels):
        base_val = base_cases[param]
        base_rev = base_revenues[param]
        
        # Interpoliere für -50%, -20%, base, +20%, +50%
        multipliers = [0.5, 0.8, 1.0, 1.2, 1.5]
        row = []
        
        for mult in multipliers:
            target_val = base_val * mult
            # Finde nächsten Wert in results
            vals = results[param]['values']
            revs = results[param]['revenue']
            
            # Linear interpolation
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
    
    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = np.array(sensitivity_data)
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=-30, vmax=30)
    
    ax.set_xticks(np.arange(len(change_labels)))
    ax.set_yticks(np.arange(len(param_labels)))
    ax.set_xticklabels(change_labels)
    ax.set_yticklabels(param_labels)
    
    # Annotate cells
    for i in range(len(param_labels)):
        for j in range(len(change_labels)):
            val = data[i, j]
            color = 'white' if abs(val) > 15 else 'black'
            ax.text(j, i, f'{val:+.1f}%', ha='center', va='center', 
                   color=color, fontsize=11, fontweight='bold')
    
    ax.set_title(f'Sensitivity Matrix: Revenue Change (%)\n100 MWh Battery, Year {test_year}', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Parameter Change')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Revenue Change (%)')
    
    plt.tight_layout()
    plt.savefig('plot_7_sensitivity_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    return fig


# ==========================================
# 9. SUMMARY TABLE AS PLOT
# ==========================================
def plot_summary_table(results_df):
    """Plot 8: Summary Tabelle als Grafik"""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    
    # Prepare data
    table_data = []
    for _, row in results_df.iterrows():
        table_data.append([
            row['Szenario'],
            f"{row['Kapazität (MWh)']:,.0f}",
            f"{row['Gesamterlös (€)']:,.0f}",
            f"{row['Erlös/MWh (€/MWh)']:,.0f}",
            f"{row['Zyklen']:,.0f}",
        ])
    
    # Add change rates
    for i in range(len(table_data)):
        if i == 0:
            table_data[i].append('-')
        else:
            prev = results_df.iloc[i-1]['Erlös/MWh (€/MWh)']
            curr = results_df.iloc[i]['Erlös/MWh (€/MWh)']
            pct = ((curr - prev) / prev) * 100
            table_data[i].append(f'{pct:+.1f}%')
    
    columns = ['Scenario', 'Capacity\n(MWh)', 'Total Revenue\n(€)', 
               'Revenue/MWh\n(€/MWh)', 'Cycles', 'Change vs\nPrevious']
    
    table = ax.table(cellText=table_data, colLabels=columns, loc='center',
                    cellLoc='center', colWidths=[0.12, 0.12, 0.18, 0.15, 0.12, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Header styling
    for j in range(len(columns)):
        table[(0, j)].set_facecolor('#3498db')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    ax.set_title('BESS Arbitrage Summary (2019-2024)', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('plot_8_summary_table.png', dpi=300, bbox_inches='tight')
    plt.show()
    return fig


# ==========================================
# MAIN
# ==========================================
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
        
        print("\n1. Arbitrage €/MWh by Battery Size...")
        plot_arbitrage_per_mwh(results_df)
        
        print("\n2. Cycles by Battery Size...")
        plot_cycles(results_df)
        
        print("\n3. Change Rates Between Sizes...")
        plot_change_rates(results_df)
        
        print("\n4. Exogen vs Endogen Comparison...")
        plot_exogen_vs_endogen(results_df)
        
        print("\n5. Year-by-Year Comparison...")
        fig_yearly, yearly_results = plot_yearly_comparison()
        
        print("\n6-7. Sensitivity Analysis...")
        sensitivity_results, test_year = run_sensitivity_analysis()
        plot_sensitivity(sensitivity_results, test_year)
        plot_sensitivity_matrix(sensitivity_results, test_year)
        
        print("\n8. Full Sensitivity Matrix (all scenarios)...")
        pct_changes, full_results, baselines = run_full_sensitivity_matrix()
        plot_sensitivity_matrix_table(pct_changes)
        
        print("\n9. Summary Table...")
        plot_summary_table(results_df)
        
        print("\n" + "=" * 60)
        print("✓ All plots generated and saved!")
        print("=" * 60)
        print("\nSaved files:")
        print("  - plot_1_arbitrage_per_mwh.png")
        print("  - plot_2_cycles.png")
        print("  - plot_3_change_rates.png")
        print("  - plot_4_exogen_vs_endogen.png")
        print("  - plot_5_yearly_comparison.png")
        print("  - plot_6_sensitivity.png")
        print("  - plot_7_sensitivity_matrix.png")
        print("  - plot_sensitivity_matrix_table.png  ← NEW!")
        print("  - plot_8_summary_table.png")
    else:
        print("ERROR: No results found. Please run Jakob.py first!")
