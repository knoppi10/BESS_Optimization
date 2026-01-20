"""
Kostenanalyse: Nachberechnung der tatsächlichen Kosten aus Optimierungsergebnissen

Liest die Optimierungsergebnisse (simulation_decisions_multiyear.csv) und die Marktdaten
und berechnet detailliert:
- Energieerlös (Arbitrage)
- Price Impact Kosten
- Hurdle-Rate Kosten (Durchsatz)
- Effizienzverluste (Energieverlust × Preis)

Parameter werden direkt aus Jakob.py SimulationConfig importiert!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Importiere Config und Konstanten aus Jakob.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Jakob import SimulationConfig, MARKET_DATA_FILE, YEARS, SCENARIOS

# ==========================================
# PARAMETER AUS JAKOB.PY LADEN
# ==========================================
cfg = SimulationConfig()

ALPHA = cfg.slope  # Price Impact [€/(MW²·h)]
HURDLE_RATE = cfg.hurdle_rate  # [€/MWh Durchsatz]
ETA_CH = cfg.efficiency_charge  # Lade-Effizienz
ETA_DIS = cfg.efficiency_discharge  # Entlade-Effizienz
ETA_RT = cfg.roundtrip_efficiency  # Rundreiseeffizienz
DT_HOURS = 1.0  # Zeitauflösung (wird später erkannt)

print(f"Parameter aus Jakob.py geladen:")
print(f"  slope (α):              {ALPHA}")
print(f"  hurdle_rate:            {HURDLE_RATE} €/MWh")
print(f"  efficiency_charge:      {ETA_CH:.4f}")
print(f"  efficiency_discharge:   {ETA_DIS:.4f}")
print(f"  roundtrip_efficiency:   {ETA_RT}")
print(f"  market_data_file:       {MARKET_DATA_FILE}")
print(f"  years:                  {YEARS}")
print(f"  scenarios:              {SCENARIOS}")

# ==========================================
# DATEN LADEN
# ==========================================
def load_data():
    """Lade Optimierungsergebnisse und Marktdaten"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Optimierungsergebnisse
    decisions_path = os.path.join(script_dir, 'simulation_decisions_multiyear.csv')
    df_decisions = pd.read_csv(decisions_path)
    df_decisions['timestamp'] = pd.to_datetime(df_decisions['timestamp'], utc=True)
    df_decisions = df_decisions.set_index('timestamp')
    
    # Marktdaten (nutze MARKET_DATA_FILE aus Jakob.py)
    market_path = os.path.join(script_dir, MARKET_DATA_FILE)
    df_market = pd.read_csv(market_path)
    df_market.columns = df_market.columns.str.strip()
    
    if 'timestamp' in df_market.columns:
        df_market['timestamp'] = pd.to_datetime(df_market['timestamp'], utc=True)
    else:
        first_col = df_market.columns[0]
        df_market[first_col] = pd.to_datetime(df_market[first_col], utc=True)
        df_market = df_market.rename(columns={first_col: 'timestamp'})
    
    df_market = df_market.set_index('timestamp')
    df_market.index = df_market.index.tz_convert('Europe/Berlin')
    
    # Filter auf konfigurierten Zeitraum (aus YEARS)
    min_year = min(YEARS)
    max_year = max(YEARS)
    df_market = df_market[(df_market.index.year >= min_year) & (df_market.index.year <= max_year)]
    
    return df_decisions, df_market

# ==========================================
# KOSTENBERECHNUNG
# ==========================================
def calculate_costs(df_decisions, df_market):
    """
    Berechne Kosten für alle Szenarien
    
    ANNAHME: net_discharge = d_t - c_t (aus CSV)
    Da wir nur Nettofluss haben, nehmen wir an:
    - Wenn net_discharge > 0: reiner Entlade-Schritt
    - Wenn net_discharge < 0: reiner Lade-Schritt
    (Dies sollte ökonomisch optimal sein)
    """
    
    results = []
    summary_by_scenario = {}
    
    print("Berechne Kosten für alle Szenarien...")
    print("=" * 80)
    
    for scenario, (capacity, slope) in SCENARIOS.items():
        if scenario not in df_decisions.columns:
            print(f"Warnung: Szenario {scenario} nicht gefunden")
            continue
        
        net_discharge = df_decisions[scenario].values  # MW
        prices = df_market['price_da'].values  # €/MWh
        
        # Sicherstelle gleiche Länge
        min_len = min(len(net_discharge), len(prices))
        net_discharge = net_discharge[:min_len]
        prices = prices[:min_len]
        timestamps = df_market.index[:min_len]
        
        # ===== ENERGIEERLÖS (Arbitrage) =====
        # Wenn net_discharge > 0: Entladen und verkaufen
        # Wenn net_discharge < 0: Laden und kaufen (negativer Erlös)
        energy_revenue = DT_HOURS * net_discharge * prices
        
        # ===== PRICE IMPACT =====
        # Kosten durch Markteinfluss: alpha * dt * (net_power)^2
        # WICHTIG: slope ist szenario-spezifisch! Ex=0 (Price Taker), andere=0.05
        price_impact = slope * DT_HOURS * (net_discharge ** 2)
        
        # ===== DURCHSATZ & HURDLE-KOSTEN =====
        # Durchsatz = |Laden| + |Entladen| = 2 * |net_discharge| wenn nur Ein-Richtung
        throughput = DT_HOURS * np.abs(net_discharge)  # MWh
        hurdle_costs = HURDLE_RATE * throughput
        
        # ===== EFFIZIENZVERLUSTE =====
        # Wenn entladen (d_t > 0): Verlust = d_t * (1 - eta_dis)
        # Wenn laden (c_t > 0): Verlust = c_t * (1/eta_ch - 1)
        efficiency_loss = np.zeros_like(net_discharge)
        
        for i in range(len(net_discharge)):
            if net_discharge[i] > 0:  # Entladen
                loss_mwh = net_discharge[i] * (1 - ETA_DIS)
                efficiency_loss[i] = loss_mwh * prices[i]
            elif net_discharge[i] < 0:  # Laden
                loss_mwh = np.abs(net_discharge[i]) * (1 / ETA_CH - 1)
                efficiency_loss[i] = loss_mwh * prices[i]
        
        # ===== NETTO-ERGEBNIS =====
        net_profit = energy_revenue - price_impact - hurdle_costs - efficiency_loss
        
        # ===== SPEICHERN =====
        for i in range(len(net_discharge)):
            results.append({
                'timestamp': timestamps[i],
                'scenario': scenario,
                'price': prices[i],
                'net_power': net_discharge[i],
                'energy_revenue': energy_revenue[i],
                'price_impact': price_impact[i],
                'hurdle_costs': hurdle_costs[i],
                'efficiency_loss': efficiency_loss[i],
                'net_profit': net_profit[i]
            })
        
        # ===== SZENARIO-ZUSAMMENFASSUNG =====
        total_energy_revenue = np.sum(energy_revenue)
        total_price_impact = np.sum(price_impact)
        total_hurdle = np.sum(hurdle_costs)
        total_efficiency_loss = np.sum(efficiency_loss)
        total_net_profit = np.sum(net_profit)
        total_throughput = np.sum(throughput)
        cycles = total_throughput / (2.0 * capacity) if capacity > 0 else 0
        
        summary_by_scenario[scenario] = {
            'Kapazität (MWh)': capacity,
            'slope (α)': slope,
            'Durchsatz (MWh)': total_throughput,
            'Zyklen': cycles,
            'Energieerlös (€)': total_energy_revenue,
            'Price Impact (€)': total_price_impact,
            'Hurdle-Kosten (€)': total_hurdle,
            'Effizienzverluste (€)': total_efficiency_loss,
            'Netto-Gewinn (€)': total_net_profit,
        }
    
    df_results = pd.DataFrame(results)
    return df_results, summary_by_scenario

# ==========================================
# AUSGABE
# ==========================================
def print_summary(summary):
    """Gebe Zusammenfassung aus"""
    print("\n" + "=" * 100)
    print("KOSTENANALYSE - SZENARIO-ZUSAMMENFASSUNG")
    print("=" * 100)
    
    summary_df = pd.DataFrame(summary).T
    summary_df = summary_df[[
        'Kapazität (MWh)', 'Durchsatz (MWh)', 'Zyklen', 
        'Energieerlös (€)', 'Price Impact (€)', 'Hurdle-Kosten (€)', 
        'Effizienzverluste (€)', 'Netto-Gewinn (€)'
    ]]
    
    print(summary_df.to_string())
    print("=" * 100)
    
    # Kostenaufschlüsselung in Prozent
    print("\nKOSTENAUFSCHLÜSSELUNG (% des Energieerlös):")
    print("-" * 100)
    for scenario in summary.keys():
        erlös = summary[scenario]['Energieerlös (€)']
        if erlös > 0:
            impact_pct = 100 * summary[scenario]['Price Impact (€)'] / erlös
            hurdle_pct = 100 * summary[scenario]['Hurdle-Kosten (€)'] / erlös
            eff_pct = 100 * summary[scenario]['Effizienzverluste (€)'] / erlös
            netto_pct = 100 * summary[scenario]['Netto-Gewinn (€)'] / erlös
            
            print(f"\n{scenario}:")
            print(f"  Energieerlös:        100.0%  (€ {erlös:>15,.0f})")
            print(f"  Price Impact:       -{impact_pct:>6.1f}%  (€ {summary[scenario]['Price Impact (€)']:>15,.0f})")
            print(f"  Hurdle-Kosten:      -{hurdle_pct:>6.1f}%  (€ {summary[scenario]['Hurdle-Kosten (€)']:>15,.0f})")
            print(f"  Effizienzverluste:  -{eff_pct:>6.1f}%  (€ {summary[scenario]['Effizienzverluste (€)']:>15,.0f})")
            print(f"  Netto-Gewinn:       {netto_pct:>6.1f}%  (€ {summary[scenario]['Netto-Gewinn (€)']:>15,.0f})")

# ==========================================
# PLOTS
# ==========================================
def plot_cost_breakdown(summary):
    """
    Cost breakdown plot: Stacked bar chart in % of energy revenue
    """
    scenarios = list(summary.keys())
    
    # Berechne Prozentanteile
    netto_pct = []
    eff_pct = []
    hurdle_pct = []
    impact_pct = []
    
    for scenario in scenarios:
        erlös = summary[scenario]['Energieerlös (€)']
        
        # Prozent
        if erlös > 0:
            impact_pct.append(100 * summary[scenario]['Price Impact (€)'] / erlös)
            hurdle_pct.append(100 * summary[scenario]['Hurdle-Kosten (€)'] / erlös)
            eff_pct.append(100 * summary[scenario]['Effizienzverluste (€)'] / erlös)
            netto_pct.append(100 * summary[scenario]['Netto-Gewinn (€)'] / erlös)
        else:
            impact_pct.append(0)
            hurdle_pct.append(0)
            eff_pct.append(0)
            netto_pct.append(0)
    
    # Single plot
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    ax.set_facecolor('white')
    
    x = np.arange(len(scenarios))
    width = 0.6
    
    # Gray shades for costs (to distinguish from battery size colors)
    # Net Profit stays green, costs in graduated grays (light to dark)
    ax.bar(x, netto_pct, width, label='Net Profit', color='#2ecc71')
    ax.bar(x, eff_pct, width, bottom=netto_pct, label='Efficiency Losses', color='#b0b0b0')  # Light gray
    ax.bar(x, hurdle_pct, width, bottom=np.array(netto_pct)+np.array(eff_pct), 
            label='Hurdle Costs', color='#707070')  # Medium gray
    ax.bar(x, impact_pct, width, 
            bottom=np.array(netto_pct)+np.array(eff_pct)+np.array(hurdle_pct),
            label='Price Impact', color='#404040')  # Dark gray
    
    ax.set_ylabel('Share of Energy Revenue (%)', fontsize=12)
    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_title('Cost Breakdown by Scenario (% of Gross Energy Revenue)\n2019-2024', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=11)
    ax.set_ylim(0, 105)
    ax.axhline(y=100, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Prozentwerte in die Balken schreiben
    for i, scenario in enumerate(scenarios):
        # Netto-Gewinn
        if netto_pct[i] > 5:
            ax.text(i, netto_pct[i]/2, f'{netto_pct[i]:.1f}%', ha='center', va='center', 
                    fontsize=10, fontweight='bold', color='white')
        # Effizienz
        if eff_pct[i] > 5:
            ax.text(i, netto_pct[i] + eff_pct[i]/2, f'{eff_pct[i]:.1f}%', ha='center', va='center',
                    fontsize=10, color='white')
        # Hurdle
        if hurdle_pct[i] > 5:
            ax.text(i, netto_pct[i] + eff_pct[i] + hurdle_pct[i]/2, f'{hurdle_pct[i]:.1f}%', 
                    ha='center', va='center', fontsize=10, color='white')
        # Price Impact
        if impact_pct[i] > 5:
            ax.text(i, netto_pct[i] + eff_pct[i] + hurdle_pct[i] + impact_pct[i]/2, 
                    f'{impact_pct[i]:.1f}%', ha='center', va='center', fontsize=10, color='white')
    
    plt.tight_layout()
    
    # Save as PNG
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'cost_breakdown.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"\n✓ Plot gespeichert: {output_path}")
    
    plt.show()
    
    return fig


# ==========================================
# HAUPTPROGRAMM
# ==========================================
if __name__ == "__main__":
    print("Lade Daten...")
    df_decisions, df_market = load_data()
    
    print(f"Optimierungsergebnisse: {df_decisions.shape}")
    print(f"Marktdaten: {df_market.shape}")
    print(f"Gemeinsamer Zeitraum: {df_decisions.index.min()} bis {df_decisions.index.max()}")
    
    # ===== DURCHSCHNITTSPREISE BERECHNEN =====
    avg_price_total = df_market['price_da'].mean()
    std_price_total = df_market['price_da'].std()
    min_price_total = df_market['price_da'].min()
    max_price_total = df_market['price_da'].max()
    
    # Januar 2019
    jan_2019_mask = (df_market.index.year == 2019) & (df_market.index.month == 1)
    jan_2019_prices = df_market.loc[jan_2019_mask, 'price_da']
    avg_price_jan2019 = jan_2019_prices.mean()
    std_price_jan2019 = jan_2019_prices.std()
    min_price_jan2019 = jan_2019_prices.min()
    max_price_jan2019 = jan_2019_prices.max()
    
    print("\n" + "=" * 60)
    print("DAY-AHEAD PRICE STATISTICS")
    print("=" * 60)
    print(f"\nEntire Period ({YEARS[0]}-{YEARS[-1]}):")
    print(f"  Average $p_t$:  {avg_price_total:>8.2f} €/MWh")
    print(f"  Std Dev:        {std_price_total:>8.2f} €/MWh")
    print(f"  Min:            {min_price_total:>8.2f} €/MWh")
    print(f"  Max:            {max_price_total:>8.2f} €/MWh")
    print(f"  Data Points:    {len(df_market):>8,} hours")
    
    print(f"\nJanuary 2019 only:")
    print(f"  Average $p_t$:  {avg_price_jan2019:>8.2f} €/MWh")
    print(f"  Std Dev:        {std_price_jan2019:>8.2f} €/MWh")
    print(f"  Min:            {min_price_jan2019:>8.2f} €/MWh")
    print(f"  Max:            {max_price_jan2019:>8.2f} €/MWh")
    print(f"  Data Points:    {len(jan_2019_prices):>8,} hours")
    print("=" * 60)
    
    # Berechne Kosten
    df_results, summary = calculate_costs(df_decisions, df_market)
    
    # Speichere detaillierte Ergebnisse
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cost_analysis_detailed.csv')
    df_results.to_csv(output_path, index=False)
    print(f"\n✓ Detaillierte Ergebnisse gespeichert: {output_path}")
    
    # Ausgabe Zusammenfassung
    print_summary(summary)
    
    # Speichere Zusammenfassung
    summary_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cost_analysis_summary.csv')
    pd.DataFrame(summary).T.to_csv(summary_path)
    
    # Plot
    plot_cost_breakdown(summary)
    print(f"✓ Zusammenfassung gespeichert: {summary_path}")
