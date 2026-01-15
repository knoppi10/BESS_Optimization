import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress
from scipy.optimize import curve_fit

def load_data(filepath='market_data_2019_2025.csv'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filepath)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Datei nicht gefunden: {filepath}")
    
    print(f"Loading data from {filepath}...")

    # 1. Einlesen ohne sofortigen Index-Wechsel
    # 'utf-8-sig' hilft, falls versteckte Zeichen (BOM) am Anfang der Datei stehen
    df = pd.read_csv(filepath, encoding='utf-8-sig')

    # 2. Umwandlung in Datetime (während es noch eine Spalte ist)
    # utc=True erkennt dein +02:00 Format automatisch
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # 3. Jetzt die Zeitzone auf Berlin konvertieren
    df['timestamp'] = df['timestamp'].dt.tz_convert('Europe/Berlin')

    # 4. Erst ganz am Ende zum Index machen
    df.set_index('timestamp', inplace=True)
    
    return df

def analyze_empirical_slope(df: pd.DataFrame):
    """
    Calculates the empirical market slope by binning the data and performing
    a linear regression on each bin.
    """
    print("Analyzing empirical slope from data...")
    
    # 1. Define bins for residual load (e.g., in 2 GW steps)
    min_load = df['residual_load'].min()
    max_load = df['residual_load'].max()
    bin_width = 2000  # 2000 MW = 2 GW
    bins = np.arange(min_load, max_load + bin_width, bin_width)
    
    # 2. Assign each data point to a bin
    df['load_bin'] = pd.cut(df['residual_load'], bins=bins, right=False)

    results = []
    # 3. Group by bins and calculate slope for each
    for bin_interval, group in df.groupby('load_bin'):
        # Ensure we have enough data points for a meaningful regression
        if len(group) < 50:
            continue
            
        # 4. Perform linear regression: price = slope * residual_load + intercept
        regression = linregress(x=group['residual_load'], y=group['price_da'])
        
        # The slope of the regression is our empirical market slope for this bin
        slope = regression.slope
        
        # We use the midpoint of the bin as our x-value for plotting
        bin_midpoint = bin_interval.mid
        
        results.append({'residual_load': bin_midpoint, 'empirical_slope': slope})

    return pd.DataFrame(results)

def sigmoid_func(x, slope_min, slope_max, inflection, sensitivity):
    """Die Sigmoid-Funktion, die wir fitten wollen."""
    sigmoid = 1 / (1 + np.exp(-sensitivity * (x - inflection)))
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

def demonstrate_granularity_issue(df):
    """
    Zeigt visuell, warum man nicht auf Tagesbasis fitten kann.
    Vergleicht einen einzelnen Tag mit einem ganzen Jahr.
    """
    # Wähle einen beispielhaften Tag und ein Jahr
    sample_day = '2023-11-07' # Ein Tag mit Bewegung, aber nicht alles
    sample_year = 2023
    
    day_data = df.loc[sample_day]
    year_data = df[df.index.year == sample_year]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True, sharex=True)
    
    # Plot 1: Ein Tag
    axes[0].scatter(day_data['residual_load'], day_data['price_da'], color='blue', alpha=0.6)
    axes[0].set_title(f"Sichtfeld eines Tages ({sample_day})")
    axes[0].set_xlabel("Residuallast (MW)")
    axes[0].set_ylabel("Strompreis (€/MWh)")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Ein Jahr
    axes[1].scatter(year_data['residual_load'], year_data['price_da'], color='black', alpha=0.05, s=2)
    axes[1].set_title(f"Sichtfeld eines Jahres ({sample_year})")
    axes[1].set_xlabel("Residuallast (MW)")
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle("Warum wir jährlich kalibrieren: Man braucht die volle Bandbreite", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_yearly_slope_analysis(full_df: pd.DataFrame):
    """
    Führt die Slope-Analyse für jedes Jahr durch und plottet die Ergebnisse.
    """
    years = sorted(full_df.index.year.unique())
    # Erstelle ein Grid von Subplots
    fig, axes = plt.subplots(len(years), 1, figsize=(10, 5 * len(years)), sharex=True, sharey=True)
    fig.suptitle('Evolution der Markt-Slope-Kurve über die Jahre', fontsize=16)

    for i, year in enumerate(years):
        ax = axes[i]
        year_df = full_df[full_df.index.year == year]
        empirical_slopes = analyze_empirical_slope(year_df)
        
        if empirical_slopes.empty:
            ax.set_title(f"{year} - Nicht genügend Daten")
            continue

        # Plot der empirischen Datenpunkte
        ax.scatter(empirical_slopes['residual_load'], empirical_slopes['empirical_slope'], 
                   color='black', alpha=0.6, label='Empirischer Slope (aus Daten)')

        # Fitte eine Sigmoid-Kurve an die empirischen Daten
        popt, _ = curve_fit(sigmoid_func, empirical_slopes['residual_load'], empirical_slopes['empirical_slope'], 
                            p0=[0.001, 0.05, 40000, 0.0001], maxfev=5000)
        
        # Plot der gefitteten Kurve
        x_smooth = np.linspace(empirical_slopes['residual_load'].min(), empirical_slopes['residual_load'].max(), 500)
        ax.plot(x_smooth, sigmoid_func(x_smooth, *popt), color='red', linewidth=2, 
                linestyle='--', label=f'Gefittete Kurve (Jahr {year})')

        ax.set_title(f"Jahr {year}")
        ax.set_ylabel('Slope (€/MW)')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend()

    axes[-1].set_xlabel('Residuallast (MW)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_seasonal_slope_analysis(full_df: pd.DataFrame):
    """
    Führt die Slope-Analyse für jede Jahreszeit durch (aggregiert) und plottet die Ergebnisse.
    """
    full_df['season'] = full_df.index.month.map(get_season)
    seasons = ['Winter', 'Frühling', 'Sommer', 'Herbst']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = {'Winter': 'blue', 'Frühling': 'green', 'Sommer': 'red', 'Herbst': 'orange'}

    ax.set_title('Saisonale Markt-Slope-Kurven (Aggregiert 2019-2025)', fontsize=16)

    for season in seasons:
        print(f"Analysiere Saison: {season}")
        season_df = full_df[full_df['season'] == season]
        empirical_slopes = analyze_empirical_slope(season_df)
        
        if empirical_slopes.empty: continue

        try:
            popt, _ = curve_fit(sigmoid_func, empirical_slopes['residual_load'], empirical_slopes['empirical_slope'], 
                                p0=[0.001, 0.05, 40000, 0.0001], maxfev=10000)
            
            x_smooth = np.linspace(full_df['residual_load'].min(), full_df['residual_load'].max(), 500)
            ax.plot(x_smooth, sigmoid_func(x_smooth, *popt), color=colors[season], linewidth=2.5, 
                    label=f'{season}')
            
            # Optional: Scatter plot der Punkte, um den Fit zu prüfen (kann man auskommentieren)
            # ax.scatter(empirical_slopes['residual_load'], empirical_slopes['empirical_slope'], color=colors[season], alpha=0.2, s=10)
            
        except RuntimeError:
            print(f"Fehler beim Fitten von {season}")

    ax.set_xlabel('Residuallast (MW)')
    ax.set_ylabel('Preissensitivität (Slope) [€/MW]')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    market_df = load_data()
    # demonstrate_granularity_issue(market_df)
    # plot_yearly_slope_analysis(market_df)
    plot_seasonal_slope_analysis(market_df)