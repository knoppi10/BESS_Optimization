# @CELIA Data Fetch Script for ENTSO-E Market Data
# This script fetches day-ahead prices, load, and renewable generation data
# from the ENTSO-E Transparency Platform using the EntsoePandasClient.
# It processes the data to compute residual load and saves the final dataset
# to a CSV file for further analysis.



import pandas as pd
import os
from entsoe import EntsoePandasClient
from dotenv import load_dotenv

# ------------------------------------------------------------------------------
# KONFIGURATION
# ------------------------------------------------------------------------------
# Lade Umgebungsvariablen aus der .env-Datei.
# Diese Datei wird NICHT im Git-Repository gespeichert und enthält den API-Schlüssel.
load_dotenv()

# Hole den API-Schlüssel aus der Umgebungsvariable.
API_KEY = os.getenv('ENTSOE_API_KEY')

COUNTRY_CODE = 'DE_LU'  # Bidding Zone Deutschland/Luxemburg
START_DATE = '2019-01-01'
END_DATE = '2025-12-31'
TIMEZONE = 'Europe/Berlin'

def fetch_and_process_data():
    if not API_KEY or API_KEY == 'YOUR_API_KEY_HERE':
        raise ValueError(
            "ENTSO-E API Key nicht gefunden oder nicht gesetzt. "
            "Bitte erstellen Sie eine `.env` Datei (basierend auf der `.env.example` Vorlage) "
            "und tragen Sie dort Ihren persönlichen API-Schlüssel ein."
        )

    print(f"Verbinde zu ENTSO-E Client...")
    client = EntsoePandasClient(api_key=API_KEY)

    start = pd.Timestamp(START_DATE, tz=TIMEZONE)
    end = pd.Timestamp(END_DATE, tz=TIMEZONE)

    # 1. Day-Ahead Preise abrufen
    print(f"Lade Day-Ahead Preise für {COUNTRY_CODE} ({START_DATE} bis {END_DATE})...")
    prices = client.query_day_ahead_prices(COUNTRY_CODE, start=start, end=end)
    prices.name = 'price_da'

    # 2. Last (Load) abrufen
    print("Lade Stromverbrauch (Load)...")
    load = client.query_load(COUNTRY_CODE, start=start, end=end)
    # Load ist manchmal ein DataFrame, wir brauchen die 'Actual Load' Spalte
    if isinstance(load, pd.DataFrame):
        # Nimm die Spalte, die am ehesten nach "Actual" aussieht (oder die erste)
        load = load.iloc[:, 0]
    load.name = 'load'

    # 3. Erneuerbare Erzeugung (Wind & Solar) abrufen
    print("Lade Erzeugung (Wind & Solar)...")
    generation = client.query_generation(COUNTRY_CODE, start=start, end=end, psr_type=None)
    
    # Filtern und Summieren der relevanten Spalten
    # Wir suchen nach 'Solar', 'Wind Onshore', 'Wind Offshore'
    gen_cols = [c for c in generation.columns if 'Solar' in str(c) or 'Wind' in str(c)]
    renewable_total = generation[gen_cols].sum(axis=1)
    renewable_total.name = 'generation_renewable'

    # 4. Zusammenfügen (Merge)
    print("Füge Daten zusammen...")
    # Wir nutzen 'prices' als Basis-Index (stündlich)
    df = pd.DataFrame(prices)
    
    # Resample auf Stundenbasis (falls Load/Gen in 15min Auflösung kommen)
    df = df.join(load.resample('h').mean())
    df = df.join(renewable_total.resample('h').mean())

    # Datenlücken füllen (Interpolieren)
    df = df.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

    # 5. Residuallast berechnen (für unsere fundamentale Slope-Berechnung)
    df['residual_load'] = df['load'] - df['generation_renewable']

    # Speichern
    # Speichert die Datei im gleichen Ordner wie dieses Skript
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, 'market_data_2019_2025.csv')
    df.index.name = 'timestamp'
    df.to_csv(filename)
    print(f"Fertig! Daten gespeichert in: {filename}")
    
    # Kurzer Blick auf die Daten
    print("\nErste 5 Zeilen:")
    print(df.head())

if __name__ == "__main__":
    fetch_and_process_data()