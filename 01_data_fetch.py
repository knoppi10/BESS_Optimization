"""ENTSO-E Market Data Fetcher"""

import pandas as pd
import os
from entsoe import EntsoePandasClient
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('ENTSOE_API_KEY')

COUNTRY_CODE = 'DE_LU'
START_DATE = '2019-01-01'
END_DATE = '2025-12-31'
TIMEZONE = 'Europe/Berlin'

def fetch_and_process_data():
    if not API_KEY or API_KEY == 'YOUR_API_KEY_HERE':
        raise ValueError("ENTSO-E API key not found. Create a .env file with your key.")

    print("Connecting to ENTSO-E...")
    client = EntsoePandasClient(api_key=API_KEY)
    start = pd.Timestamp(START_DATE, tz=TIMEZONE)
    end = pd.Timestamp(END_DATE, tz=TIMEZONE)

    print(f"Fetching day-ahead prices ({START_DATE} to {END_DATE})...")
    prices = client.query_day_ahead_prices(COUNTRY_CODE, start=start, end=end)
    prices.name = 'price_da'

    print("Fetching load...")
    load = client.query_load(COUNTRY_CODE, start=start, end=end)
    if isinstance(load, pd.DataFrame):
        load = load.iloc[:, 0]
    load.name = 'load'

    print("Fetching renewables (wind & solar)...")
    generation = client.query_generation(COUNTRY_CODE, start=start, end=end, psr_type=None)
    gen_cols = [c for c in generation.columns if 'Solar' in str(c) or 'Wind' in str(c)]
    renewable_total = generation[gen_cols].sum(axis=1)
    renewable_total.name = 'generation_renewable'

    print("Merging...")
    df = pd.DataFrame(prices)
    df = df.join(load.resample('h').mean())
    df = df.join(renewable_total.resample('h').mean())
    df = df.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    df['residual_load'] = df['load'] - df['generation_renewable']

    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, '04_market_data_2019_2025.csv')
    df.index.name = 'timestamp'
    df.to_csv(filename)
    print(f"Done! Saved to: {filename}")
    print(df.head())

if __name__ == "__main__":
    fetch_and_process_data()