import os
import pandas as pd
from entsoe import EntsoePandasClient

def quick_test(api_key=None):
    api_key = api_key or os.environ.get('ENTSOE_API_KEY')
    if not api_key:
        print('Kein ENTSO-E API Key gefunden. Setze ENV `ENTSOE_API_KEY` oder übergebe als Argument.')
        return 2

    client = EntsoePandasClient(api_key=api_key)
    tz = 'Europe/Berlin'
    end = pd.Timestamp.now(tz=tz).normalize()
    start = end - pd.Timedelta(days=2)

    print(f'Teste Day-Ahead-Abfrage für DE_LU von {start} bis {end}...')
    try:
        prices = client.query_day_ahead_prices('DE_LU', start=start, end=end)
    except Exception as e:
        print('Fehler beim Abfragen der API:')
        print(repr(e))
        return 3

    if prices is None or len(prices) == 0:
        print('Keine Daten zurückgegeben (leeres Ergebnis).')
        return 4

    print('Erfolgreich — Beispielausgabe:')
    print(prices.head())
    print(f'Anzahl Zeilen: {len(prices)}')
    return 0

if __name__ == '__main__':
    import sys
    key = None
    if len(sys.argv) > 1:
        key = sys.argv[1]
    rc = quick_test(api_key=key)
    sys.exit(rc)
