
# tools/download_data.py
import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from data.binance import get_klines_bian

def download_latest_data():
    """
    Downloads the latest 1 year of 1-hour BTC-USDT data from Binance.
    """
    print("--- Starting data download ---")
    get_klines_bian(
        client=None,          # The function handles requests internally, no client needed
        symbol='BTC-USDT',
        interval='1h',
        years=1,
        ignore_local=True,    # Force a fresh download, ignoring any existing file
        save=True
    )
    print("--- Data download finished ---")

if __name__ == "__main__":
    download_latest_data()
