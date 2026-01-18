import requests
import os
import zipfile
import io
import pandas as pd
from datetime import date, timedelta
import time

# Configuration
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
START_DATE = date(2025, 12, 1)
END_DATE = date(2026, 1, 15)  # Adjust as needed

BASE_URL = "https://data.binance.vision/data/futures/um/daily"
TARGET_DIR = r"e:\pycode\auto-trader-okx\data\history\temp_2026"

def generate_dates(start, end):
    curr = start
    while curr <= end:
        yield curr
        curr += timedelta(days=1)

def download_file(url, target_path):
    if os.path.exists(target_path):
        print(f"Skipping {os.path.basename(target_path)} (already exists)")
        return True
        
    print(f"Downloading {url}...")
    try:
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(target_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        elif r.status_code == 404:
            print(f"Not Found (404): {url}")
            return False
        else:
            print(f"Failed: {r.status_code}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def extract_zip(zip_path, extract_to):
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_to)
        # print(f"Extracted {zip_path}")
        return True
    except zipfile.BadZipFile:
        print(f"Bad Zip: {zip_path}")
        return False

def main():
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)

    # 1. Download Loop
    for single_date in generate_dates(START_DATE, END_DATE):
        date_str = single_date.strftime("%Y-%m-%d")
        
        for symbol in SYMBOLS:
            # --- Type 1: Klines (1h) ---
            # Pattern: https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/1h/BTCUSDT-1h-2025-12-01.zip
            kline_url = f"{BASE_URL}/klines/{symbol}/1h/{symbol}-1h-{date_str}.zip"
            kline_zip = os.path.join(TARGET_DIR, f"{symbol}-1h-{date_str}.zip")
            
            if download_file(kline_url, kline_zip):
                extract_zip(kline_zip, TARGET_DIR)

            # --- Type 2: Metrics ---
            # Pattern: https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-2025-12-01.zip
            metrics_url = f"{BASE_URL}/metrics/{symbol}/{symbol}-metrics-{date_str}.zip"
            metrics_zip = os.path.join(TARGET_DIR, f"{symbol}-metrics-{date_str}.zip")
            
            if download_file(metrics_url, metrics_zip):
                extract_zip(metrics_zip, TARGET_DIR)
            
            # Be nice to the server
            # time.sleep(0.1) 

    print("\nDownload complete. Merging files...")
    merge_data()

def merge_data():
    # Merging logic to be implemented after verifying downloads
    # For now, just listing what we have
    files = os.listdir(TARGET_DIR)
    csv_files = [f for f in files if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files in {TARGET_DIR}")

if __name__ == "__main__":
    main()
