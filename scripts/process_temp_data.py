import pandas as pd
import os
import shutil

# Config
DATA_DIR = r"e:\pycode\auto-trader-okx\data\history"
TEMP_DIR = os.path.join(DATA_DIR, "temp_2026")
EXISTING_CLEAN = os.path.join(DATA_DIR, "CLEAN_UNIVERSAL_2022_2025.csv")
NEW_CLEAN = os.path.join(DATA_DIR, "CLEAN_UNIVERSAL_2022_2026.csv")

def load_and_merge_daily(symbol, date_str):
    """
    Loads klines and metrics for a specific day and merges them.
    Returns a dataframe.
    """
    # File Patterns
    kline_file = os.path.join(TEMP_DIR, f"{symbol}-1h-{date_str}.csv")
    metrics_file = os.path.join(TEMP_DIR, f"{symbol}-metrics-{date_str}.csv")
    
    # Check existence (unzipped CSVs might be inside a folder if zip contained folder, 
    # normally Binance zips contain the CSV directly)
    # Note: verify if extraction created subfolders or direct files.
    # Usually: BTCUSDT-1h-2025-12-01.zip -> BTCUSDT-1h-2025-12-01.csv
    
    if not os.path.exists(kline_file):
        # Retry check for extracted zip dynamics if zip contained matching folder name
        # But script uses extractall(TARGET_DIR), so likely flat if zip is flat.
        return None

    
    # Load Klines
    # Binance daily CSVs usually HAVE headers: open_time, open, high, low, close, volume, ...
    # Read with default header behavior first
    try:
        df_k = pd.read_csv(kline_file)
        
        # Helper to standardize columns
        def clean_col_names(df):
            df.columns = [c.strip().lower() for c in df.columns]
            rename_map = {
                'open_time': 'timestamp',
                'opentime': 'timestamp',
                'quote_asset_volume': 'quote_vol',
                'number_of_trades': 'trades',
                'taker_buy_base_asset_volume': 'taker_buy_base',
                'taker_buy_quote_asset_volume': 'taker_buy_quote',
                'sum_open_interest': 'open_interest',
                'sum_open_interest_value': 'open_interest_value'
            }
            df.rename(columns=rename_map, inplace=True)
            return df

        df_k = clean_col_names(df_k)
        
        # Verify if timestamp exists
        if 'timestamp' not in df_k.columns:
            # Maybe it had no header? Try reloading with names
            kline_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_vol', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
            df_k = pd.read_csv(kline_file, header=None, names=kline_cols)
        
        # Convert timestamp
        # Binance timestamps are ms int
        df_k['timestamp'] = pd.to_datetime(df_k['timestamp'], unit='ms')
        
        # Filter columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df_k = df_k[required_cols]

    except Exception as e:
        print(f"Error reading kline {kline_file}: {e}")
        return None
    
    # Load Metrics if available
    if os.path.exists(metrics_file):
        try:
            df_m = pd.read_csv(metrics_file)
            df_m = clean_col_names(df_m)
            
            # Metrics usually have 'create_time'
            if 'create_time' in df_m.columns:
                 df_m.rename(columns={'create_time': 'timestamp'}, inplace=True)
            
            if 'timestamp' in df_m.columns:
                 df_m['timestamp'] = pd.to_datetime(df_m['timestamp'])
                 # Drop symbol column if exists
                 if 'symbol' in df_m.columns:
                     df_m.drop(columns=['symbol'], inplace=True)
                 
                 # Resample to 1H
                 df_m.set_index('timestamp', inplace=True)
                 df_m_1h = df_m.resample('1h').last() # Take last snapshot
                 # SHIFT +1H to strictly avoid lookahead bias
                 df_m_1h.index = df_m_1h.index + pd.Timedelta(hours=1)
                 df_m_1h.reset_index(inplace=True)
                 
                 df_merged = pd.merge_asof(
                     df_k.sort_values('timestamp'), 
                     df_m_1h.sort_values('timestamp'), 
                     on='timestamp', 
                     direction='backward', 
                     tolerance=pd.Timedelta('10 minutes')
                 )
            else:
                print(f"Metrics file {metrics_file} lacks timestamp column.")
                df_merged = df_k
        except Exception as e:
            print(f"Error parsing metrics {metrics_file}: {e}")
            df_merged = df_k
    else:
        df_merged = df_k

    # Add Symbol ID
    df_merged['symbol'] = 0.0 if symbol == "BTCUSDT" else 1.0
    
    return df_merged

def main():
    print("Starting Processing of Temp Data...")
    
    # 1. Load Existing Data
    if os.path.exists(EXISTING_CLEAN):
        print("Loading existing clean dataset...")
        df_main = pd.read_csv(EXISTING_CLEAN)
        df_main['timestamp'] = pd.to_datetime(df_main['timestamp'])
    else:
        print("Error: CLEAN_UNIVERSAL_2022_2025.csv not found.")
        return

    # 2. Iterate Temp Files
    # Logic: Get unique dates from file names
    files = os.listdir(TEMP_DIR)
    # Extract dates like 2025-12-01
    dates = []
    # Implementation: dumb scan
    # ... actually simple loop over date range again is safer
    
    new_data = []
    
    # Re-generate same date range as fetch script
    # Or just parse files. 
    # Let's parse files to be robust to partial downloads.
    
    # Find all CSVs
    files = [f for f in files if f.endswith('.csv')]
    
    # Group by (Symbol, Date)
    # Expected: BTCUSDT-1h-2025-12-01.csv
    
    # Just reusing the generator logic is cleaner
    from datetime import date, timedelta
    start_date = date(2025, 12, 1)
    end_date = date(2026, 1, 15)
    
    symbols = ["BTCUSDT", "ETHUSDT"]
    
    curr = start_date
    while curr <= end_date:
        date_str = curr.strftime("%Y-%m-%d")
        for sym in symbols:
            print(f"Processing {sym} {date_str}...")
            df_day = load_and_merge_daily(sym, date_str)
            if df_day is not None and not df_day.empty:
                new_data.append(df_day)
        curr += timedelta(days=1)
        
    if new_data:
        print(f"Merging {len(new_data)} daily chunks...")
        df_new = pd.concat(new_data)
        
        # Concat with Main
        df_final = pd.concat([df_main, df_new])
        
        # Deduplicate (just in case of overlap)
        df_final.drop_duplicates(subset=['timestamp', 'symbol'], keep='last', inplace=True)
        
        # Sort
        df_final.sort_values(by='timestamp', inplace=True)
        
        # Save
        print(f"Saving to {NEW_CLEAN}...")
        df_final.to_csv(NEW_CLEAN, index=False)
        print("Done!")
        
        # Verify 2026
        print("Max Date:", df_final['timestamp'].max())
        
    else:
        print("No new data found/processed.")

if __name__ == "__main__":
    main()
