import os
import requests
import zipfile
import io
import pandas as pd
import argparse
import logging
from datetime import datetime

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_URL_SPOT = "https://data.binance.vision/data/spot"
BASE_URL_FUTURES_UM = "https://data.binance.vision/data/futures/um"
BASE_URL_FUTURES_CM = "https://data.binance.vision/data/futures/cm"

def download_and_process_file(symbol, interval, date_obj, save_dir, data_type, market_type, frequency):
    """
    Downloads, extracts, and processes a single file (Daily or Monthly).
    """
    # Date formatting
    if frequency == 'monthly':
        date_str = date_obj.strftime('%Y-%m')
        path_date = date_str
    else: # daily
        date_str = date_obj.strftime('%Y-%m-%d')
        path_date = date_str

    # Base URL
    if market_type == 'spot':
        base = BASE_URL_SPOT
    elif market_type == 'futures': # Default to UM for generics
        base = BASE_URL_FUTURES_UM
    elif market_type == 'futures_cm':
        base = BASE_URL_FUTURES_CM
    else:
        base = BASE_URL_FUTURES_UM

    # URL Construction
    # Pattern: base / frequency / data_type / symbol / interval? / filename
    # Metrics/FundingRate often don't have interval in path for daily/monthly?
    
    if data_type == 'klines':
        # .../klines/BTCUSDT/1h/BTCUSDT-1h-2024-01.zip
        path = f"{frequency}/klines/{symbol}/{interval}/{symbol}-{interval}-{date_str}.zip"
    elif data_type == 'fundingRate':
        # .../fundingRate/BTCUSDT/BTCUSDT-fundingRate-2024-01.zip (Monthly)
        path = f"{frequency}/fundingRate/{symbol}/{symbol}-fundingRate-{date_str}.zip"
    elif data_type == 'metrics':
        # .../metrics/BTCUSDT/BTCUSDT-metrics-2024-01-01.zip (Daily)
        path = f"{frequency}/metrics/{symbol}/{symbol}-metrics-{date_str}.zip"
    
    url = f"{base}/{path}"
    
    # logger.info(f"Downloading: {url}") # Verbose
    
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 404:
            # logger.warning(f"404: {url}")
            return None
        if response.status_code != 200:
            logger.error(f"Failed {url}: {response.status_code}")
            return None
            
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_name = f"{symbol}-{data_type}-{date_str}.csv" if data_type != 'klines' else f"{symbol}-{interval}-{date_str}.csv"
            
            # Simple fallback search
            target_file = None
            for f in z.namelist():
                if f.endswith('.csv'):
                    target_file = f
                    break
            
            if not target_file: return None
                
            with z.open(target_file) as f:
                # Header logic
                # Klines: None
                # FundingRate: 0 (calc_time, ...)
                # Metrics: 0 (create_time, symbol, ...)
                header_arg = None if data_type == 'klines' else 0
                df = pd.read_csv(f, header=header_arg)
                
                # Standardization
                if data_type == 'klines':
                    df = df.iloc[:, :6]
                    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                elif data_type == 'fundingRate':
                    # Expect: calc_time, funding_rate
                    # Use iloc to be safe against column name shifts
                    df = df.iloc[:, [0, 2]] if df.shape[1] > 2 else df.iloc[:, [0, 1]]
                    df.columns = ['timestamp', 'funding_rate']
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

                elif data_type == 'metrics':
                    # Columns: create_time, symbol, sum_open_interest, sum_open_interest_value, ...
                    # We want: create_time -> timestamp, sum_open_interest, count_long_short_ratio
                    # Let's keep all relevant ones
                    
                    # Fix timestamp (create_time is usually "2023-03-30 00:00:00")
                    # It might be string, not ms timestamp?
                    # User snippet: "2023-03-30 00:00:00"
                    
                    if 'create_time' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['create_time'])
                        df.drop(columns=['create_time', 'symbol'], inplace=True, errors='ignore')
                    else:
                        # Fallback
                        df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                return df
                
    except Exception as e:
        logger.error(f"Error {date_str}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Download Binance Data")
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--interval", type=str, default="1h")
    parser.add_argument("--start_year", type=int, required=True)
    parser.add_argument("--end_year", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default="data/history")
    parser.add_argument("--data_type", type=str, default="klines", choices=['klines', 'fundingRate', 'metrics'])
    parser.add_argument("--market_type", type=str, default="futures", choices=['spot', 'futures', 'futures_cm'])
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine Frequency
    # metrics -> daily usually
    # fundingRate -> monthly usually
    # klines -> monthly usually
    frequency = 'daily' if args.data_type == 'metrics' else 'monthly'
    logger.info(f"Downloading {args.data_type} ({frequency}) for {args.symbol}...")
    
    all_dfs = []
    
    # Loop Logic
    start_date = datetime(args.start_year, 1, 1)
    end_date = datetime(args.end_year, 12, 31)
    current_date = start_date
    now = datetime.now()
    
    while current_date <= end_date and current_date <= now:
        # Download
        df = download_and_process_file(
            args.symbol, args.interval, current_date, args.output_dir,
            args.data_type, args.market_type, frequency
        )
        
        if df is not None:
            all_dfs.append(df)
            print(".", end="", flush=True) # Progress bar
        
        # Increment
        if frequency == 'monthly':
            # Next month
            if current_date.month == 12:
                current_date = datetime(current_date.year + 1, 1, 1)
            else:
                current_date = datetime(current_date.year, current_date.month + 1, 1)
        else: # Daily
            current_date += pd.Timedelta(days=1)
            
    print() # Newline

    if not all_dfs:
        logger.error("No data downloaded.")
        return

    logger.info("Merging...")
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df.sort_values('timestamp', inplace=True)
    full_df.drop_duplicates(subset=['timestamp'], inplace=True)
    
    # Save
    if args.data_type == 'fundingRate':
        output_filename = f"{args.symbol}_fundingRate_{args.start_year}_{args.end_year}.csv"
    elif args.data_type == 'metrics':
        output_filename = f"{args.symbol}_metrics_{args.start_year}_{args.end_year}.csv"
    else: # Default for klines and any other data_type
        output_filename = f"{args.symbol}_{args.interval}_{args.start_year}_{args.end_year}.csv"
    
    path = os.path.join(args.output_dir, output_filename)
    full_df.to_csv(path, index=False)
    logger.info(f"✅ Saved {len(full_df)} rows to {path}")

if __name__ == "__main__":
    main()
