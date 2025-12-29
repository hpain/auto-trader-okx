import os
import requests
import zipfile
import io
import pandas as pd
import argparse
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_URL_SPOT = "https://data.binance.vision/data/spot"
BASE_URL_FUTURES_UM = "https://data.binance.vision/data/futures/um"
BASE_URL_FUTURES_CM = "https://data.binance.vision/data/futures/cm"

# Global lock/counter not needed if we process results in main, 
# but for progress printing we can use logic in main loop.

def download_and_process_url(symbol, interval, date_obj, data_type, market_type, frequency, output_dir):
    """
    Worker function to download and process a single date's data.
    Returns: DataFrame or None
    """
    # 1. Date Formatting
    if frequency == 'monthly':
        date_str = date_obj.strftime('%Y-%m')
    else: # daily
        date_str = date_obj.strftime('%Y-%m-%d')

    # 2. Base URL
    if market_type == 'spot':
        base = BASE_URL_SPOT
    elif market_type == 'futures': # Default to UM
        base = BASE_URL_FUTURES_UM
    elif market_type == 'futures_cm':
        base = BASE_URL_FUTURES_CM
    else:
        base = BASE_URL_FUTURES_UM

    # 3. Path Construction
    if data_type == 'klines':
        path = f"{frequency}/klines/{symbol}/{interval}/{symbol}-{interval}-{date_str}.zip"
    elif data_type == 'fundingRate':
        path = f"{frequency}/fundingRate/{symbol}/{symbol}-fundingRate-{date_str}.zip"
    elif data_type == 'metrics':
        path = f"{frequency}/metrics/{symbol}/{symbol}-metrics-{date_str}.zip"
    
    url = f"{base}/{path}"
    
    # 4. Download
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=30) # Increased timeout
            if response.status_code == 404:
                return None
            if response.status_code == 200:
                break # Success
            logger.debug(f"Failed {url}: {response.status_code}")
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                time.sleep(1) # Wait a bit before retry
                continue
            else:
                logger.error(f"Error processing {date_str} after {max_retries} retries: {e}")
                return None
                
    if response.status_code != 200:
        return None
            
    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Find CSV
            target_file = None
            for f in z.namelist():
                if f.endswith('.csv'):
                    target_file = f
                    break
            
            if not target_file: return None
                
            with z.open(target_file) as f:
                # --- HEADER FIX LOGIC ---
                # Peek at the first line to detect header
                first_line = f.readline().decode('utf-8')
                f.seek(0) # Reset pointer
                
                has_header = False
                if 'open_time' in first_line or 'create_time' in first_line or 'calc_time' in first_line:
                    has_header = True

                header_arg = 0 if has_header else None
                
                # Special handling for Klines: sometimes header=None was expected
                # If data_type is klines and no header detected, header=None.
                # If header detected (2024+ files), header=0.
                
                df = pd.read_csv(f, header=header_arg)
                
                # --- STANDARDIZATION ---
                if data_type == 'klines':
                    if has_header:
                        # Columns should be auto-detected, but let's enforce standard names
                        # Standard Binance: open_time, open, high, low, close, volume, ...
                        # We only need first 6 usually
                        # Rename to standard internal names
                        df.rename(columns={
                            'open_time': 'timestamp', 
                            'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'
                        }, inplace=True)
                        # Keep only relevant
                        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    else:
                        # Old style, no header
                        df = df.iloc[:, :6]
                        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                elif data_type == 'fundingRate':
                    # Expect: calc_time, funding_rate
                    # If header exists: 'calc_time', 'funding_rate'
                    if has_header:
                         df.rename(columns={'calc_time': 'timestamp', 'last_funding_rate': 'funding_rate'}, inplace=True)
                         df = df[['timestamp', 'funding_rate']]
                    else:
                         df = df.iloc[:, [0, 2]] if df.shape[1] > 2 else df.iloc[:, [0, 1]]
                         df.columns = ['timestamp', 'funding_rate']
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

                elif data_type == 'metrics':
                    # Columns: create_time, ...
                    if has_header:
                         df.rename(columns={'create_time': 'timestamp'}, inplace=True)
                         # Drop symbol if exists
                         if 'symbol' in df.columns: df.drop(columns=['symbol'], inplace=True)
                         # Rename sum_open_interest
                         df.rename(columns={'sum_open_interest': 'open_interest', 'sum_open_interest_value': 'open_interest_value'}, inplace=True)
                    else:
                        # Fallback (rare for metrics to not have header but just in case)
                        df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)
                        if len(df.columns) > 1 and isinstance(df.iloc[0,1], str) and len(df.iloc[0,1]) > 5:
                             # Assume col 1 is symbol
                             df = df.iloc[:, [0] + list(range(2, len(df.columns)))]

                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                return df
                
    except Exception as e:
        logger.error(f"Error processing {date_str}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Download Binance Data (Parallel)")
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--interval", type=str, default="1h")
    parser.add_argument("--start_year", type=int, required=True)
    parser.add_argument("--end_year", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default="data/history")
    parser.add_argument("--data_type", type=str, default="klines", choices=['klines', 'fundingRate', 'metrics'])
    parser.add_argument("--market_type", type=str, default="futures", choices=['spot', 'futures', 'futures_cm'])
    parser.add_argument("--workers", type=int, default=20, help="Number of parallel download threads")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    frequency = 'daily' if args.data_type == 'metrics' else 'monthly'
    
    # 1. Generate Date List
    dates = []
    start_date = datetime(args.start_year, 1, 1)
    end_date = datetime(args.end_year, 12, 31)
    current_date = start_date
    now = datetime.now()
    
    while current_date <= end_date and current_date <= now:
        dates.append(current_date)
        if frequency == 'monthly':
            if current_date.month == 12:
                current_date = datetime(current_date.year + 1, 1, 1)
            else:
                current_date = datetime(current_date.year, current_date.month + 1, 1)
        else:
            current_date += pd.Timedelta(days=1)
            
    logger.info(f"Downloading {args.data_type} ({frequency}) for {args.symbol} from {args.start_year} to {args.end_year}...")
    logger.info(f"Total files to check: {len(dates)}")
    
    all_dfs = []
    
    # 2. Parallel Download
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_date = {
            executor.submit(
                download_and_process_url, 
                args.symbol, args.interval, d, args.data_type, args.market_type, frequency, args.output_dir
            ): d for d in dates
        }
        
        # Process as they complete
        completed_count = 0
        total_count = len(dates)
        
        for future in as_completed(future_to_date):
            d = future_to_date[future]
            try:
                df = future.result()
                if df is not None:
                    all_dfs.append(df)
            except Exception as exc:
                logger.error(f"{d} needs help: {exc}")
            
            completed_count += 1
            # Simple progress bar
            if completed_count % 10 == 0 or completed_count == total_count:
                print(f"\rProgress: {completed_count}/{total_count} ({(completed_count/total_count)*100:.1f}%)", end="")
    
    print() # Newline
    
    if not all_dfs:
        logger.warning(f"No data downloaded for {args.symbol} {args.data_type}.")
        return

    logger.info("Merging downloaded chunks...")
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df.sort_values('timestamp', inplace=True)
    full_df.drop_duplicates(subset=['timestamp'], inplace=True)
    
    # Save
    if args.data_type == 'fundingRate':
        output_filename = f"{args.symbol}_fundingRate_{args.start_year}_{args.end_year}.csv"
    elif args.data_type == 'metrics':
        output_filename = f"{args.symbol}_metrics_{args.start_year}_{args.end_year}.csv"
    else: 
        output_filename = f"{args.symbol}_{args.interval}_{args.start_year}_{args.end_year}.csv"
    
    path = os.path.join(args.output_dir, output_filename)
    full_df.to_csv(path, index=False)
    logger.info(f"✅ Saved {len(full_df)} rows to {path}")

if __name__ == "__main__":
    main()
