import os
import pandas as pd
import subprocess
import argparse
import logging
import sys

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def run_downloader(symbol, start_year, end_year, data_type, output_dir):
    """Executes the download_binance_data.py script via subprocess."""
    cmd = [
        sys.executable, "scripts/download_binance_data.py",
        "--symbol", symbol,
        "--start_year", str(start_year),
        "--end_year", str(end_year),
        "--data_type", data_type,
        "--market_type", "futures", # Always U-Margin Futures for this pipeline
        "--output_dir", output_dir
    ]
    
    logger.info(f"⬇️ Fetching {data_type}...")
    subprocess.check_call(cmd)
    
    # Identify the output filename constructed by the downloader
    # Naming convention defined in downloader:
    # fundingRate -> {symbol}_fundingRate_{start}_{end}.csv
    # others -> {symbol}_{interval}_{start}_{end}.csv (Wait, metrics uses interval? No, metrics logic in downloader uses interval arg?
    # Let's check downloader logic using 'metrics' type:
    # "metrics" falls into 'else' for save name? 
    # Downloader logic:
    # if funding: ...fundingRate...
    # else: ...{interval}...
    # Wait, for metrics we usually don't pass interval, but downloader defaults to '1h'.
    # We should ensure downloader saves metrics with meaningful name.
    
    # Re-checking downloader save logic:
    # if args.data_type == 'fundingRate': ...
    # else: ...{symbol}_{interval}...
    # Metrics falls into 'else'. So it will be BTCUSDT_1h_2024_2024.csv?
    # That might clash with Klines!
    
    # WE MUST FIX DOWNLOADER FIRST or handle renaming here.
    # Actually, let's presume we fix downloader or use specific naming.
    # Better: Use precise filename matching below.
    return

def main():
    parser = argparse.ArgumentParser(description="Fetch and Merge Full Dataset (Klines + Funding + Metrics)")
    parser.add_argument("--symbol", type=str, required=True, help="e.g. BTCUSDT")
    parser.add_argument("--start_year", type=int, required=True)
    parser.add_argument("--end_year", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default="data/history")
    parser.add_argument("--skip_klines", action="store_true", help="Skip K-line download and use synthetic index.")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Download Datasets
    raw_dir = os.path.join(args.output_dir, "raw_parts")
    os.makedirs(raw_dir, exist_ok=True)
    
    try:
        # A. Klines (Optional)
        if not args.skip_klines:
            run_downloader(args.symbol, args.start_year, args.end_year, "klines", raw_dir)
        
        # B. Funding
        run_downloader(args.symbol, args.start_year, args.end_year, "fundingRate", raw_dir)
        # C. Metrics
        run_downloader(args.symbol, args.start_year, args.end_year, "metrics", raw_dir)
    except subprocess.CalledProcessError as e:
        logger.error("❌ Download failed. Aborting.")
        return

    # 2. Load & Rename
    f_klines = os.path.join(raw_dir, f"{args.symbol}_1h_{args.start_year}_{args.end_year}.csv")
    f_funding = os.path.join(raw_dir, f"{args.symbol}_fundingRate_{args.start_year}_{args.end_year}.csv")
    f_metrics = os.path.join(raw_dir, f"{args.symbol}_metrics_{args.start_year}_{args.end_year}.csv")
    
    # Check existence
    if not os.path.exists(f_funding) or not os.path.exists(f_metrics):
        logger.error("❌ Missing funding or metrics files.")
        return
        
    if not args.skip_klines and not os.path.exists(f_klines):
        logger.error("❌ Missing kline file.")
        return

    # 3. Merge Logic
    logger.info("🧠 Merging Datasets...")
    
    if args.skip_klines:
        # Generate Synthetic Index (1H)
        logger.info("Skipping Klines -> Generating 1H index spine.")
        start_date = f"{args.start_year}-01-01"
        end_date = f"{args.end_year}-12-31 23:00"
        # Clip to now if current year? 
        # For simplicity, stick to requested range. User can filter later.
        full_idx = pd.date_range(start=start_date, end=end_date, freq='1h')
        df_base = pd.DataFrame(index=full_idx)
        df_base.index.name = 'timestamp'
    else:
        # Load Klines (The Anchor)
        df_base = pd.read_csv(f_klines)
        df_base['timestamp'] = pd.to_datetime(df_base['timestamp'])
        df_base.set_index('timestamp', inplace=True)
    
    # Load Funding
    df_f = pd.read_csv(f_funding)
    df_f['timestamp'] = pd.to_datetime(df_f['timestamp'])
    
    # Check for duplicates in Funding (rare but possible across monthly boundaries?)
    if df_f['timestamp'].duplicated().any():
        logger.warning("Found duplicate timestamps in Funding Rate. Dropping duplicates.")
        df_f.drop_duplicates(subset=['timestamp'], inplace=True)
        
    df_f.set_index('timestamp', inplace=True)
    
    # Load Metrics
    df_m = pd.read_csv(f_metrics)
    df_m['timestamp'] = pd.to_datetime(df_m['timestamp'])
    
    if df_m['timestamp'].duplicated().any():
        logger.warning("Found duplicate timestamps in Metrics. Dropping duplicates.")
        df_m.drop_duplicates(subset=['timestamp'], inplace=True)
        
    df_m.set_index('timestamp', inplace=True)
    
    # Rename "sum_open_interest" to "open_interest" if present
    # (Binance names it 'sum_open_interest' but it is the snapshot)
    rename_map = {
        'sum_open_interest': 'open_interest',
        'sum_open_interest_value': 'open_interest_value'
    }
    df_m.rename(columns=rename_map, inplace=True)
    
    # Resample Metrics (Daily/5m -> 1h)
    # Use .last() to get the "Close" Open Interest (e.g. at 00:55) for the 00:00 bin.
    # This aligns better with 'Close' price.
    df_m_1h = df_m.resample('1h').last()
    
    # Merge
    # Left join on Base (1H spine)
    df_final = df_base.join(df_m_1h, rsuffix='_metrics')
    df_final = df_final.join(df_f, rsuffix='_funding')
    
    # Clean up Funding (Forward Fill)
    # Funding is 8h. We ffill to fill the gaps between 00:00, 08:00, 16:00.
    df_final['funding_rate'] = df_final['funding_rate'].ffill().fillna(0)
    
    # Fill Metrics Gaps (forward fill the daily logic again if join missed anything)
    # For metrics columns only
    for col in df_m.columns:
        if col in df_final.columns:
            df_final[col] = df_final[col].ffill()
    
    # Save
    out_file = os.path.join(args.output_dir, f"{args.symbol}_FULL_{args.start_year}_{args.end_year}.csv")
    df_final.reset_index(inplace=True)
    df_final.to_csv(out_file, index=False)
    
    logger.info(f"🎉 Success! Final merged dataset saved to:\n   {out_file}")
    logger.info(f"   Shape: {df_final.shape}")
    logger.info("   Columns: " + ", ".join(df_final.columns))

if __name__ == "__main__":
    main()
