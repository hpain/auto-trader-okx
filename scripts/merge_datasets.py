import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def merge_universal(output_file="data/history/UNIVERSAL_FULL_2020_2025.csv"):
    files = [
        "data/history/BTCUSDT_FULL_2020_2025.csv",
        "data/history/ETHUSDT_FULL_2020_2025.csv"
    ]
    
    dfs = []
    
    for f in files:
        if not os.path.exists(f):
            logging.error(f"File not found: {f}")
            continue
            
        logging.info(f"Loading {f}...")
        df = pd.read_csv(f)
        
        # Ensure timestamp is datetime for consistent sorting (though concat is mainly appending)
        if 'timestamp' in df.columns:
             df['timestamp'] = pd.to_datetime(df['timestamp'])
             
        # Temporal Shift Trick for Universal Model:
        # Shift ETH by +20 years so it sits "after" BTC in the timeline.
        # This prevents duplicate index errors in TimeSeriesSplit while preserving local temporal structure.
        if "ETH" in f:
            logging.info("Shifting ETH timestamps by +20 years (Panel Data Construction)...")
            df['timestamp'] = df['timestamp'] + pd.DateOffset(years=20)
        
        dfs.append(df)
        logging.info(f"Loaded {len(df)} rows.")

    if not dfs:
        logging.error("No data loaded.")
        return

    logging.info("Concatenating datasets...")
    universal_df = pd.concat(dfs, ignore_index=True)
    
    # Sort by timestamp to ensure BTC comes first, then ETH (in the future)
    # This is CRITICAL for TimeSeriesSplit
    universal_df.sort_values('timestamp', inplace=True)
    
    logging.info(f"Saving Universal Dataset ({len(universal_df)} rows) to {output_file}...")
    universal_df.to_csv(output_file, index=False)

    logging.info("Done.")

if __name__ == "__main__":
    merge_universal()
