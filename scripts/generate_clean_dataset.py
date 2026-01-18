import pandas as pd
import os
import sys

# Define constants
DATA_DIR = r"e:\pycode\auto-trader-okx\data\history"
BTC_SOURCE = os.path.join(DATA_DIR, "BTCUSDT_FULL_2020_2025.csv")
ETH_SOURCE = os.path.join(DATA_DIR, "ETHUSDT_FULL_2020_2025.csv")
START_DATE = "2021-12-01"

def process_file(file_path, symbol_label):
    print(f"\n--- Processing {symbol_label} from {file_path} ---")
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return None

    df = pd.read_csv(file_path)
    print(f"Original shape: {df.shape}")
    
    # Ensure timestamp is datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        print("Error: No 'timestamp' column found.")
        return None

    # Filter
    df_clean = df[df['timestamp'] >= START_DATE].copy()
    print(f"Filtered (>= {START_DATE}) shape: {df_clean.shape}")
    
    if df_clean.empty:
        print("Error: Filter resulted in empty dataframe!")
        return None

    # Audit for NaNs in critical columns
    critical_cols = ['open', 'close', 'open_interest', 'funding_rate']
    missing_stats = {}
    
    print("Missing Value Audit:")
    for col in critical_cols:
        if col in df_clean.columns:
            missing = df_clean[col].isna().sum()
            missing_pct = (missing / len(df_clean)) * 100
            missing_stats[col] = missing
            print(f"  - {col}: {missing} missing ({missing_pct:.2f}%)")
        else:
            print(f"  - {col}: COLUMN MISSING!")
            missing_stats[col] = -1

    # Add symbol column for universal dataset (0 for BTC, 1 for ETH)
    # Mapping: BTC=0, ETH=1 is common convention
    symbol_id = 0.0 if symbol_label == "BTC/USDT" else 1.0
    df_clean['symbol'] = symbol_id
    
    return df_clean

def main():
    print(f"Starting Dataset Purification. Target Start Date: {START_DATE}")
    
    # 1. Process BTC
    btc_clean = process_file(BTC_SOURCE, "BTC/USDT")
    
    # 2. Process ETH
    eth_clean = process_file(ETH_SOURCE, "ETH/USDT")
    
    if btc_clean is not None and eth_clean is not None:
        # 3. Save Individual Clean Files
        btc_out = os.path.join(DATA_DIR, "CLEAN_BTC_2022_2025.csv")
        eth_out = os.path.join(DATA_DIR, "CLEAN_ETH_2022_2025.csv")
        
        btc_clean.to_csv(btc_out, index=False)
        eth_clean.to_csv(eth_out, index=False)
        print(f"\nSaved individual clean files:\n  - {btc_out}\n  - {eth_out}")
        
        # 4. Create Universal Dataset
        # Interleave or Concat? 
        # For simple training data, concat is fine, but sorting by timestamp helps
        # time-series splits to work correctly across symbols (though standard TimeSeriesSplit
        # might bleed if symbols are just stacked. Group-aware split is better).
        # We will sort by timestamp to emulate a single time stream of multiple assets.
        
        universal_df = pd.concat([btc_clean, eth_clean], axis=0)
        universal_df.sort_values(by='timestamp', inplace=True)
        
        univ_out = os.path.join(DATA_DIR, "CLEAN_UNIVERSAL_2022_2025.csv")
        universal_df.to_csv(univ_out, index=False)
        print(f"\nSaved Universal Clean Dataset:\n  - {univ_out}")
        print(f"  - Total Shape: {universal_df.shape}")
        
        # Final Verification
        nans = universal_df[['open_interest', 'funding_rate']].isna().sum().sum()
        if nans == 0:
            print("\n[SUCCESS] Datasets generated with ZERO missing values in critical columns! 🧹💎")
        else:
            print(f"\n[WARNING] Dataset still contains {nans} NaNs in critical columns. Check audit above.")
            
    else:
        print("\n[FAILURE] One or more files failed to process.")

if __name__ == "__main__":
    main()
