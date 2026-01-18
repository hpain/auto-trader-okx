import pandas as pd
import sys

csv_path = "data/history/CLEAN_UNIVERSAL_2022_2026.csv"
print(f"Inspecting {csv_path}...")

try:
    df = pd.read_csv(csv_path)
    print(f"Successfully loaded. Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())
    
    print("\nUnique Symbols and Counts:")
    print(df['symbol'].value_counts())
    
    print("\nFirst 10 rows (timestamp, symbol, close):")
    cols = ['timestamp', 'symbol', 'close'] if 'timestamp' in df.columns else ['symbol', 'close']
    print(df[cols].head(10))
    
    print("\nData Types:")
    print(df.dtypes)
    
    # Check for interleaving
    print("\nChecking interleaving pattern (first 20 rows symbol sequence):")
    print(df['symbol'].head(20).tolist())

    if 'symbol' not in df.columns:
        print("\nCRITICAL: 'symbol' column missing!")
    else:
        print("\n'symbol' column present.")

except Exception as e:
    print(f"\nCRITICAL ERROR reading CSV: {e}")
