"""
Fix the symbol column in the CSV file.
Convert 0.0 -> BTCUSDT, 1.0 -> ETHUSDT, and ensure proper sorting.
"""
import pandas as pd

input_path = 'data/history/CLEAN_UNIVERSAL_2022_2026.csv'
output_path = 'data/history/CLEAN_UNIVERSAL_2022_2026_FIXED.csv'

print(f"Loading: {input_path}")
df = pd.read_csv(input_path)

# 1. Convert symbol from float to string
symbol_map = {0.0: 'BTCUSDT', 1.0: 'ETHUSDT'}
df['symbol'] = df['symbol'].map(symbol_map)

print(f"Symbol mapping applied: {df['symbol'].value_counts().to_dict()}")

# 2. Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 3. Sort by (symbol, timestamp) to ensure each asset's time series is contiguous
df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)

# 4. Save
df.to_csv(output_path, index=False)
print(f"Saved to: {output_path}")

# Verification
print("\n=== VERIFICATION ===")
print(f"Shape: {df.shape}")
print(f"Symbols: {df['symbol'].unique()}")
print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# Check contiguity per symbol
for sym in df['symbol'].unique():
    sym_df = df[df['symbol'] == sym]
    gaps = sym_df['timestamp'].diff().dropna()
    expected_gap = pd.Timedelta(hours=1)
    irregular = gaps[gaps != expected_gap]
    if len(irregular) > 0:
        print(f"  {sym}: {len(irregular)} irregular gaps")
    else:
        print(f"  {sym}: OK - no gaps")
