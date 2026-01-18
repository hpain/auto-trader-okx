import pandas as pd
import numpy as np

csv_path = 'data/history/CLEAN_UNIVERSAL_2022_2026.csv'
print(f"Loading: {csv_path}")

df = pd.read_csv(csv_path)

print("\n" + "="*60)
print("CSV INSPECTION REPORT")
print("="*60)

print(f"\n[SHAPE] {df.shape[0]:,} rows x {df.shape[1]} columns")

print("\n[COLUMNS]")
for i, col in enumerate(df.columns):
    print(f"  {i+1:2d}. {col}")

print("\n[FIRST 3 ROWS]")
print(df.head(3).to_string())

print("\n[LAST 3 ROWS]")
print(df.tail(3).to_string())

print("\n[DATA TYPES]")
print(df.dtypes)

print("\n[NULL VALUES]")
null_counts = df.isnull().sum()
null_cols = null_counts[null_counts > 0]
if len(null_cols) > 0:
    print(null_cols)
else:
    print("  No null values found!")

# Check timestamp column
ts_cols = ['timestamp', 'open_time', 'ts', 'date', 'time']
ts_col = next((c for c in ts_cols if c in df.columns), None)
if ts_col:
    df[ts_col] = pd.to_datetime(df[ts_col])
    print(f"\n[DATE RANGE] ({ts_col})")
    print(f"  Start: {df[ts_col].min()}")
    print(f"  End:   {df[ts_col].max()}")
    
    # Check for gaps
    if len(df) > 1:
        time_diffs = df[ts_col].diff().dropna()
        most_common_gap = time_diffs.mode()[0]
        gaps = time_diffs[time_diffs != most_common_gap]
        print(f"  Expected Interval: {most_common_gap}")
        if len(gaps) > 0:
            print(f"  WARNING: Found {len(gaps)} irregular gaps!")
        else:
            print("  OK: No irregular gaps detected")

# Check for symbol column
if 'symbol' in df.columns:
    print(f"\n[SYMBOLS]")
    print(df['symbol'].value_counts())

# Check for required columns
required_cols = ['open', 'high', 'low', 'close', 'volume']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    print(f"\n[ERROR] MISSING REQUIRED COLUMNS: {missing}")
else:
    print(f"\n[OK] All OHLCV columns present")
    
# Check for derivatives data
deriv_cols = ['funding_rate', 'open_interest', 'long_short_ratio']
present_deriv = [c for c in deriv_cols if c in df.columns]
if present_deriv:
    print(f"\n[DERIVATIVES COLUMNS] {present_deriv}")
    for col in present_deriv:
        non_null = df[col].notna().sum()
        print(f"  {col}: {non_null:,}/{len(df):,} ({100*non_null/len(df):.1f}% coverage)")
