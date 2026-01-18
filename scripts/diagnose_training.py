"""
Quick diagnostic script to check training data and feature generation.
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("="*60)
print("TRAINING DATA DIAGNOSTIC")
print("="*60)

# 1. Check CSV data
csv_path = 'data/history/CLEAN_UNIVERSAL_2022_2026.csv'
print(f"\n[1] Loading CSV: {csv_path}")
df = pd.read_csv(csv_path)

print(f"    Total rows: {len(df):,}")
print(f"    Columns: {len(df.columns)}")
print(f"    Symbol values: {df['symbol'].unique().tolist()}")
print(f"    Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# Check symbol distribution
symbol_counts = df['symbol'].value_counts()
print(f"\n    Symbol distribution:")
for sym, count in symbol_counts.items():
    print(f"      {sym}: {count:,} rows ({count/len(df)*100:.1f}%)")

# 2. Check for time continuity within each symbol
print(f"\n[2] Time continuity check:")
for symbol in df['symbol'].unique():
    sym_df = df[df['symbol'] == symbol].copy()
    sym_df['timestamp'] = pd.to_datetime(sym_df['timestamp'])
    sym_df = sym_df.sort_values('timestamp')
    
    time_diffs = sym_df['timestamp'].diff().dropna()
    median_diff = time_diffs.median()
    
    # Check for gaps
    expected_gap = pd.Timedelta(hours=1)
    large_gaps = time_diffs[time_diffs > expected_gap * 2]
    
    print(f"    {symbol}: median gap = {median_diff}, large gaps = {len(large_gaps)}")

# 3. Test feature generation
print(f"\n[3] Feature generation test:")
try:
    from features.feature_engineering import generate_features
    
    # Get a single symbol's data
    test_df = df[df['symbol'] == 'BTCUSDT'].head(500).copy()
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
    test_df = test_df.set_index('timestamp')
    
    # Generate features
    features_df = generate_features(test_df)
    
    print(f"    Input rows: {len(test_df)}")
    print(f"    Output rows: {len(features_df)}")
    print(f"    Features generated: {len(features_df.columns)}")
    
    # Check for NaN
    nan_cols = features_df.columns[features_df.isna().any()].tolist()
    print(f"    Columns with NaN: {len(nan_cols)}")
    
    # Check feature summary
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    print(f"    Numeric features: {len(numeric_cols)}")
    
except Exception as e:
    print(f"    ERROR: {e}")
    import traceback
    traceback.print_exc()

# 4. Check for label distribution (if y column exists)
print(f"\n[4] Checking label generation:")
try:
    from features.feature_engineering import apply_triple_barrier
    
    test_df = df[df['symbol'] == 'BTCUSDT'].tail(1000).copy()
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
    test_df = test_df.set_index('timestamp')
    
    labeled = apply_triple_barrier(test_df, tp=0.008, sl=0.005, timeout=12)
    
    if 'y' in labeled.columns:
        label_dist = labeled['y'].value_counts()
        print(f"    Label distribution:")
        for label, count in sorted(label_dist.items()):
            print(f"      {label}: {count} ({count/len(labeled)*100:.1f}%)")
        
        # Check class imbalance
        if len(label_dist) >= 2:
            ratio = label_dist.max() / label_dist.min()
            print(f"    Class imbalance ratio: {ratio:.2f}")
    else:
        print("    WARNING: No 'y' column generated!")
        
except Exception as e:
    print(f"    ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)
