import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.feature_engineering import apply_triple_barrier

# Load the recently generated cache file
cache_path = r"data/cache/features_69fc32ed5c.parquet"
if not os.path.exists(cache_path):
    # Try finding any cache file
    import glob
    files = glob.glob(r"data/cache/*.parquet")
    if files:
        cache_path = files[0]
        print(f"Using cache: {cache_path}")
    else:
        print("No cache found.")
        sys.exit(1)

df = pd.read_parquet(cache_path)
print(f"Processing data with {len(df)} rows...")

# Test different barriers
barriers = [
    (0.015, 0.01, 12),
    (0.01, 0.01, 12),
    (0.01, 0.005, 8),
    (0.005, 0.005, 12)
]

for tp, sl, timeout in barriers:
    result = apply_triple_barrier(df, tp=tp, sl=sl, timeout=timeout)
    dist = result['y'].value_counts(normalize=True).to_dict()
    pos_count = int(result['y'].sum())
    print(f"TP={tp:.1%}, SL={sl:.1%}, Timeout={timeout}H -> Positive Label: {dist.get(1.0, 0):.2%} ({pos_count} samples)")
