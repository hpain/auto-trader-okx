
import sys
import os
import pandas as pd
import numpy as np
import logging
import warnings

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock config and imports
from config import config

def main():
    logging.basicConfig(level=logging.INFO)
    
    # Path to the cache file identified in logs
    feature_cache_path = r"data/cache/features_3016d4a4c9.parquet"
    
    if not os.path.exists(feature_cache_path):
        print(f"Cache file not found: {feature_cache_path}")
        return

    print("Loading parquet...")
    dfm = pd.read_parquet(feature_cache_path)
    print(f"Loaded dfm shape: {dfm.shape}")
    
    # Mock making supervised
    # Logic from improved_evolution.py lines 339+
    if not isinstance(dfm.index, pd.DatetimeIndex):
        dfm = dfm.set_index('timestamp')
        
    print(f"Index type: {type(dfm.index)}")
    
    # We can't easily import make_supervised without all dependencies, 
    # but let's try to simulate the dropping logic which is the suspect.
    # Assuming make_supervised just adds target 'y' and shifts.
    
    data = dfm.copy()
    data['y'] = 1 # Dummy target
    
    non_feature_cols = ["ts", "dt", "y", "future_high", "future_low", "future_close", "future_ret", "date", "timestamp", "vol_ccy", "vol_ccy_quote", "confirm"]
    feature_cols = [c for c in data.columns if c not in non_feature_cols]
    
    print(f"Initial feature_cols count: {len(feature_cols)}")
    
    # --- Feature Density Filter ---
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    nan_threshold = 0.2
    dense_feature_cols = []
    dropped_sparse_cols = []

    print("\nChecking NaN ratios:")
    for col in feature_cols:
        nan_ratio = data[col].isna().mean()
        if nan_ratio > 0.1: # Print if meaningful
            print(f"  {col}: {nan_ratio:.2%}")
            
        if nan_ratio > nan_threshold:
            dropped_sparse_cols.append(f"{col} ({nan_ratio:.1%})")
            # data.drop(columns=[col], inplace=True) # Don't actually drop for debug, just count
        else:
            dense_feature_cols.append(col)
            
    print(f"\nDropped sparse cols: {dropped_sparse_cols}")
    print(f"Remaining dense cols: {len(dense_feature_cols)}")

if __name__ == "__main__":
    main()
