import pandas as pd
import os

cache_path = r"data/cache/features_69fc32ed5c.parquet"
if os.path.exists(cache_path):
    try:
        df = pd.read_parquet(cache_path)
        print(f"Shape: {df.shape}")
        print(f"Columns ({len(df.columns)}):")
        print(df.columns.tolist()[:20])
        
        # Check specific mined features
        mined_cols = [c for c in df.columns if "enhanced_gp" in c]
        print(f"Mined Features: {mined_cols}")
        
    except Exception as e:
        print(f"Error reading parquet: {e}")
else:
    print(f"File not found: {cache_path}")
