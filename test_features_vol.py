
import pandas as pd
import numpy as np
from features.feature_engineering import generate_features

def test_vol_alias():
    # Create dummy data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    df = pd.DataFrame({
        'open': np.random.rand(100) * 100,
        'high': np.random.rand(100) * 100,
        'low': np.random.rand(100) * 100,
        'close': np.random.rand(100) * 100,
        'volume': np.random.rand(100) * 1000
    }, index=dates)

    print("Original columns:", df.columns)

    # Generate features
    try:
        featured_df = generate_features(df)
        print("Featured columns:", featured_df.columns)
        
        if 'vol' in featured_df.columns:
            print("SUCCESS: 'vol' column exists.")
        else:
            print("FAILURE: 'vol' column MISSING.")
            
        if 'volume' in featured_df.columns:
            print("SUCCESS: 'volume' column exists.")
            
    except Exception as e:
        print(f"Error during feature generation: {e}")

if __name__ == "__main__":
    test_vol_alias()
