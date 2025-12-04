import pandas as pd
import numpy as np
from analysis.market_regime_detection_new import detect_regime_change

# Create stable market data similar to the test with fixed seed for reproducibility
np.random.seed(42)  # Fix seed for consistent results
dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=100, freq="h"))
price = 100 + (np.random.randn(100) * 0.1).cumsum()
stable_market_data = pd.DataFrame({"close": price}, index=dates)

print("Testing with stable market data...")
print("Data shape:", stable_market_data.shape)
print("Price range:", stable_market_data['close'].min(), "to", stable_market_data['close'].max())
print("Sample data:")
print(stable_market_data.head())

result = detect_regime_change(stable_market_data)
print("\nResult:", result)
print("F-statistic:", result.get('f_value'))