import pandas as pd
import numpy as np
from analysis.market_regime_detection_new import detect_regime_change

# Create stable market data similar to the test
np.random.seed(42)  # Fixed seed to reproduce the failing case
dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=100, freq="h"))
price = 100 + (np.random.randn(100) * 0.1).cumsum()
stable_market_data = pd.DataFrame({"close": price}, index=dates)

print("Testing with stable market data...")
result = detect_regime_change(stable_market_data)
print("Result:", result)

# Create volatile market data similar to the test
np.random.seed(123)  # Different seed to check the other case
dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=100, freq="h"))
stable_part = 100 + np.random.randn(50).cumsum() * 0.1
volatile_part = stable_part[-1] + np.random.randn(50).cumsum() * 2.0  #波动放大20倍
price = np.concatenate([stable_part, volatile_part])
volatile_market_data = pd.DataFrame({"close": price}, index=dates)

print("\nTesting with volatile market data...")
result2 = detect_regime_change(volatile_market_data)
print("Result:", result2)