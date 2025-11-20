import pandas as pd
import numpy as np
from analysis.market_regime_detection_new import detect_regime_change

# Test the exact same data generation as in the test
dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=100, freq="h"))
# The random seed is different each time pytest runs, so this creates different data each time
price = 100 + (np.random.randn(100) * 0.1).cumsum()
stable_market_data = pd.DataFrame({"close": price}, index=dates)

print("Testing with generated stable market data...")
result = detect_regime_change(stable_market_data)

print(f"F-value: {result['f_value']:.2f}")
print(f"Effect size: {result.get('effect_size', 0):.2f}")
print(f"Regime changed: {result['regime_changed']}")
print(f"Expected: False")

if result['regime_changed'] is False:
    print("PASS Test would PASS - regime changed is False as expected")
else:
    print("FAIL Test would FAIL - regime changed is True but should be False")