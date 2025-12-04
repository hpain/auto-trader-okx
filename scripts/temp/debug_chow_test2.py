import pandas as pd
import numpy as np
from scipy.stats import f
import statsmodels.api as sm

# Create stable market data similar to the test with fixed seed for reproducibility
np.random.seed(42)  # Fix seed for consistent results
dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=100, freq="h"))
price = 100 + (np.random.randn(100) * 0.1).cumsum()
stable_market_data = pd.DataFrame({"close": price}, index=dates)

# Replicate the volatility calculation from the detection function
df_copy = stable_market_data.copy()
df_copy['returns'] = df_copy['close'].pct_change()
df_copy['volatility'] = df_copy['returns'].rolling(window=20).std() * 100
volatility_series = df_copy['volatility'].dropna()

print("Volatility series length:", len(volatility_series))
print("First 10 volatility values:", volatility_series.head(10).values)
print("Last 10 volatility values:", volatility_series.tail(10).values)

# Volatility in first half vs second half
split_point = int(len(volatility_series) * 0.5)
first_half = volatility_series.iloc[:split_point]
second_half = volatility_series.iloc[split_point:]

print("\nFirst half statistics:")
print(f"  Mean: {first_half.mean():.6f}")
print(f"  Std: {first_half.std():.6f}")
print(f"  Length: {len(first_half)}")

print("\nSecond half statistics:")
print(f"  Mean: {second_half.mean():.6f}")
print(f"  Std: {second_half.std():.6f}")
print(f"  Length: {len(second_half)}")

print(f"\nDifference in means: {abs(second_half.mean() - first_half.mean()):.6f}")


def _chow_test_manual(y: pd.Series, split_point: int) -> tuple:
    """
    Using statsmodels.api.OLS to perform Chow structural break test for more stable and accurate SSR.
    This version only tests for a change in the mean of the series.
    """
    # Add constant only for the intercept, to test for a mean change
    X = sm.add_constant(np.ones(len(y)))
    X1 = X[:split_point]
    X2 = X[split_point:]
    y1 = y.iloc[:split_point]
    y2 = y.iloc[split_point:]

    # 1. OLS regression on the entire dataset, get SSR_pooled
    model_pooled = sm.OLS(y, X).fit()
    ssr_pooled = model_pooled.ssr

    # 2. Regression on subset 1, get SSR1
    model1 = sm.OLS(y1, X1).fit()
    ssr1 = model1.ssr

    # 3. Regression on subset 2, get SSR2
    model2 = sm.OLS(y2, X2).fit()
    ssr2 = model2.ssr

    # 4. Calculate F-statistic for Chow test
    k = X.shape[1]  # k is the number of explanatory variables (here it's 1, the intercept)
    N = len(y)
    
    print(f"\nChow test calculation:")
    print(f"  Total SSR (pooled): {ssr_pooled:.6f}")
    print(f"  SSR1 (first half): {ssr1:.6f}")
    print(f"  SSR2 (second half): {ssr2:.6f}")
    print(f"  N (total observations): {N}")
    print(f"  k (number of params): {k}")

    numerator = (ssr_pooled - (ssr1 + ssr2)) / k
    denominator = (ssr1 + ssr2) / (N - 2 * k)
    
    f_stat = numerator / denominator if denominator > 1e-9 else 0.0
    p_value = f.sf(f_stat, k, N - 2 * k)

    print(f"\n  Numerator: (ssr_pooled - (ssr1 + ssr2)) / k = ({ssr_pooled:.6f} - ({ssr1:.6f} + {ssr2:.6f})) / {k} = {numerator:.6f}")
    print(f"  Denominator: (ssr1 + ssr2) / (N - 2*k) = ({ssr1:.6f} + {ssr2:.6f}) / ({N} - {2*k}) = {denominator:.6f}")
    print(f"  F-statistic: {f_stat:.6f}")
    print(f"  P-value: {p_value:.6f}")
    
    return f_stat, p_value

# Test the Chow test function
f_value, p_value = _chow_test_manual(y=volatility_series, split_point=split_point)
print(f"\nManual Chow Test Result - F-statistic: {f_value:.6f}, P-value: {p_value:.6f}")