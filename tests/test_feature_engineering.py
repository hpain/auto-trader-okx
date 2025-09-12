# tests/test_feature_engineering.py

# ============== Test Setup: Mock config before any project imports =========
# This MUST be at the top of the file.
import os
import yaml
import builtins
from unittest.mock import mock_open
import numpy as np

# Define a dummy config structure that tests can rely on.
_dummy_config = {
    "okx": {"api_key": "dummy", "secret_key": "dummy", "passphrase": "dummy", "flag": "0"},
    "trade": {"symbol": "BTC-USDT", "interval": "1H", "quantity": 0.001},
    "paths": {"model_dir": "models", "feature_cache_dir": "data/cache", "history_data_dir": "data/history"}
}

# Patch the functions that access the filesystem
os.path.exists = lambda path: True
builtins.open = mock_open(read_data="dummy: yaml")
yaml.safe_load = lambda stream: _dummy_config
# ========================================================================

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

# Make sure the features module can be found
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.feature_engineering import generate_features, merge_price_and_sentiment
from data.news import load_news_from_csv, aggregate_daily_sentiment

# Helper function to create a base DataFrame for tests
def create_base_df(rows=50):
    data = {
        'open': range(100, 100 + rows),
        'high': range(102, 102 + rows),
        'low': range(99, 99 + rows),
        'close': range(101, 101 + rows),
        'vol': range(1000, 1000 + rows)
    }
    # Create a DatetimeIndex
    index = pd.to_datetime(pd.date_range(start='2023-01-01', periods=rows, freq='H'))
    return pd.DataFrame(data, index=index)

def test_generate_features_basic():
    # 1. Arrange: Create a sample DataFrame
    df = create_base_df(rows=50)

    # 2. Act: Generate features without news data
    df_features = generate_features(df.copy())

    # 3. Assert: Check for expected columns and basic properties
    expected_cols = [
        # Original
        'open', 'high', 'low', 'close', 'vol',
        # Volume
        'vol_ma20', 'vol_ratio', 'obv',
        # Volatility
        'atr_14', 'bb_mavg', 'bb_hband', 'bb_lband',
        # Trend
        'sma_5', 'ema_5', 'sma_10', 'ema_10', 'sma_20', 'ema_20', 'sma_50', 'ema_50',
        'macd', 'macd_signal', 'adx', 'adx_pos', 'adx_neg',
        # Momentum
        'rsi_14', 'roc_10', 'williams_r', 'stoch_k', 'stoch_d',
        # Other
        'return', 'cum_return'
    ]
    
    # Check if all expected columns are present
    for col in expected_cols:
        assert col in df_features.columns, f"Missing expected column: {col}"

    # Check for NaN values in critical columns (after initial rows due to windowing)
    assert np.isnan(df_features['sma_5'].iloc[3])
    assert df_features['sma_5'].iloc[4:].notnull().all()
    
    assert np.isnan(df_features['rsi_14'].iloc[12])
    assert df_features['rsi_14'].iloc[13:].notnull().all()

    # Check value ranges for some indicators
    assert (df_features['rsi_14'].dropna() >= 0).all() and (df_features['rsi_14'].dropna() <= 100).all()
    assert (df_features['williams_r'].dropna() >= -100).all() and (df_features['williams_r'].dropna() <= 0).all()

    # Check that original columns are unchanged
    original_cols = ['open', 'high', 'low', 'close', 'vol']
    assert_frame_equal(df_features[original_cols], df[original_cols])

    # Check that sentiment columns are created with default zero values
    assert 'sent_mean' in df_features.columns
    assert (df_features['sent_mean'] == 0.0).all()
    assert (df_features['count'] == 0).all()

def test_generate_features_with_news_sentiment(monkeypatch):
    """
    Test that sentiment features are correctly loaded, merged, and handled.
    """
    # 1. Arrange
    price_df = create_base_df(rows=72) # 3 days of hourly data
    
    # Create a dummy news CSV content
    news_content = """timestamp,title
2023-01-01T10:00:00Z,"Good news for crypto, prices are up"
2023-01-02T12:00:00Z,"Bad news for crypto, prices are down"
"""
    dummy_news_path = "/dummy/path/to/news.csv"

    # Mock the file reading for the news CSV
    m = mock_open(read_data=news_content)
    monkeypatch.setattr("builtins.open", m)
    
    # 2. Act
    df_features = generate_features(price_df.copy(), news_csv_path=dummy_news_path)

    # 3. Assert
    # Check that the open mock was called with our dummy path
    m.assert_called_with(dummy_news_path, encoding='utf-8-sig')
    
    # Check that sentiment columns exist
    assert 'sent_mean' in df_features.columns
    assert 'count' in df_features.columns

    # Check that sentiment values were correctly applied and forward-filled
    # Day 1 (2023-01-01) should have a positive sentiment
    assert df_features.loc['2023-01-01']['sent_mean'].iloc[0] > 0
    # Day 2 (2023-01-02) should have a negative sentiment
    assert df_features.loc['2023-01-02']['sent_mean'].iloc[0] < 0
    # Day 3 (2023-01-03) should have the same sentiment as Day 2 due to ffill
    assert df_features.loc['2023-01-03']['sent_mean'].iloc[0] == df_features.loc['2023-01-02']['sent_mean'].iloc[0]
    
    # Check counts
    assert df_features.loc['2023-01-01']['count'].iloc[0] == 1
    assert df_features.loc['2023-01-02']['count'].iloc[0] == 1

def test_generate_features_news_file_not_found(capsys):
    """
    Test that a warning is printed and dummy columns are created if news file is not found.
    """
    # 1. Arrange
    price_df = create_base_df()
    non_existent_path = "/dummy/path/that/does/not/exist.csv"

    # 2. Act
    df_features = generate_features(price_df.copy(), news_csv_path=non_existent_path)
    
    # 3. Assert
    # Check that dummy columns were created
    assert 'sent_mean' in df_features.columns
    assert (df_features['sent_mean'] == 0.0).all()
    
    # Check that a warning was printed to stderr/stdout
    captured = capsys.readouterr()
    assert "Warning: News file not found" in captured.out

def test_generate_features_empty_input():
    # Test with an empty DataFrame
    df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'vol'])
    df_features = generate_features(df)
    assert df_features.empty

def test_generate_features_insufficient_data():
    # Test with insufficient data for some indicators
    data = {
        'open': [100, 101, 102, 103, 104],
        'high': [102, 103, 104, 105, 106],
        'low': [99, 100, 101, 102, 103],
        'close': [101, 102, 103, 104, 105],
        'vol': [1000, 1100, 1200, 1300, 1400]
    }
    df = pd.DataFrame(data)
    df_features = generate_features(df.copy())
    
    # SMA_5 should be calculable on the last row, but SMA_10 should not
    assert not np.isnan(df_features['sma_5'].iloc[4])
    assert df_features['sma_10'].isnull().all()
    assert df_features['rsi_14'].isnull().all() # RSI needs 14 periods

def test_merge_price_and_sentiment():
    """
    Unit test for the merge_price_and_sentiment function.
    """
    # Arrange
    price_df = create_base_df(rows=72) # 3 days
    
    # Create a dummy daily sentiment DataFrame
    sent_dates = pd.to_datetime(['2023-01-01', '2023-01-03'])
    sent_data = {'sent_mean': [0.5, -0.5], 'sent_median': [0.5, -0.5], 'count': [10, 5]}
    daily_sent_df = pd.DataFrame(sent_data, index=sent_dates)

    # Act
    merged_df = merge_price_and_sentiment(price_df, daily_sent_df)

    # Assert
    # Check that the first day has the correct sentiment
    assert merged_df.loc['2023-01-01', 'sent_mean'].iloc[0] == 0.5
    # Check that the second day is forward-filled from the first
    assert merged_df.loc['2023-01-02', 'sent_mean'].iloc[0] == 0.5
    # Check that the third day has its own new sentiment
    assert merged_df.loc['2023-01-03', 'sent_mean'].iloc[0] == -0.5
    # Check that the final row has the correct sentiment
    assert merged_df['sent_mean'].iloc[-1] == -0.5
    # Check that there are no NaNs in the sentiment columns
    assert not merged_df[['sent_mean', 'sent_median', 'count']].isnull().values.any()