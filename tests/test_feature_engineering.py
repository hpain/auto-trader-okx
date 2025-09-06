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

from features.feature_engineering import generate_features

def test_generate_features_basic():
    # 1. Arrange: Create a sample DataFrame with enough data for feature calculation
    # We need at least 50 rows for sma_50/ema_50
    data = {
        'open': range(100, 150),
        'high': range(102, 152),
        'low': range(99, 149),
        'close': range(101, 151),
        'vol': range(1000, 1050)
    }
    df = pd.DataFrame(data)

    # 2. Act: Generate features
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