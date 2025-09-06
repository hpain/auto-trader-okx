# ============== Test Setup: Mock config before any project imports =========
# This MUST be at the top of the file.
import os
import yaml
import builtins
from unittest.mock import mock_open

# Define a dummy config structure that tests can rely on.
_dummy_config = {
    "okx": {"api_key": "dummy", "secret_key": "dummy", "passphrase": "dummy", "flag": "0"},
    "trade": {"symbol": "BTC-USDT", "interval": "1H", "quantity": 0.001},
    "paths": {"model_dir": "models", "feature_cache_dir": "data/cache", "history_data_dir": "data/history"}
}

# Patch the functions that access the filesystem, BEFORE they are called by any module import
os.path.exists = lambda path: True
builtins.open = mock_open(read_data="dummy: yaml")
yaml.safe_load = lambda stream: _dummy_config
# ========================================================================

# Now, with the config loading patched, we can safely import project code
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

# Make sure the utils module can be found
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_normalization import normalize_binance_df

def test_normalize_binance_df_happy_path():
    # 1. Arrange: Create a sample DataFrame similar to Binance API output
    raw_data = {
        'o': ['100.0', '101.0'],
        'h': ['105.0', '106.0'],
        'l': ['99.0', '100.0'],
        'c': ['104.0', '105.0'],
        'v': ['1000.0', '1200.0'],
        'other_col': [1, 2] # An extra column that should be kept
    }
    raw_df = pd.DataFrame(raw_data)

    # Expected output DataFrame
    expected_data = {
        'open': [100.0, 101.0],
        'high': [105.0, 106.0],
        'low': [99.0, 100.0],
        'close': [104.0, 105.0],
        'vol': [1000.0, 1200.0],
        'other_col': [1, 2]
    }
    expected_df = pd.DataFrame(expected_data)
    # Ensure correct dtypes
    for col in ['open', 'high', 'low', 'close', 'vol']:
        expected_df[col] = expected_df[col].astype(float)


    # 2. Act: Call the function to be tested
    normalized_df = normalize_binance_df(raw_df)

    # 3. Assert: Check if the output is as expected
    assert_frame_equal(normalized_df, expected_df)

def test_normalize_binance_df_with_bad_data():
    # 1. Arrange: Create a sample DataFrame with non-numeric data
    raw_data = {
        'o': ['100.0', 'not_a_number'],
        'h': ['105.0', '106.0'],
        'l': ['99.0', '100.0'],
        'c': ['104.0', '105.0'],
        'v': ['1000.0', '1200.0'],
    }
    raw_df = pd.DataFrame(raw_data)

    # 2. Act: Call the function
    normalized_df = normalize_binance_df(raw_df)

    # 3. Assert: Check that the row with bad data was coerced to NaN
    assert pd.isna(normalized_df.loc[1, 'open'])
    assert normalized_df.loc[0, 'open'] == 100.0