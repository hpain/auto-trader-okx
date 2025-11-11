# tests/test_feature_engineering.py

# ============== Test Setup: Mock config before any project imports =========
# This MUST be at the top of the file.

# --- BEGIN VADER SENTIMENT MONKEY-PATCH ---
# This patch fixes a compatibility issue between older versions of
# vaderSentiment and newer Python versions (3.10+). The issue causes a
# TypeError during the lexicon file reading. We replace the faulty
# __init__ method with a corrected one.
try:
    import os
    import sys
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    def patched_vader_init(self, lexicon_file="vader_lexicon.txt", emoji_lexicon="emoji_utf8_lexicon.txt"):
        # Correctly locate the lexicon files relative to the vaderSentiment package
        vader_module = sys.modules[SentimentIntensityAnalyzer.__module__]
        vader_path = os.path.dirname(os.path.abspath(vader_module.__file__))
        
        # Load Lexicon
        lexicon_full_filepath = os.path.join(vader_path, lexicon_file)
        with open(lexicon_full_filepath, encoding='utf-8') as f:
            self.lexicon = {}
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    self.lexicon[parts[0]] = float(parts[1])

        # Load Emoji Lexicon
        emoji_full_filepath = os.path.join(vader_path, emoji_lexicon)
        with open(emoji_full_filepath, encoding='utf-8') as f:
            self.emoji_lexicon = {}
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    self.emoji_lexicon[parts[0]] = parts[1]
        # --- 核心修复：添加缺失的 emojis 属性初始化 ---
        self.emojis = self.emoji_lexicon

    SentimentIntensityAnalyzer.__init__ = patched_vader_init
except Exception as e:
    print(f"Could not apply VADER monkey-patch: {e}")
# --- END VADER SENTIMENT MONKEY-PATCH ---

import pandas as pd
import pytest
from unittest.mock import mock_open, patch

# Make sure the features module can be found
import sys, numpy as np, json
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.feature_engineering import generate_features, merge_price_and_sentiment
from pandas.testing import assert_frame_equal
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
    # --- 核心修复：确保测试数据使用带时区的 DatetimeIndex ---
    index = pd.date_range(start='2023-01-01', periods=rows, freq='h', tz='UTC')
    return pd.DataFrame(data, index=index)

def test_generate_features_basic():
    # 1. Arrange: Create a sample DataFrame
    df = create_base_df(rows=250) # Use enough rows for all indicators

    # 2. Act: Generate features without news data
    df_features = generate_features(df.copy())

    # 3. Assert: Check for expected columns and basic properties
    expected_cols = [
        'open', 'high', 'low', 'close', 'vol',
        'return', 'return_lag_5', 'vol_lag_5',
        'atr_14', 'bb_width_14', 'volatility_10',
        'sma_10', 'ema_50', 'adx_14', 'sma_spread_10_50',
        'rsi_14', 'roc_14', 'williams_r_14',
        'prob_bull',
    ]
    
    for col in expected_cols:
        assert col in df_features.columns, f"Missing expected column: {col}"

    assert np.isnan(df_features['sma_10'].iloc[8])
    assert df_features['sma_10'].iloc[9:].notnull().all()
    
    # For a 14-period RSI on monotonically increasing data, the 'ta' library
    # calculates the first value (100.0) at index 13, not 14.
    assert not np.isnan(df_features['rsi_14'].iloc[13])
    assert df_features['rsi_14'].iloc[13:].notnull().all()

    assert (df_features['rsi_14'].dropna() >= 0).all() and (df_features['rsi_14'].dropna() <= 100).all()
    assert (df_features['williams_r_14'].dropna() >= -100).all() and (df_features['williams_r_14'].dropna() <= 0).all()

    original_cols = ['open', 'high', 'low', 'close', 'vol']
    assert_frame_equal(df_features[original_cols], df[original_cols])

    assert 'sent_mean' in df_features.columns
    assert (df_features['sent_mean'] == 0.0).all()
    assert (df_features['count'] == 0).all()

def test_generate_features_with_news_sentiment():
    """
    Test that sentiment features are correctly loaded, merged, and handled.
    """
    # 1. Arrange
    price_df = create_base_df(rows=72) # 3 days of hourly data
    dummy_news_path = "/dummy/path/to/news.csv"

    # Create a dummy DataFrame that would be returned by pd.read_csv
    mock_news_data = pd.DataFrame({
        'timestamp': pd.to_datetime(['2023-01-01T10:00:00Z', '2023-01-02T12:00:00Z']),
        'title': ["Good news for crypto, prices are up", "Bad news for crypto, prices are down"]
    })

    # 2. Act: Patch pandas.read_csv directly to avoid interfering with other file operations
    with patch('pandas.read_csv', return_value=mock_news_data) as mock_read_csv:
        df_features = generate_features(price_df.copy(), news_csv_path=dummy_news_path)

    # 3. Assert
    mock_read_csv.assert_called_once_with(dummy_news_path)
    
    assert 'sent_mean' in df_features.columns
    assert 'count' in df_features.columns

    assert df_features.loc['2023-01-01']['sent_mean'].iloc[0] > 0
    assert df_features.loc['2023-01-02']['sent_mean'].iloc[0] < 0
    assert df_features.loc['2023-01-03']['sent_mean'].iloc[0] == df_features.loc['2023-01-02']['sent_mean'].iloc[0]
    
    assert df_features.loc['2023-01-01', 'count'].iloc[0] == 1
    assert df_features.loc['2023-01-02', 'count'].iloc[0] == 1

@patch('builtins.open', new_callable=mock_open)
def test_generate_features_news_file_not_found(mock_file_open, capsys):
    """
    Test that a warning is printed and dummy columns are created if news file is not found.
    """
    price_df = create_base_df()
    non_existent_path = "/dummy/path/that/does/not/exist.csv"
    mock_file_open.side_effect = FileNotFoundError

    df_features = generate_features(price_df.copy(), news_csv_path=non_existent_path)
    
    assert 'sent_mean' in df_features.columns
    assert (df_features['sent_mean'] == 0.0).all()
    
    captured = capsys.readouterr()
    assert "Warning: News file not found" in captured.out

def test_generate_features_empty_input():
    df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'vol'], index=pd.to_datetime([]))
    df_features = generate_features(df)
    assert df_features.empty

def test_generate_features_insufficient_data():
    df = create_base_df(rows=5)
    df_features = generate_features(df.copy())
    
    # A 5-row dataframe cannot calculate a 10-period SMA.
    assert df_features['sma_10'].isnull().all()
    assert df_features['rsi_14'].isnull().all() # RSI needs 14 periods

def test_merge_price_and_sentiment():
    """
    Unit test for the merge_price_and_sentiment function.
    """
    price_df = create_base_df(rows=72) # 3 days
    
    # --- 核心修复：确保情绪数据的索引也带有时区 ---
    sent_dates = pd.to_datetime(['2023-01-01', '2023-01-03'], utc=True)
    sent_data = {'sent_mean': [0.5, -0.5], 'sent_median': [0.5, -0.5], 'count': [10, 5]}
    daily_sent_df = pd.DataFrame(sent_data, index=sent_dates)

    merged_df = merge_price_and_sentiment(price_df, daily_sent_df)

    assert merged_df.loc['2023-01-01', 'sent_mean'].iloc[0] == 0.5
    assert merged_df.loc['2023-01-02', 'sent_mean'].iloc[0] == 0.5
    assert merged_df.loc['2023-01-03', 'sent_mean'].iloc[0] == -0.5
    assert merged_df['sent_mean'].iloc[-1] == -0.5
    assert not merged_df[['sent_mean', 'sent_median', 'count']].isnull().values.any()

def test_generate_programmatic_features_extensibility(monkeypatch):
    """
    Tests that the feature generation is extensible.
    """
    df = create_base_df(rows=300) 
    
    from features import feature_engineering

    df_default = feature_engineering._generate_programmatic_features(df.copy())
    num_cols_default = len(df_default.columns)

    original_params = feature_engineering.FEATURE_PARAMS.copy()
    extended_params = original_params.copy()
    extended_params["ma_windows"] = original_params["ma_windows"] + [150, 250]
    monkeypatch.setattr(feature_engineering, 'FEATURE_PARAMS', extended_params, raising=True)

    df_extended = feature_engineering._generate_programmatic_features(df.copy())
    num_cols_extended = len(df_extended.columns)

    # We added 2 new MA windows. This should result in:
    # sma_150, ema_150, sma_250, ema_250 (4 cols)
    # sma_spread_x_150, ema_spread_x_150 (multiple new spread cols)
    # The exact number of spread columns depends on the `fast` loop.
    # The spread calculation is hardcoded and doesn't use the new windows, so exactly 4 columns are added.
    assert num_cols_extended == num_cols_default + 4

def test_generate_features_with_mined_feature(tmp_path):
    """
    Tests that generate_features can load a formula from a JSON file.
    """
    mined_feature_data = {
        "name": "gp_test_feature_add_close_vol",
        "formula": "add(X0, X1)",
        "base_features": ["close", "vol"],
        "performance": {"accuracy": 0.65}
    }
    mined_features_path = tmp_path / "mined_features.json"
    with open(mined_features_path, 'w') as f:
        json.dump(mined_feature_data, f)

    df = create_base_df(rows=10)

    df_with_mined_feature = generate_features(df.copy(), mined_features_path=str(mined_features_path))

    expected_new_col = mined_feature_data["name"]
    assert expected_new_col in df_with_mined_feature.columns, f"Mined feature column '{expected_new_col}' was not created."

    expected_values = df['close'] + df['vol']
    
    pd.testing.assert_series_equal(df_with_mined_feature[expected_new_col], expected_values, check_names=False)
