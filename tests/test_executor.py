# tests/test_executor.py

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
import sys
import os

# Append project root to path to allow importing project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now we can import the module to be tested
from trader import executor

# Define a dummy config for all tests in this file
@pytest.fixture(autouse=True)
def mock_config():
    dummy_config_data = {
        "okx": {
            "api_key": "dummy_api_key",
            "secret_key": "dummy_secret_key",
            "passphrase": "dummy_passphrase",
            "flag": "0"
        },
        "trade": {
            "symbol": "BTC-USDT",
            "interval": "1H",
            "quantity": 0.001
        },
        "paths": {
            "model_dir": "mock_models_dir"
        }
    }
    with patch.dict(executor.config, dummy_config_data, clear=True):
        yield

@pytest.fixture
def mock_dependencies():
    """Central fixture to mock all external dependencies of the executor."""
    with patch('trader.executor.get_klines') as mock_get_klines:
        with patch('trader.executor.OKXClient') as mock_okx_client:
            with patch('trader.executor.ma_signal') as mock_ma_signal:
                with patch('trader.executor.plot_moving_averages') as mock_plot:
                    with patch('trader.executor._load_model') as mock_load_model:
                        # --- Setup mock return values ---
                        # Mock OKXClient instance and its methods
                        mock_client_instance = MagicMock()
                        mock_okx_client.return_value = mock_client_instance

                        # Mock get_klines to return a valid DataFrame
                        mock_df = pd.DataFrame({
                            'close': [100, 110, 120, 130, 140, 150],
                            'feature1': [1, 2, 3, 4, 5, 6]
                        })
                        mock_get_klines.return_value = mock_df

                        # Mock the ML model and its metadata
                        mock_pipe = MagicMock()
                        mock_meta = {
                            "feature_cols": ["feature1"],
                            "best_params": {"buy_th": 0.6, "sell_th": 0.4}
                        }
                        mock_load_model.return_value = (mock_pipe, mock_meta)

                        yield {
                            'get_klines': mock_get_klines,
                            'okx_client': mock_okx_client,
                            'client_instance': mock_client_instance,
                            'ma_signal': mock_ma_signal,
                            'plot': mock_plot,
                            'load_model': mock_load_model,
                            'pipe': mock_pipe
                        }

def test_run_model_buy_signal(mock_dependencies):
    """Test the happy path where the model predicts a BUY signal."""
    # Arrange: Mock the model to predict a high probability (buy)
    mock_dependencies['pipe'].predict_proba.return_value = np.array([[0.1, 0.9]]) # High prob of class 1 (buy)

    # Act
    executor.run()

    # Assert
    # 1. OKX client was initialized
    mock_dependencies['okx_client'].assert_called_once()
    # 2. Klines were fetched
    mock_dependencies['get_klines'].assert_called_once()
    # 3. Model was loaded
    mock_dependencies['load_model'].assert_called_once()
    # 4. Model was used for prediction
    mock_dependencies['pipe'].predict_proba.assert_called_once()
    # 5. MA signal was NOT used
    mock_dependencies['ma_signal'].assert_not_called()
    # 6. A buy order was placed
    mock_dependencies['client_instance'].place_order.assert_called_once_with(
        "BTC-USDT", "buy", 0.001
    )

def test_run_model_hold_signal(mock_dependencies):
    """Test the happy path where the model predicts a HOLD signal."""
    # Arrange: Mock the model to predict a neutral probability (hold)
    mock_dependencies['pipe'].predict_proba.return_value = np.array([[0.5, 0.5]])

    # Act
    executor.run()

    # Assert
    # 1. Prediction happened
    mock_dependencies['pipe'].predict_proba.assert_called_once()
    # 2. No order was placed
    mock_dependencies['client_instance'].place_order.assert_not_called()

def test_run_fallback_to_ma_on_model_load_fail(mock_dependencies):
    """Test fallback to MA strategy if the model fails to load."""
    # Arrange
    # 1. Simulate model loading failure
    mock_dependencies['load_model'].return_value = (None, None)
    # 2. Set a specific signal from the MA fallback
    mock_dependencies['ma_signal'].return_value = "buy"

    # Act
    executor.run()

    # Assert
    # 1. It tried to load the model
    mock_dependencies['load_model'].assert_called_once()
    # 2. It did NOT use the model for prediction
    mock_dependencies['pipe'].predict_proba.assert_not_called()
    # 3. It called the MA signal function
    mock_dependencies['ma_signal'].assert_called_once()
    # 4. It placed an order based on the MA signal
    mock_dependencies['client_instance'].place_order.assert_called_once_with(
        "BTC-USDT", "buy", 0.001
    )

def test_run_fallback_to_ma_on_missing_features(mock_dependencies):
    """Test fallback to MA strategy if the model requires missing features."""
    # Arrange
    # 1. Modify metadata to require a feature not in the DataFrame
    mock_pipe, mock_meta = mock_dependencies['load_model'].return_value
    mock_meta['feature_cols'] = ["feature1", "non_existent_feature"]
    # 2. Set a specific signal from the MA fallback
    mock_dependencies['ma_signal'].return_value = "sell"

    # Act
    executor.run()

    # Assert
    # 1. It tried to load the model
    mock_dependencies['load_model'].assert_called_once()
    # 2. It did NOT use the model for prediction
    mock_dependencies['pipe'].predict_proba.assert_not_called()
    # 3. It called the MA signal function
    mock_dependencies['ma_signal'].assert_called_once()
    # 4. It placed an order based on the MA signal
    mock_dependencies['client_instance'].place_order.assert_called_once_with(
        "BTC-USDT", "sell", 0.001
    )

def test_run_fallback_to_ma_on_inference_fail(mock_dependencies):
    """Test fallback to MA strategy if model inference raises an exception."""
    # Arrange
    # 1. Simulate a failure during model prediction
    mock_dependencies['pipe'].predict_proba.side_effect = Exception("Inference Error")
    # 2. Set a specific signal from the MA fallback
    mock_dependencies['ma_signal'].return_value = "hold"

    # Act
    executor.run()

    # Assert
    # 1. It tried to predict
    mock_dependencies['pipe'].predict_proba.assert_called_once()
    # 2. It fell back to the MA signal function
    mock_dependencies['ma_signal'].assert_called_once()
    # 3. No order was placed because the MA signal was "hold"
    mock_dependencies['client_instance'].place_order.assert_not_called()

@pytest.mark.parametrize("bad_data", [None, pd.DataFrame()])
def test_run_exit_on_data_fetch_fail(mock_dependencies, bad_data):
    """Test that the function exits early if kline data cannot be fetched."""
    # Arrange
    mock_dependencies['get_klines'].return_value = bad_data

    # Act
    executor.run()

    # Assert
    # 1. It tried to get klines
    mock_dependencies['get_klines'].assert_called_once()
    # 2. It did NOT continue to plotting or model loading
    mock_dependencies['plot'].assert_not_called()
    mock_dependencies['load_model'].assert_not_called()
    mock_dependencies['client_instance'].place_order.assert_not_called()

# --- Tests for _load_model --- 

@patch('trader.executor.os.path.join')
@patch('trader.executor.load')
@patch('trader.executor.logging')
def test_load_model_file_not_found(mock_logging, mock_joblib_load, mock_path_join):
    """Test _load_model's handling of FileNotFoundError."""
    # Arrange
    mock_joblib_load.side_effect = FileNotFoundError

    # Act
    pipe, meta = executor._load_model()

    # Assert
    assert pipe is None
    assert meta is None
    mock_logging.error.assert_not_called()

@patch('trader.executor.os.path.join')
@patch('trader.executor.load')
@patch('trader.executor.logging')
def test_load_model_other_exception(mock_logging, mock_joblib_load, mock_path_join):
    """Test _load_model's handling of other exceptions."""
    # Arrange
    mock_joblib_load.side_effect = Exception("Corrupted File")

    # Act
    pipe, meta = executor._load_model()

    # Assert
    assert pipe is None
    assert meta is None
    mock_logging.error.assert_called_once()