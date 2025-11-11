import pytest
import os
import json
from joblib import dump
import pandas as pd
from unittest.mock import patch, MagicMock

# Ensure project root is on sys.path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.model_pipeline import validate_model

# Helper function to create a dummy model and metadata
def create_dummy_model_files(model_dir, model_name, confidence_threshold):
    """Creates a fake model.pkl and metadata.json in the specified directory."""
    os.makedirs(model_dir, exist_ok=True)
    
    # Create a very simple dummy model object (e.g., a dictionary)
    dummy_model = {"name": model_name}
    dump(dummy_model, os.path.join(model_dir, "best_model.pkl"))
    
    # Create dummy metadata
    meta = {
        "best_params": {
            "confidence_threshold": confidence_threshold
        }
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(meta, f)

@pytest.fixture
def setup_test_environment(tmp_path):
    """
    Pytest fixture to set up a temporary directory structure for testing.
    tmp_path is a built-in pytest fixture that provides a temporary directory.
    """
    prod_dir = tmp_path / "prod_models"
    staging_dir = tmp_path / "staging_models"
    
    # Create dummy validation data
    validation_df = pd.DataFrame({
        'close': [100, 101, 102, 103, 104],
        'feature1': [1, 2, 3, 4, 5]
    })
    feature_cols = ['feature1']
    
    return {
        "prod_dir": str(prod_dir),
        "staging_dir": str(staging_dir),
        "validation_df": validation_df,
        "feature_cols": feature_cols
    }

@patch('tools.model_pipeline.run_backtest')
def test_validate_model_approve_deployment(mock_run_backtest, setup_test_environment):
    """
    Test Case 1: New model is significantly better and should be approved.
    """
    # Arrange
    env = setup_test_environment
    create_dummy_model_files(env["prod_dir"], "prod_model", 0.6)
    create_dummy_model_files(env["staging_dir"], "new_model", 0.65)

    # Mock run_backtest to return different results for each model
    # New model returns a high Sharpe, old model returns a low Sharpe
    mock_run_backtest.side_effect = [
        (0, 0, 0, 0, pd.Series([0.01, 0.01, -0.005])),  # New model's returns
        (0, 0, 0, 0, pd.Series([0.001, 0.001, -0.005])) # Prod model's returns
    ]

    # Act
    is_approved = validate_model(
        new_model_dir=env["staging_dir"],
        prod_model_dir=env["prod_dir"],
        validation_df=env["validation_df"],
        feature_cols=env["feature_cols"],
        improvement_threshold=0.10 # 10%
    )

    # Assert
    assert is_approved is True
    assert mock_run_backtest.call_count == 2

@patch('tools.model_pipeline.run_backtest')
def test_validate_model_deny_worse(mock_run_backtest, setup_test_environment):
    """
    Test Case 2: New model is worse than the production model and should be denied.
    """
    # Arrange
    env = setup_test_environment
    create_dummy_model_files(env["prod_dir"], "prod_model", 0.6)
    create_dummy_model_files(env["staging_dir"], "new_model", 0.65)

    # New model returns a low Sharpe, old model returns a high Sharpe
    mock_run_backtest.side_effect = [
        (0, 0, 0, 0, pd.Series([0.001, 0.001, -0.005])), # New model's returns
        (0, 0, 0, 0, pd.Series([0.01, 0.01, -0.005]))   # Prod model's returns
    ]

    # Act
    is_approved = validate_model(
        env["staging_dir"], env["prod_dir"], env["validation_df"], env["feature_cols"]
    )

    # Assert
    assert is_approved is False

@patch('tools.model_pipeline.run_backtest')
def test_validate_model_deny_insufficient_improvement(mock_run_backtest, setup_test_environment):
    """
    Test Case 3: New model is slightly better but not enough to pass the threshold.
    """
    # Arrange
    env = setup_test_environment
    create_dummy_model_files(env["prod_dir"], "prod_model", 0.6)
    create_dummy_model_files(env["staging_dir"], "new_model", 0.65)

    # New model returns a slightly better Sharpe
    mock_run_backtest.side_effect = [
        (0, 0, 0, 0, pd.Series([0.0105, 0.0105, -0.005])), # New model (Sharpe will be ~5% better)
        (0, 0, 0, 0, pd.Series([0.01, 0.01, -0.005]))      # Prod model
    ]

    # Act
    is_approved = validate_model(
        env["staging_dir"], env["prod_dir"], env["validation_df"], env["feature_cols"], improvement_threshold=0.10 # 10% threshold
    )

    # Assert
    assert is_approved is False