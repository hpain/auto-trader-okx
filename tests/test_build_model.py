# tests/test_build_model.py

import pytest
import sys
import os
from unittest.mock import patch

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

# Append project root to path to allow importing project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.build_model import build_model, RANDOM_STATE

# Define the mock parameters that will be patched into the module
_dummy_model_params = {
    "logreg": {
        "solver": "lbfgs",
        "max_iter": 1000,
        "use_scaler": True,
        "C": 1.0
    },
    "lgb": {
        "num_leaves": 31,
        "learning_rate": 0.1
    }
}

@patch.dict('config.model_config.MODEL_PARAMS', _dummy_model_params)
def test_build_logreg_model():
    """Tests building a logistic regression model with default and custom params."""
    # 1. Test with default parameters (and scaler)
    model = build_model("logreg")
    assert isinstance(model, Pipeline)
    assert isinstance(model.steps[0][1], StandardScaler)
    clf = model.steps[1][1].estimator # It's wrapped in CalibratedClassifierCV
    assert isinstance(clf, LogisticRegression)
    assert clf.C == _dummy_model_params["logreg"]["C"]
    assert clf.random_state == RANDOM_STATE

    # 2. Test without scaler
    model_no_scaler = build_model("logreg", use_scaler=False)
    assert not isinstance(model_no_scaler, Pipeline)
    assert isinstance(model_no_scaler.estimator, LogisticRegression)

    # 3. Test with custom C value
    custom_c = 100.0
    model_custom = build_model("logreg", C=custom_c)
    clf_custom = model_custom.steps[1][1].estimator
    assert clf_custom.C == custom_c

def test_build_rf_model():
    """Tests building a random forest model with default and custom params."""
    # 1. Test with default parameters
    model = build_model("rf")
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 200 # Default value in function
    assert model.random_state == RANDOM_STATE

    # 2. Test with custom parameters
    model_custom = build_model("rf", n_estimators=300, max_depth=10)
    assert model_custom.n_estimators == 300
    assert model_custom.max_depth == 10

@patch.dict('config.model_config.MODEL_PARAMS', _dummy_model_params)
def test_build_lgb_model():
    """Tests building a LightGBM model with default and custom params."""
    # 1. Test with default parameters from mock config
    model = build_model("lgb")
    assert isinstance(model, lgb.LGBMClassifier)
    assert model.num_leaves == _dummy_model_params["lgb"]["num_leaves"]
    assert model.learning_rate == _dummy_model_params["lgb"]["learning_rate"]
    assert model.random_state == RANDOM_STATE

    # 2. Test with custom parameters
    model_custom = build_model("lgb", num_leaves=50, learning_rate=0.05)
    assert model_custom.num_leaves == 50
    assert model_custom.learning_rate == 0.05

def test_build_model_unknown_type():
    """Tests that a ValueError is raised for an unknown model type."""
    with pytest.raises(ValueError, match="Unknown model_type: xgboost"):
        build_model("xgboost")