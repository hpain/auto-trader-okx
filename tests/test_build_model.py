# tests/test_build_model.py

import pytest
import copy
from unittest.mock import patch

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

from models.build_model import build_model, RANDOM_STATE

# This dictionary simulates the structure that the build_model function
# expects to find within the global `config` object.
MOCK_CONFIG = {
    'model': {
        'hyperparameters': {
            "logreg": {
                "solver": "lbfgs",
                "max_iter": 1000,
                "use_scaler": True,
                "C": 1.0
            },
            "lgb": {
                "num_leaves": 31,
                "learning_rate": 0.1
            },
            "rf": {
                # RF in build_model doesn't use config defaults, but we include it for completeness
                "n_estimators": 200 
            }
        }
    }
}

# By patching 'models.build_model.config', we replace the global config proxy
# with our mock dictionary for the duration of the tests in this class.
@patch('models.build_model.config', MOCK_CONFIG)
class TestBuildModel:
    
    def test_build_logreg_model(self):
        """Tests building a logistic regression model with default and custom params."""
        # 1. Test with default parameters from mock config
        model = build_model("logreg")
        assert isinstance(model, Pipeline)
        assert isinstance(model.steps[0][1], StandardScaler)
        clf = model.steps[1][1].estimator # It's wrapped in CalibratedClassifierCV
        assert isinstance(clf, LogisticRegression)
        assert clf.C == MOCK_CONFIG['model']['hyperparameters']["logreg"]["C"]
        assert clf.random_state == RANDOM_STATE

        # 2. Test without scaler, using deepcopy to prevent state pollution
        mock_config_no_scaler = copy.deepcopy(MOCK_CONFIG)
        mock_config_no_scaler['model']['hyperparameters']['logreg']['use_scaler'] = False
        with patch('models.build_model.config', mock_config_no_scaler):
            model_no_scaler = build_model("logreg")
            assert not isinstance(model_no_scaler, Pipeline)
            assert isinstance(model_no_scaler.estimator, LogisticRegression)

        # 3. Test with custom C value passed as kwarg.
        # This now runs in a clean state because MOCK_CONFIG was not mutated.
        custom_c = 100.0
        model_custom = build_model("logreg", C=custom_c)
        assert isinstance(model_custom, Pipeline) # Verify it's a pipeline
        clf_custom = model_custom.steps[1][1].estimator
        assert clf_custom.C == custom_c

    def test_build_rf_model(self):
        """Tests building a random forest model with default and custom params."""
        # 1. Test with default parameters (hardcoded in build_model)
        model = build_model("rf")
        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 200 # Default value in function
        assert model.random_state == RANDOM_STATE

        # 2. Test with custom parameters
        model_custom = build_model("rf", n_estimators=300, max_depth=10)
        assert model_custom.n_estimators == 300
        assert model_custom.max_depth == 10

    def test_build_lgb_model(self):
        """Tests building a LightGBM model with default and custom params."""
        # 1. Test with default parameters from mock config
        model = build_model("lgb")
        assert isinstance(model, lgb.LGBMClassifier)
        assert model.num_leaves == MOCK_CONFIG['model']['hyperparameters']["lgb"]["num_leaves"]
        assert model.learning_rate == MOCK_CONFIG['model']['hyperparameters']["lgb"]["learning_rate"]
        assert model.random_state == RANDOM_STATE

        # 2. Test with custom parameters
        model_custom = build_model("lgb", num_leaves=50, learning_rate=0.05)
        assert model_custom.num_leaves == 50
        assert model_custom.learning_rate == 0.05

    def test_build_model_unknown_type(self):
        """Tests that a ValueError is raised for an unknown model type."""
        with pytest.raises(ValueError, match="Unknown model_type: xgboost"):
            build_model("xgboost")