# models/build_model.py
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
# Import the new central config object
from config import config

RANDOM_STATE = 42

def build_model(model_type: str, **kwargs):
    """Builds a model based on the specified type and hyperparameters."""
    # Get the model hyperparameters section from the new config
    model_hyperparams = config['model']['hyperparameters']

    if model_type == "logreg":
        # Combine defaults from config with any runtime kwargs from Optuna
        params = {**model_hyperparams["logreg"], **kwargs}
        base = LogisticRegression(
            # C is a range in config for Optuna, so we rely on kwargs from the study,
            # or provide a sensible default if called directly.
            C=params.get("C", 1.0),
            solver=params.get("solver", "lbfgs"),
            max_iter=params.get("max_iter", 1000),
            random_state=RANDOM_STATE
        )
        # The old config had 'use_scaler', which is not a standard param.
        # We'll keep this logic, defaulting to True.
        use_scaler = params.get("use_scaler", True)
        clf = CalibratedClassifierCV(base, cv=3)
        return make_pipeline(StandardScaler(), clf) if use_scaler else clf

    elif model_type == "rf":
        # This part primarily uses kwargs from Optuna, so it remains largely unchanged.
        return RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", None),
            min_samples_leaf=kwargs.get("min_samples_leaf", 1),
            n_jobs=-1,
            random_state=RANDOM_STATE
        )

    elif model_type == "lgb":
        # This part also primarily uses kwargs from Optuna.
        return lgb.LGBMClassifier(
            num_leaves=kwargs.get("num_leaves", 31),
            learning_rate=kwargs.get("learning_rate", 0.1),
            n_estimators=kwargs.get("n_estimators", 200),
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    raise ValueError(f"Unknown model_type: {model_type}")
