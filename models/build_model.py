# models/build_model.py
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
from config.model_config import MODEL_PARAMS

RANDOM_STATE = 42

def build_model(model_type: str, **kwargs):
    if model_type == "logreg":
        params = {**MODEL_PARAMS["logreg"], **kwargs}
        base = LogisticRegression(
            C=params.get("C", 1.0),
            solver=params["solver"],
            max_iter=params["max_iter"],
            random_state=RANDOM_STATE
        )
        # 保留原来的概率校准
        clf = CalibratedClassifierCV(base, cv=3)
        return make_pipeline(StandardScaler(), clf) if params["use_scaler"] else clf

    elif model_type == "rf":
        return RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", None),
            min_samples_leaf=kwargs.get("min_samples_leaf", 1),
            n_jobs=-1,
            random_state=RANDOM_STATE
        )

    elif model_type == "lgb":
        return lgb.LGBMClassifier(
            num_leaves=kwargs.get("num_leaves", 31),
            learning_rate=kwargs.get("learning_rate", 0.1),
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    raise ValueError(f"Unknown model_type: {model_type}")
