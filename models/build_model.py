# models/build_model.py

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config.model_config import MODEL_PARAMS

def build_model(model_type: str, **kwargs):
    """
    根据模型类型返回模型实例。
    kwargs 会覆盖 MODEL_PARAMS 中的默认值。
    """
    if model_type not in MODEL_PARAMS:
        raise ValueError(f"Unknown model_type: {model_type}")

    # 合并默认参数和传入参数
    params = {**MODEL_PARAMS[model_type], **kwargs}

    if model_type == "logreg":
        base_model = LogisticRegression(
            solver=params["solver"],
            max_iter=params["max_iter"],
            C=params.get("C", 1.0)
        )
        if params["use_scaler"]:
            return make_pipeline(StandardScaler(), base_model)
        else:
            return base_model

    # 其他模型...
    raise NotImplementedError(f"Model builder not implemented for {model_type}")
