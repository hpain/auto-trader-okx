# config/model_config.py

MODEL_PARAMS = {
    "logreg": {
        # 默认参数
        "solver": "lbfgs",
        "max_iter": 1000,
        "use_scaler": True,

        # Optuna 搜索空间
        "C_range": [1e-3, 1e2]
    },

    # 未来可以加其它模型的默认配置
    # "lgbm": {...}
}
