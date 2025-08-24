# config/model_config.py
MODEL_PARAMS = {
    "logreg": {
        "solver": "lbfgs",
        "max_iter": 1000,
        "use_scaler": True,
        "C_range": [1e-3, 1e2]
    },
    "rf": {
        "n_estimators_range": [100, 500],
        "max_depth_range": [3, 12],
        "min_samples_leaf_range": [1, 8]
    },
    "lgb": {
        "num_leaves_range": [15, 63],
        "learning_rate_range": [0.01, 0.2]
    }
}

