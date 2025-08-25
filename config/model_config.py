MODEL_PARAMS = {
    "logreg": {
        "solver": "lbfgs",
        "max_iter": 1000,
        "use_scaler": True,
        "C_range": [2, 12],   # 原来是 [1e-3, 1e2]
        "buy_th_range": [0.58, 0.62],
        "sell_th_range": [0.44, 0.48],
        "fee_range": [0.0002, 0.0010]
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
