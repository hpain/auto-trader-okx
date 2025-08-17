# models/evolution.py
import json, os
import numpy as np
import pandas as pd
import optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from joblib import dump

from .backtest import simple_backtest

RANDOM_STATE = 42

def split_walk_forward(df: pd.DataFrame, n_splits=4):
    """
    简单walk-forward：按时间切成 n_splits 段，依次扩容训练，最后一段留作验证。
    返回 [(train_idx, valid_idx), ...]
    """
    n = len(df)
    idx = np.arange(n)
    folds = []
    step = n // (n_splits + 1)
    for i in range(1, n_splits+1):
        cut = step * i
        train_idx = idx[:cut]
        valid_idx = idx[cut: cut + step]
        if len(valid_idx) == 0: break
        folds.append((train_idx, valid_idx))
    return folds

def make_model(trial: optuna.Trial):
    model_type = trial.suggest_categorical("model_type", ["logreg", "rf"])
    if model_type == "logreg":
        C = trial.suggest_float("C", 0.01, 10.0, log=True)
        clf = LogisticRegression(C=C, max_iter=200, random_state=RANDOM_STATE)
    else:
        n_estimators = trial.suggest_int("n_estimators", 100, 500, step=50)
        max_depth = trial.suggest_int("max_depth", 3, 12)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 8)
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),  # 稀疏安全
        ("clf", clf)
    ])
    return pipe

def objective(trial: optuna.Trial, df: pd.DataFrame, feature_cols, fee=0.0005):
    pipe = make_model(trial)
    buy_th = trial.suggest_float("buy_th", 0.50, 0.65)
    sell_th = trial.suggest_float("sell_th", 0.35, 0.50)

    folds = split_walk_forward(df, n_splits=4)
    scores = []
    for tr_idx, va_idx in folds:
        tr, va = df.iloc[tr_idx], df.iloc[va_idx]
        X_tr, y_tr = tr[feature_cols], tr["y"]
        X_va, y_va = va[feature_cols], va["y"]
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_va)[:,1]
        # 以回测指标为优化目标（例如Sharpe）
        va2 = va.copy()
        va2["proba"] = proba
        bt = simple_backtest(va2, proba_col="proba", buy_th=buy_th, sell_th=sell_th, fee=fee)
        scores.append(bt["sharpe"])
    # 最大化sharpe
    return np.mean(scores)

def train_evolve(df: pd.DataFrame, feature_cols, out_dir="models", n_trials=30):
    os.makedirs(out_dir, exist_ok=True)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, df, feature_cols), n_trials=n_trials, show_progress_bar=False)

    # 用最优参数在全量上fit并保存
    best_params = study.best_trial.params
    best_pipe = make_model(optuna.trial.FixedTrial(best_params))
    X, y = df[feature_cols], df["y"]
    best_pipe.fit(X, y)

    dump(best_pipe, os.path.join(out_dir, "best_model.pkl"))
    meta = {
        "feature_cols": feature_cols,
        "best_params": best_params,
        "n_samples": int(len(df))
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return study.best_value, best_params
