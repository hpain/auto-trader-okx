# models/evolution.py
import json, os
import numpy as np
import pandas as pd
import optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
from joblib import dump
from .backtest import simple_backtest

RANDOM_STATE = 42

def split_walk_forward(df: pd.DataFrame, n_splits=5):
    n = len(df)
    idx = np.arange(n)
    folds = []
    step = n // (n_splits + 1)
    for i in range(1, n_splits + 1):
        cut = step * i
        train_idx = idx[:cut]
        valid_idx = idx[cut: cut + step]
        if len(valid_idx) == 0:
            break
        folds.append((train_idx, valid_idx))
    return folds

def make_model(trial: optuna.Trial):
    model_type = trial.suggest_categorical("model_type", ["logreg", "rf", "lgb"])
    if model_type == "logreg":
        C = trial.suggest_float("C", 0.01, 10.0, log=True)
        base = LogisticRegression(C=C, max_iter=200, random_state=RANDOM_STATE)
        clf = CalibratedClassifierCV(base, cv=3)
    elif model_type == "rf":
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
    else:  # LightGBM
        num_leaves = trial.suggest_int("num_leaves", 15, 63, step=4)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
        clf = lgb.LGBMClassifier(
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    return Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", clf)
    ])

def objective(trial: optuna.Trial, df: pd.DataFrame, feature_cols):
    pipe = make_model(trial)
    buy_th = trial.suggest_float("buy_th", 0.45, 0.75)
    sell_th = trial.suggest_float("sell_th", 0.25, 0.55)
    fee = trial.suggest_float("fee", 0.0003, 0.0010)

    folds = split_walk_forward(df, n_splits=5)
    sh_list, trades_list, mdd_list = [], [], []

    for i, (tr_idx, va_idx) in enumerate(folds):
        tr, va = df.iloc[tr_idx], df.iloc[va_idx]
        X_tr, y_tr = tr[feature_cols], tr["y"]
        X_va, y_va = va[feature_cols], va["y"]

        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_va)[:, 1]

        va2 = va.copy()
        va2["proba"] = proba
        bt = simple_backtest(va2, proba_col="proba", buy_th=buy_th, sell_th=sell_th, fee=fee)

        sh_list.append(bt["sharpe"])
        trades_list.append(bt["trades"])
        mdd_list.append(bt.get("max_dd", 0))

    mean_sharpe = np.mean(sh_list[:-1])  # 前几段作为验证
    oos_sharpe = sh_list[-1]              # 最后一段作为OOS
    mean_trades = np.mean(trades_list)
    mean_mdd = np.mean(mdd_list)

    # 惩罚高换手和高回撤
    if mean_trades > 200:
        mean_sharpe -= 0.5
    if mean_mdd < -0.15:  # 回撤超过15%扣分
        mean_sharpe -= 0.2

    trial.set_user_attr("oos_sharpe", oos_sharpe)
    trial.set_user_attr("trades", mean_trades)
    trial.set_user_attr("mean_mdd", mean_mdd)
    return mean_sharpe

def train_evolve(df: pd.DataFrame, feature_cols, out_dir="models", n_trials=50, patience=10):
    os.makedirs(out_dir, exist_ok=True)

    results = []
    best_so_far = -1e9
    no_improve_rounds = 0

    def log_objective(trial):
        nonlocal best_so_far, no_improve_rounds
        score = objective(trial, df, feature_cols)
        results.append({
            **trial.params,
            "mean_sharpe": score,
            "oos_sharpe": trial.user_attrs.get("oos_sharpe"),
            "trades": trial.user_attrs.get("trades"),
            "mean_mdd": trial.user_attrs.get("mean_mdd")
        })

        # 记录和最佳的差距
        gap = score - best_so_far
        
        # 早停逻辑
        if score > best_so_far + 1e-6:
            best_so_far = score
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1
            if no_improve_rounds >= patience:
                print(f"⚠️ Trial {trial.number} pruned: 连续 {patience} 次无提升 (gap={gap:.4f})")
                raise optuna.exceptions.TrialPruned()
                
        # 如果当前得分比最佳差太多，也直接剪掉
        if gap < -1.0:  # 比最佳低 1.0 Sharpe
            print(f"⚠️ Trial {trial.number} pruned: 当前 score 比最佳低 {abs(gap):.3f}")
            raise optuna.exceptions.TrialPruned()
            
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(log_objective, n_trials=n_trials, show_progress_bar=False)

    pd.DataFrame(results).to_csv(os.path.join(out_dir, "optuna_trials.csv"), index=False)

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
