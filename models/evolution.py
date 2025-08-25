# models/evolution.py
import json, os
import numpy as np
import pandas as pd
import optuna
from joblib import dump
from .backtest import simple_backtest
from models.build_model import build_model
from config.model_config import MODEL_PARAMS

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

def objective(trial: optuna.Trial, df: pd.DataFrame, feature_cols, model_choices=None):
    if model_choices is None:
        model_choices = list(MODEL_PARAMS.keys())
    model_type = trial.suggest_categorical("model_type", model_choices)
    
    if model_type == "logreg":
        C = trial.suggest_float("C", *MODEL_PARAMS["logreg"]["C_range"], log=True)
        model = build_model(model_type, C=C)
    elif model_type == "rf":
        model = build_model(
            model_type,
            n_estimators=trial.suggest_int("n_estimators", *MODEL_PARAMS["rf"]["n_estimators_range"], step=50),
            max_depth=trial.suggest_int("max_depth", *MODEL_PARAMS["rf"]["max_depth_range"]),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", *MODEL_PARAMS["rf"]["min_samples_leaf_range"])
        )
    elif model_type == "lgb":
        model = build_model(
            model_type,
            num_leaves=trial.suggest_int("num_leaves", *MODEL_PARAMS["lgb"]["num_leaves_range"], step=4),
            learning_rate=trial.suggest_float("learning_rate", *MODEL_PARAMS["lgb"]["learning_rate_range"], log=True)
        )

    buy_th = trial.suggest_float("buy_th", 0.45, 0.75)
    sell_th = trial.suggest_float("sell_th", 0.25, 0.55)
    fee = trial.suggest_float("fee", 0.0003, 0.0010)

    folds = split_walk_forward(df, n_splits=5)
    sh_list, trades_list, mdd_list = [], [], []

    for tr_idx, va_idx in folds:
        tr, va = df.iloc[tr_idx], df.iloc[va_idx]
        X_tr, y_tr = tr[feature_cols], tr["y"]
        X_va, y_va = va[feature_cols], va["y"]
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_va)[:, 1]
        va2 = va.copy()
        va2["proba"] = proba
        bt = simple_backtest(va2, "proba", buy_th, sell_th, fee)
        sh_list.append(bt["sharpe"])
        trades_list.append(bt["trades"])
        mdd_list.append(bt.get("max_dd", 0))

    mean_sharpe = np.mean(sh_list[:-1])
    oos_sharpe = sh_list[-1]
    mean_trades = np.mean(trades_list)
    mean_mdd = np.mean(mdd_list)

    if mean_trades > 200:
        mean_sharpe -= 0.5
    if mean_mdd < -0.15:
        mean_sharpe -= 0.2

    trial.set_user_attr("oos_sharpe", oos_sharpe)
    trial.set_user_attr("trades", mean_trades)
    trial.set_user_attr("mean_mdd", mean_mdd)

    return mean_sharpe

def train_evolve(df: pd.DataFrame, feature_cols, out_dir="models", n_trials=50, patience=10, model_list=None):
    from config.model_config import MODEL_PARAMS
    os.makedirs(out_dir, exist_ok=True)

    # 过滤模型搜索空间
    if model_list is not None:
        model_choices = [m for m in model_list if m in MODEL_PARAMS]
        if not model_choices:
            raise ValueError(f"❌ 未找到指定模型: {model_list}")
    else:
        model_choices = list(MODEL_PARAMS.keys())

    results = []
    best_so_far = -1e9
    no_improve_rounds = 0

    def log_objective(trial):
        nonlocal best_so_far, no_improve_rounds
        score = objective(trial, df, feature_cols, model_choices=model_choices)
        results.append({
            **trial.params,
            "mean_sharpe": score,
            "oos_sharpe": trial.user_attrs.get("oos_sharpe"),
            "trades": trial.user_attrs.get("trades"),
            "mean_mdd": trial.user_attrs.get("mean_mdd")
        })
        gap = score - best_so_far
        if score > best_so_far + 1e-6:
            best_so_far = score
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1
        if no_improve_rounds >= patience:
            raise optuna.exceptions.TrialPruned()
        if gap < -1.0:
            raise optuna.exceptions.TrialPruned()
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(log_objective, n_trials=n_trials, show_progress_bar=False)

    pd.DataFrame(results).to_csv(os.path.join(out_dir, "optuna_trials.csv"), index=False)
    best_params = study.best_trial.params
    best_model = build_model(**best_params)
    X, y = df[feature_cols], df["y"]
    best_model.fit(X, y)
    dump(best_model, os.path.join(out_dir, "best_model.pkl"))
    meta = {
        "feature_cols": feature_cols,
        "best_params": best_params,
        "n_samples": int(len(df))
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # OOS 评估保持原逻辑...

   # === 固定留出集 OOS 评估 ===
    split_point = int(len(df) * 0.85)

    # 用最佳参数重新训练模型（仅用前 85% 数据）
    oos_model = build_model(**best_params)
    oos_model.fit(X.iloc[:split_point], y.iloc[:split_point])

    # 在留出集上预测
    proba = oos_model.predict_proba(X.iloc[split_point:])[:, 1]

    # 复制留出集 DataFrame 并添加预测概率
    oos_df = df.iloc[split_point:].copy()
    oos_df["proba"] = proba

    # 回测留出集表现
    bt = simple_backtest(
        oos_df,
        proba_col="proba",
        buy_th=best_params["buy_th"],
        sell_th=best_params["sell_th"],
        fee=best_params["fee"]
    )

    print("OOS Sharpe:", bt["sharpe"])
    print("OOS MDD:", bt.get("max_drawdown", 0))
    print("OOS Trades:", bt["trades"])

