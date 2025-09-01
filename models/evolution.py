# models/evolution.py
import os
import json
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from joblib import dump
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .build_model import build_model
from utils.backtest import run_backtest

def train_evolve(
    data: pd.DataFrame,
    feature_cols: list,
    out_dir: str,
    n_trials: int,
    patience: int,
    model_list: list,
    profit_threshold: float,
    confidence_threshold: float,
    stop_loss_pct: float,
    max_drawdown_limit: float,
    success_rate_threshold: float,
):
    X = data[feature_cols]
    y = data["y"]

    def objective(trial):
        model_name = trial.suggest_categorical("model", model_list)
        
        if model_name == "rf":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            }
            model = RandomForestClassifier(random_state=42, class_weight='balanced', **params)
        elif model_name == "lgb":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            }
            model = lgb.LGBMClassifier(random_state=42, class_weight='balanced', **params)
        else: # logreg
            params = {"C": trial.suggest_float("C", 1e-4, 1e2, log=True)}
            model = LogisticRegression(random_state=42, solver="liblinear", class_weight='balanced', **params)

        conf_threshold = trial.suggest_float("confidence_threshold", 0.5, 0.95)

        tscv = TimeSeriesSplit(n_splits=5)
        all_returns_series = []
        all_success_rates = []
        all_trade_counts = []

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            model.fit(X_train, y_train)
            predictions = pd.Series(model.predict(X_test), index=X_test.index)
            probabilities = model.predict_proba(X_test)[:, 1]

            total_ret, max_dd, success_rate, trade_count, returns_series = run_backtest(
                predictions, probabilities, data.loc[X_test.index],
                conf_threshold, 
                stop_loss_pct,
            )
            
            all_returns_series.append(returns_series)
            all_trade_counts.append(trade_count)
            if trade_count > 0:
                all_success_rates.append(success_rate)

        final_returns = pd.concat(all_returns_series) if all_returns_series else pd.Series(dtype=float)
        
        if final_returns.std() == 0 or len(final_returns.loc[final_returns != 0]) < 10:
            sharpe_ratio = 0.0
        else:
            annualization_factor = np.sqrt(252)
            sharpe_ratio = (final_returns.mean() / final_returns.std()) * annualization_factor
        
        if not np.isfinite(sharpe_ratio):
            sharpe_ratio = 0.0

        avg_success_rate = np.mean(all_success_rates) if all_success_rates else 0.0
        avg_trade_count = np.mean(all_trade_counts)
        print(
            f"Trial {trial.number}: "
            f"Sharpe={sharpe_ratio:.2f}, "
            f"SuccessRate={avg_success_rate:.2%}, "
            f"Trades={avg_trade_count:.1f}, "
            f"Params={trial.params}"
        )

        return sharpe_ratio

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)
    
    best_score = study.best_value
    best_params = study.best_params

    # --- Retraining and Saving ---
    print("\n--- Retraining best model on full data and saving ---")
    os.makedirs(out_dir, exist_ok=True)

    model_type = best_params.pop('model', None)
    if model_type is None:
        raise ValueError("Best params from Optuna study does not contain 'model' key.")

    final_model = build_model(model_type=model_type, **best_params)
    final_model.fit(X, y)
    print(f"✅ Final model trained: {final_model}")

    model_path = os.path.join(out_dir, "best_model.pkl")
    dump(final_model, model_path)
    print(f"💾 Model saved to: {model_path}")

    meta = {
        "feature_cols": feature_cols,
        "best_params": study.best_trial.params,
        "n_samples": len(data),
        "best_score": best_score,
        "training_timestamp": pd.Timestamp.utcnow().isoformat()
    }
    metadata_path = os.path.join(out_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"💾 Metadata saved to: {metadata_path}")
    
    return best_score, best_params