# models/evolution.py
import os
import json
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
import logging
import csv
from joblib import dump
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

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
    interval: str,
):
    X = data[feature_cols]
    y = data["y"]

    # Define summary log path and clear previous file
    summary_log_path = os.path.join(out_dir, "trials_summary.csv")
    if os.path.exists(summary_log_path):
        os.remove(summary_log_path)

    def save_trial_summary_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        """Callback to save trial summary to a CSV file."""
        data_to_write = {
            "trial_number": trial.number,
            "stability_score": trial.value,
            "overall_sharpe": trial.user_attrs.get("overall_sharpe_for_log"),
            "fold_sharpes": str([round(s, 4) for s in trial.user_attrs.get("fold_sharpes", [])]),
            "params": json.dumps(trial.params)
        }

        file_exists = os.path.exists(summary_log_path)
        with open(summary_log_path, "a", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data_to_write.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data_to_write)

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
            model = lgb.LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1, **params)
        else: # logreg
            params = {"C": trial.suggest_float("C", 1e-4, 1e2, log=True)}
            model = LogisticRegression(random_state=42, solver="liblinear", class_weight='balanced', **params)

        conf_threshold = trial.suggest_float("confidence_threshold", 0.5, 0.95)

        tscv = TimeSeriesSplit(n_splits=5)
        # --- Sharpe Ratio Calculation Setup ---
        trading_periods_per_year = {
            "1m": 252 * 24 * 60, "5m": 252 * 24 * 12, "15m": 252 * 24 * 4,
            "30m": 252 * 24 * 2, "1H": 252 * 24, "4H": 252 * 6, "1D": 252,
        }.get(interval, 252)
        annualization_factor = np.sqrt(trading_periods_per_year)

        all_returns_series = []
        all_success_rates = []
        all_trade_counts = []
        all_fold_sharpes = [] # For diagnostics

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
            
            # Calculate and store sharpe for this fold
            if returns_series.std() != 0 and len(returns_series.loc[returns_series != 0]) > 10:
                fold_sharpe = (returns_series.mean() / returns_series.std()) * annualization_factor
                all_fold_sharpes.append(fold_sharpe if np.isfinite(fold_sharpe) else 0.0)
            else:
                all_fold_sharpes.append(0.0)

            all_returns_series.append(returns_series)
            all_trade_counts.append(trade_count)
            if trade_count > 0:
                all_success_rates.append(success_rate)

        final_returns = pd.concat(all_returns_series) if all_returns_series else pd.Series(dtype=float)

        # --- MODIFIED OBJECTIVE ---
        # Calculate stability-adjusted score instead of overall sharpe
        if len(all_fold_sharpes) > 0:
            mean_sharpe = np.mean(all_fold_sharpes)
            std_sharpe = np.std(all_fold_sharpes)
            stability_score = mean_sharpe - std_sharpe
        else:
            stability_score = 0.0

        if not np.isfinite(stability_score):
            stability_score = 0.0

        # For logging purposes, we can still calculate the overall sharpe
        if final_returns.std() == 0 or len(final_returns.loc[final_returns != 0]) < 10:
            overall_sharpe_for_log = 0.0
        else:
            overall_sharpe_for_log = (final_returns.mean() / final_returns.std()) * annualization_factor
        if not np.isfinite(overall_sharpe_for_log):
            overall_sharpe_for_log = 0.0

        avg_success_rate = np.mean(all_success_rates) if all_success_rates else 0.0
        avg_trade_count = np.mean(all_trade_counts)
        
        logging.info(
            f"Trial {trial.number}: "
            f"StabilityScore={stability_score:.2f}, "
            f"Sharpe(log)={overall_sharpe_for_log:.2f}, "
            f"Fold Sharpes={[round(s, 2) for s in all_fold_sharpes]}, "
            f"SuccessRate={avg_success_rate:.2%}, "
            f"Trades={avg_trade_count:.1f}, "
            f"Params={trial.params}"
        )

        trial.set_user_attr("fold_sharpes", all_fold_sharpes)
        trial.set_user_attr("overall_sharpe_for_log", overall_sharpe_for_log)

        return stability_score

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials, n_jobs=-1, callbacks=[save_trial_summary_callback])
    
    best_score = study.best_value
    best_params = study.best_params

    # --- Save Best Trial Details ---
    best_trial = study.best_trial
    best_trial_details = {
        "trial_number": best_trial.number,
        "stability_score": best_trial.value,
        "overall_sharpe": best_trial.user_attrs.get("overall_sharpe_for_log"),
        "fold_sharpes": [round(s, 4) for s in best_trial.user_attrs.get("fold_sharpes", [])],
        "params": best_trial.params,
    }
    best_trial_path = os.path.join(out_dir, "best_trial_details.json")
    with open(best_trial_path, "w", encoding="utf-8") as f:
        json.dump(best_trial_details, f, ensure_ascii=False, indent=2)
    logging.info(f"💾 Best trial details saved to: {best_trial_path}")

    # --- Retraining and Saving ---
    logging.info("\n--- Retraining best model on full data and saving ---")
    os.makedirs(out_dir, exist_ok=True)

    model_type = best_params.pop('model', None)
    if model_type is None:
        raise ValueError("Best params from Optuna study does not contain 'model' key.")

    final_model = build_model(model_type=model_type, **best_params)
    final_model.fit(X, y)
    logging.info(f"✅ Final model trained: {final_model}")

    model_path = os.path.join(out_dir, "best_model.pkl")
    dump(final_model, model_path)
    logging.info(f"💾 Model saved to: {model_path}")

    # --- Feature Importance Plot ---
    if hasattr(final_model, 'feature_importances_'):
        feature_imp = pd.Series(final_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        plt.figure(figsize=(10, 8))
        plt.title("Feature Importance")
        feature_imp.plot(kind='barh')
        plt.tight_layout()
        plot_path = os.path.join(out_dir, "feature_importance.png")
        plt.savefig(plot_path)
        logging.info(f"💾 Feature importance plot saved to: {plot_path}")

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
    logging.info(f"💾 Metadata saved to: {metadata_path}")
    
    return best_score, best_params