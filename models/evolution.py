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
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

# Import the new backtesting function for regression models
from utils.backtest import run_backtest_regression

def train_evolve(
    data: pd.DataFrame,
    feature_cols: list,
    out_dir: str,
    n_trials: int,
    model_list: list, # Ignored, for signature consistency
    profit_threshold: float, # Ignored, for signature consistency
    confidence_threshold: float, # Ignored, for signature consistency
    stop_loss_pct: float,
    take_profit_pct: float, # New parameter
    max_drawdown_limit: float, # Ignored, for signature consistency
    success_rate_threshold: float, # Ignored, for signature consistency
    interval: str,
):
    logging.debug("--- DEBUG: Entered train_evolve (REGRESSION MODE) ---")
    X = data[feature_cols]
    y = data["y"] # y is now continuous return

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
            "entry_threshold": trial.params.get("entry_threshold"), # New param
            "params": json.dumps(trial.params)
        }
        file_exists = os.path.exists(summary_log_path)
        with open(summary_log_path, "a", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data_to_write.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data_to_write)

    def objective(trial):
        # We now use LGBMRegressor exclusively
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        model = lgb.LGBMRegressor(random_state=42, verbose=-1, **params)

        # New hyperparameter for regression strategy
        entry_threshold = trial.suggest_float("entry_threshold", 1e-5, 1e-2, log=True)
        take_profit_pct = trial.suggest_float("take_profit_pct", 0.005, 0.08)

        tscv = TimeSeriesSplit(n_splits=5)
        trading_periods_per_year = {"1H": 252 * 24}.get(interval, 252)
        annualization_factor = np.sqrt(trading_periods_per_year)

        all_returns_series = []
        all_success_rates = []
        all_trade_counts = []
        all_fold_sharpes = []

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            model.fit(X_train, y_train)
            predicted_returns = pd.Series(model.predict(X_test), index=X_test.index)

            # Call the new regression backtester
            _total_ret, _max_dd, success_rate, trade_count, returns_series = run_backtest_regression(
                predicted_returns=predicted_returns,
                test_data=data.loc[X_test.index],
                entry_threshold=entry_threshold, 
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
            )
            
            if returns_series.std() != 0 and len(returns_series.loc[returns_series != 0]) > 5:
                fold_sharpe = (returns_series.mean() / returns_series.std()) * annualization_factor
                all_fold_sharpes.append(fold_sharpe if np.isfinite(fold_sharpe) else 0.0)
            else:
                all_fold_sharpes.append(0.0)

            all_returns_series.append(returns_series)
            all_trade_counts.append(trade_count)
            if trade_count > 0:
                all_success_rates.append(success_rate)

        final_returns = pd.concat(all_returns_series) if all_returns_series else pd.Series(dtype=float)

        # We keep the proven stability score as the objective to maximize
        if len(all_fold_sharpes) > 0 and np.any(all_fold_sharpes):
            mean_sharpe = np.mean(all_fold_sharpes)
            std_sharpe = np.std(all_fold_sharpes)
            stability_score = mean_sharpe - std_sharpe
        else:
            stability_score = -1.0

        if not np.isfinite(stability_score):
            stability_score = -1.0

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
            f"EntryThresh={entry_threshold:.4f}, " # Log new param
            f"Params={trial.params}"
        )

        trial.set_user_attr("fold_sharpes", all_fold_sharpes)
        trial.set_user_attr("overall_sharpe_for_log", overall_sharpe_for_log)

        return stability_score

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials, n_jobs=-1, callbacks=[save_trial_summary_callback])

    if study.best_trial is None:
        logging.warning("Optuna study finished without a best trial.")
        return None, None
    
    best_score = study.best_value
    best_params = study.best_params

    # --- Save Best Trial Details (with history) ---
    try:
        best_trial_path = os.path.join(out_dir, "best_trial_details.json")
        current_best_record = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_trials_in_run": n_trials,
            "trial_number": study.best_trial.number,
            "stability_score": study.best_trial.value,
            "overall_sharpe": study.best_trial.user_attrs.get("overall_sharpe_for_log"),
            "fold_sharpes": [round(s, 4) for s in study.best_trial.user_attrs.get("fold_sharpes", [])],
            "params": study.best_trial.params,
        }
        logging.info(f"Current run's best trial ({current_best_record['trial_number']}) resulted in stability score: {current_best_record['stability_score']:.4f}")
        history_data = {"all_time_best": None, "recent_trials": []}
        if os.path.exists(best_trial_path):
            try:
                with open(best_trial_path, "r", encoding="utf-8") as f:
                    loaded_data = json.load(f)
                    if isinstance(loaded_data, dict):
                        history_data["all_time_best"] = loaded_data.get("all_time_best")
                        history_data["recent_trials"] = loaded_data.get("recent_trials", [])
            except (json.JSONDecodeError, FileNotFoundError):
                logging.warning(f"Could not read or parse {best_trial_path}. A new file will be created.")
        history_data["recent_trials"].append(current_best_record)
        history_data["recent_trials"].sort(key=lambda x: x["timestamp"], reverse=True)
        history_data["recent_trials"] = history_data["recent_trials"][:4]
        all_time_best = history_data.get("all_time_best")
        is_new_all_time_best = False
        if all_time_best and isinstance(all_time_best, dict) and "stability_score" in all_time_best:
            if current_best_record["stability_score"] > all_time_best["stability_score"]:
                is_new_all_time_best = True
        else:
            is_new_all_time_best = True
        if is_new_all_time_best:
            history_data["all_time_best"] = current_best_record
            logging.info(f"🏆 New all-time best score achieved: {current_best_record['stability_score']:.4f}")
        else:
            if not history_data["all_time_best"] and history_data["recent_trials"]:
                 history_data["all_time_best"] = max(history_data["recent_trials"], key=lambda x: x['stability_score'])
        with open(best_trial_path, "w", encoding="utf-8") as f:
            json.dump(history_data, f, ensure_ascii=False, indent=4)
        logging.info(f"💾 Best trial history successfully updated in: {best_trial_path}")
    except Exception as e:
        logging.critical(f"CRITICAL: Failed to save best trial details. This is a bug. Error: {e}", exc_info=True)

    # --- Retraining and Saving ---
    logging.info("\n--- Retraining best model on full data and saving ---")
    os.makedirs(out_dir, exist_ok=True)
    params_for_training = best_params.copy()
    if 'entry_threshold' in params_for_training:
        del params_for_training['entry_threshold']
    
    final_model = lgb.LGBMRegressor(random_state=42, verbose=-1, **params_for_training)
    final_model.fit(X, y)
    logging.info(f"✅ Final model trained: {final_model}")

    model_path = os.path.join(out_dir, "best_model.pkl")
    dump(final_model, model_path)
    logging.info(f"💾 Model saved to: {model_path}")

    # --- Feature Importance Plot ---
    if hasattr(final_model, 'feature_importances_'):
        plot_path = os.path.join(out_dir, "feature_importance.png")
        feature_imp = pd.Series(final_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        plt.figure(figsize=(10, 8))
        plt.title("Feature Importance")
        feature_imp.plot(kind='barh')
        plt.tight_layout()
        plt.savefig(plot_path)
        logging.info(f"💾 Feature importance plot saved to: {plot_path}")

    meta = {
        "model_type": "lgb_regressor",
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
    
    logging.debug("--- DEBUG: Exiting train_evolve ---")
    return best_score, best_params
