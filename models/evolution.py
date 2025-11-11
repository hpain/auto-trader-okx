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
from sklearn.ensemble import RandomForestClassifier

# Import the original backtesting function for classification models
from utils.backtest import run_backtest

def train_evolve(
    data: pd.DataFrame,
    feature_cols: list,
    out_dir: str,
    n_trials: int,
    model_list: list, # Ignored, for signature consistency
    profit_threshold: float, # Ignored, for signature consistency
    confidence_threshold: float, # This will be optimized
    stop_loss_pct: float,
    take_profit_pct: float,
    max_drawdown_limit: float, # Ignored, for signature consistency
    success_rate_threshold: float, # Ignored, for signature consistency
    interval: str,
):
    logging.debug("--- DEBUG: Entered train_evolve (CLASSIFICATION MODE) ---")
    X = data[feature_cols]
    y = data["y"] # y is a classification label (0 or 1)

    summary_log_path = os.path.join(out_dir, "trials_summary.csv")
    if os.path.exists(summary_log_path):
        os.remove(summary_log_path)

    # --- Pre-calculate feature importances once to save time in trials ---
    logging.info("Calculating preliminary feature importances...")
    prelim_model = lgb.LGBMClassifier(random_state=42, n_estimators=100, verbose=-1)
    prelim_model.fit(X, y)
    importances = prelim_model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': feature_cols, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    logging.info(f"Top 10 features from preliminary ranking: {feature_importance_df['feature'].head(10).tolist()}")

    def save_trial_summary_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        """Callback to save trial summary to a CSV file."""
        # ensure JSON-serializable/native python types
        def _make_native(x):
            # recursively convert numpy types and lists/dicts to native python types
            if isinstance(x, dict):
                return {k: _make_native(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [_make_native(v) for v in x]
            try:
                if isinstance(x, (np.floating,)):
                    return float(x)
                if isinstance(x, (np.integer,)):
                    return int(x)
            except Exception:
                pass
            return x

        fold_vals = trial.user_attrs.get("fold_sharpes", []) or []
        fold_vals = [_make_native(round(float(s), 4)) for s in fold_vals]
        overall_sharpe_val = trial.user_attrs.get("overall_sharpe_for_log")
        try:
            overall_sharpe_val = float(overall_sharpe_val) if overall_sharpe_val is not None else 0.0
        except Exception:
            overall_sharpe_val = 0.0

        params_native = _make_native(trial.params or {})

        data_to_write = {
            "trial_number": int(trial.number),
            "stability_score": float(trial.value) if trial.value is not None else -1.0,
            "overall_sharpe": overall_sharpe_val,
            "fold_sharpes": json.dumps(fold_vals, ensure_ascii=False),
            "confidence_threshold": params_native.get("confidence_threshold"),
            "params": json.dumps(params_native, ensure_ascii=False)
        }
        file_exists = os.path.exists(summary_log_path)
        with open(summary_log_path, "a", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data_to_write.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data_to_write)

    def objective(trial):
        # --- Dynamic Feature Selection ---
        num_features = trial.suggest_int("num_features", 30, 150)
        selected_feature_names = feature_importance_df['feature'].head(num_features).tolist()
        X_subset = X[selected_feature_names]

        model_name = trial.suggest_categorical("model", ["lgb", "rf"])

        if model_name == "lgb":
            params = {
                "lgb_n_estimators": trial.suggest_int("lgb_n_estimators", 50, 400),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
                "num_leaves": trial.suggest_int("num_leaves", 20, 120),
            }
            params["n_estimators"] = params.pop("lgb_n_estimators")
            model = lgb.LGBMClassifier(random_state=42, verbose=-1, **params)
        
        elif model_name == "rf":
            params = {
                "rf_n_estimators": trial.suggest_int("rf_n_estimators", 50, 400),
                "max_depth": trial.suggest_int("max_depth", 10, 100),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
                "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            }
            params["n_estimators"] = params.pop("rf_n_estimators")
            model = RandomForestClassifier(random_state=42, verbose=0, n_jobs=-1, **params)

        confidence_thresh = trial.suggest_float("confidence_threshold", 0.30, 0.55)

        tscv = TimeSeriesSplit(n_splits=5)
        trading_periods_per_year = {"1H": 252 * 24}.get(interval, 252)
        annualization_factor = np.sqrt(trading_periods_per_year)

        all_returns_series = []
        all_success_rates = []
        all_trade_counts = []
        all_fold_sharpes = []

        for train_index, test_index in tscv.split(X_subset): # Use X_subset here
            X_train, X_test = X_subset.iloc[train_index], X_subset.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            y_train = y_train.astype(int)

            model.fit(X_train, y_train)
            predictions = pd.Series(model.predict(X_test), index=X_test.index)
            probabilities = model.predict_proba(X_test)[:, 1]

            _total_ret, _max_dd, success_rate, trade_count, returns_series = run_backtest(
                predictions=predictions,
                probabilities=probabilities,
                test_data=data.loc[X_test.index],
                confidence_threshold=confidence_thresh,
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
            f"ConfThresh={confidence_thresh:.4f}, "
            f"Params={trial.params}"
        )
        try:
            trial.set_user_attr("fold_sharpes", [float(s) for s in all_fold_sharpes])
            trial.set_user_attr("overall_sharpe_for_log", float(overall_sharpe_for_log))
            trial.set_user_attr("avg_success_rate", float(avg_success_rate))
        except Exception:
            pass

        return stability_score

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials, n_jobs=-1, callbacks=[save_trial_summary_callback])

    if study.best_trial is None:
        logging.warning("Optuna study finished without a best trial.")
        return None, None
    
    best_score = study.best_value
    best_params = study.best_params

    try:
        best_trial_path = os.path.join(out_dir, "best_trial_details.json")
        def _make_native_obj(x):
            if isinstance(x, dict):
                return {k: _make_native_obj(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [_make_native_obj(v) for v in x]
            if isinstance(x, (np.integer,)):
                return int(x)
            if isinstance(x, (np.floating,)):
                return float(x)
            return x

        current_best_record = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_trials_in_run": int(n_trials),
            "trial_number": int(study.best_trial.number),
            "stability_score": float(study.best_trial.value) if study.best_trial.value is not None else -1.0,
            "overall_sharpe": float(study.best_trial.user_attrs.get("overall_sharpe_for_log") or 0.0),
            "avg_success_rate": float(study.best_trial.user_attrs.get("avg_success_rate") or 0.0),
            "fold_sharpes": [_make_native_obj(round(s, 4)) for s in study.best_trial.user_attrs.get("fold_sharpes", [])],
            "params": _make_native_obj(study.best_trial.params or {}),
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
            logging.info(f"New all-time best score achieved: {current_best_record['stability_score']:.4f}")
        else:
            if not history_data["all_time_best"] and history_data["recent_trials"]:
                 history_data["all_time_best"] = max(history_data["recent_trials"], key=lambda x: x['stability_score'])
        with open(best_trial_path, "w", encoding="utf-8") as f:
            json.dump(history_data, f, ensure_ascii=False, indent=4)
        logging.info(f"Best trial history successfully updated in: {best_trial_path}")
    except Exception as e:
        logging.critical(f"CRITICAL: Failed to save best trial details. This is a bug. Error: {e}", exc_info=True)

    logging.info("\n--- Retraining all-time best model on full data and saving ---")
    
    params_for_training = None
    all_time_best_details = {}
    try:
        with open(best_trial_path, "r", encoding="utf-8") as f:
            final_best_data = json.load(f)
            all_time_best_details = final_best_data.get("all_time_best", {})
            if not all_time_best_details:
                raise ValueError("all_time_best key is missing or empty in JSON file.")
            
            params_for_training = _make_native_obj(all_time_best_details.get("params", {})).copy()
            logging.info(
                f"Loaded all-time best parameters for final model retraining. "
                f"From run on: {all_time_best_details.get('timestamp', 'N/A')}, "
                f"Trial number: {all_time_best_details.get('trial_number', 'N/A')}, "
                f"Score: {all_time_best_details.get('stability_score', 0):.4f}"
            )

    except (FileNotFoundError, KeyError, ValueError, Exception) as e:
        logging.critical(f"CRITICAL: Could not load all-time best parameters from {best_trial_path} to retrain model. Error: {e}. No model will be saved.", exc_info=True)
        return best_score, best_params

    os.makedirs(out_dir, exist_ok=True)
    
    # Get the feature list used by the all-time best model
    # This is a crucial change to ensure the final model uses the correct features
    num_features_from_best = all_time_best_details.get("params", {}).get("num_features", len(feature_cols))
    final_feature_cols = feature_importance_df['feature'].head(num_features_from_best).tolist()

    X_final = data[final_feature_cols]
    y_final = data["y"]

    best_model_type = all_time_best_details.get("params", {}).get("model", "lgb")

    params_to_remove = ['confidence_threshold', 'model', 'entry_threshold', 'take_profit_pct', 'num_features']
    
    if best_model_type == 'lgb':
        params_to_remove.extend(['rf_n_estimators', 'max_depth', 'min_samples_leaf', 'criterion'])
    elif best_model_type == 'rf':
        params_to_remove.extend(['lgb_n_estimators', 'learning_rate', 'num_leaves'])

    for p in params_to_remove:
        if p in params_for_training:
            del params_for_training[p]
    
    if 'lgb_n_estimators' in params_for_training:
        params_for_training['n_estimators'] = params_for_training.pop('lgb_n_estimators')
    if 'rf_n_estimators' in params_for_training:
        params_for_training['n_estimators'] = params_for_training.pop('rf_n_estimators')

    logging.info(f"Retraining final model of type: {best_model_type} with {len(final_feature_cols)} features.")
    if best_model_type == 'lgb':
        final_model = lgb.LGBMClassifier(random_state=42, verbose=-1, **params_for_training)
    elif best_model_type == 'rf':
        final_model = RandomForestClassifier(random_state=42, verbose=0, n_jobs=-1, **params_for_training)
    else:
        logging.error(f"Unsupported model type '{best_model_type}' for final training. Defaulting to LGBM.")
        final_model = lgb.LGBMClassifier(random_state=42, verbose=-1, **params_for_training)

    final_model.fit(X_final, y_final)
    logging.info(f"Final model trained with all-time best parameters: {params_for_training}")

    model_path = os.path.join(out_dir, "best_model.pkl")
    dump(final_model, model_path)
    logging.info(f"All-time best model saved to: {model_path}")

    if hasattr(final_model, 'feature_importances_'):
        plot_path = os.path.join(out_dir, "feature_importance.png")
        feature_imp = pd.Series(final_model.feature_importances_, index=final_feature_cols).sort_values(ascending=False)
        plt.figure(figsize=(10, 8))
        plt.title("Feature Importance (All-Time Best Model)")
        feature_imp.head(30).plot(kind='barh') # Plot top 30 for clarity
        plt.tight_layout()
        plt.savefig(plot_path)
        logging.info(f"Feature importance plot saved to: {plot_path}")

    meta = {
        "model_type": f"{best_model_type}_classifier",
        "feature_cols": final_feature_cols,
        "best_params": _make_native_obj(all_time_best_details.get("params", {})),
        "n_samples": int(len(data)),
        "best_score": float(all_time_best_details.get("stability_score", 0)),
        "training_timestamp": pd.Timestamp.utcnow().isoformat(),
        "source_run_timestamp": all_time_best_details.get("timestamp", "N/A")
    }
    metadata_path = os.path.join(out_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logging.info(f"Metadata for all-time best model saved to: {metadata_path}")
    
    logging.debug("--- DEBUG: Exiting train_evolve ---")
    return best_score, best_params
