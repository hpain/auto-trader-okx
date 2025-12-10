# research/improved_evolution.py
import sys
import os
import json
import argparse
import logging
import hashlib
import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
import csv
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# This is fast and needs to be at the top for subsequent imports to work.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Main function to run the improved training and evolution process."""
    # Step 1: Setup parser
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--years", type=float, default=3.0, help="拉取多少年数据 (可以为小数, 如 0.5)")
    parser.add_argument("--models", type=str, default="lgb", help="使用的模型类型，逗号分隔")
    parser.add_argument("--trials", type=int, default=100, help="Optuna 搜索的 trial 数")
    parser.add_argument("--ignore-local", action="store_true", help="忽略本地CSV历史文件和特征缓存，直接重新生成")
    parser.add_argument("--top-k-features", type=int, default=20, help="在训练前预筛选出最重要的K个特征 (0表示禁用)")
    parser.add_argument("--validation-frac", type=float, default=0.1, help="用于最终验证的数据比例")
    parser.add_argument("--min-confidence", type=float, default=0.55, help="最小置信度阈值")
    parser.add_argument("--max-confidence", type=float, default=0.70, help="最大置信度阈值")

    # --- Core Parameters ---
    parser.add_argument("--profit-threshold", type=float, default=0.005, help="每日最低收益目标")
    parser.add_argument("--confidence-threshold", type=float, default=0.95, help="执行交易的最低置信度")
    parser.add_argument("--stop-loss-pct", type=float, default=0.02, help="止损百分比 (例如 0.02 代表 2%)")
    parser.add_argument("--take-profit-pct", type=float, default=0.05, help="止盈百分比 (例如 0.05 代表 5%)")
    parser.add_argument("--max-drawdown", type=float, default=0.1, help="最大回撤限制 (例如 0.1 代表 10%)")
    parser.add_argument("--success-rate-threshold", type=float, default=0.75, help="可接受的最低达标交易成功率")

    # Step 2: Parse arguments
    args = parser.parse_args()

    # Step 3: Heavy imports and setup
    from utils.logger import setup_script_logger
    setup_script_logger() # This will configure the root logger

    from config import config
    from features.feature_engineering import generate_features, make_supervised

    # --- Main logic begins ---
    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    symbol = config.get("trade", {}).get("symbol", "BTC-USDT")
    interval = config.get("trade", {}).get("interval", "1H")

    news_csv_path = os.path.join(config["paths"]["history_data_dir"], "sample_crypto_news.csv")

    feature_symbols = config.get("trading", {}).get("feature_pairs", [])
    primary_symbol = config.get("trading", {}).get("symbol", "BTC-USDT")
    
    all_symbols_to_fetch = list(set([primary_symbol] + feature_symbols)) # Use set to avoid duplicates

    logging.info("--- 改进策略参数 ---")
    logging.info(f"  拉取年数: {args.years}")
    logging.info(f"  主交易对: {primary_symbol}")
    logging.info(f"  特征交易对: {feature_symbols}")
    logging.info(f"  收益目标: >= {args.profit_threshold:.2%}")
    logging.info(f"  止损线: {args.stop_loss_pct:.2%}")
    logging.info(f"  最大回撤限制: <= {args.max_drawdown:.2%}")
    logging.info(f"  预筛选特征数: {args.top_k_features if args.top_k_features > 0 else 'Disabled'}")
    logging.info(f"  验证数据比例: {args.validation_frac:.1%}")
    logging.info(f"  置信度优化范围: [{args.min_confidence:.2%}, {args.max_confidence:.2%}]")
    logging.info("------------------")

    # Feature Caching Logic
    feature_pairs_str = "-".join(sorted(feature_symbols))
    config_str = f"{primary_symbol}-{interval}-{args.years}-{news_csv_path}-{feature_pairs_str}"
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:10]
    cache_dir = config["paths"]["feature_cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)
    feature_cache_path = os.path.join(cache_dir, f"features_{config_hash}.parquet")

    if os.path.exists(feature_cache_path) and not args.ignore_local:
        logging.info(f"OK CACHE: Found feature cache, loading from {feature_cache_path}")
        dfm = pd.read_parquet(feature_cache_path)
    else:
        logging.info("WAIT CACHE: No feature cache found or --ignore-local is set, running full data pipeline...")
        from exchange.factory import ExchangeFactory
        import asyncio

        async def fetch_all_data_async():
            """Async function to fetch all data from exchange."""
            exchange_instance = await ExchangeFactory.create_exchange('aggregated', sandbox=False)
            
            try:
                # Dictionary to hold all fetched dataframes
                all_dfs = {}

                for sym in all_symbols_to_fetch:
                    logging.info(f"Fetching historical data for {sym}...")
                    df_current_symbol = await exchange_instance.fetch_historical_data(symbol=sym, timeframe=interval, years=args.years)
                    
                    if df_current_symbol is None or df_current_symbol.empty:
                        logging.warning(f"WARN: Price data for {sym} is empty after fetching. Skipping this symbol.")
                        continue
                    
                    all_dfs[sym] = df_current_symbol
                
                if not all_dfs:
                    logging.error("FAIL: No price data available after fetching all symbols. Aborting.")
                    return None, None, None, None

                dfp = all_dfs.get(primary_symbol)
                if dfp is None:
                    logging.error(f"FAIL: Primary symbol {primary_symbol} data not found. Aborting.")
                    return None, None, None, None

                feature_dfs = {s: df for s, df in all_dfs.items() if s != primary_symbol}

                # --- Fetch Derivatives Data ---
                derivatives_dfs = {}
                fetch_funding_rates_flag = config.get("derivatives_data", {}).get("fetch_funding_rates", False)
                fetch_open_interest_flag = config.get("derivatives_data", {}).get("fetch_open_interest", False)

                from datetime import datetime, timedelta
                since_dt = datetime.utcnow() - timedelta(days=args.years * 365.25)
                since_ms = int(since_dt.timestamp() * 1000)

                if fetch_funding_rates_flag:
                    try:
                        funding_rates_df = await exchange_instance.fetch_funding_rates(symbol=primary_symbol, timeframe=interval, since=since_ms)
                        if funding_rates_df is not None and not funding_rates_df.empty:
                            derivatives_dfs['funding_rates'] = funding_rates_df
                            logging.info(f"Fetched {len(funding_rates_df)} funding rates for {primary_symbol}.")
                        else:
                            logging.warning(f"No funding rates fetched for {primary_symbol}.")
                    except Exception as e:
                        logging.warning(f"Failed to fetch funding rates: {e}")
                else:
                    logging.info("Skipping funding rates fetching as per configuration.")

                if fetch_open_interest_flag:
                    try:
                        open_interest_df = await exchange_instance.fetch_open_interest(symbol=primary_symbol, timeframe=interval, since=since_ms)
                        if open_interest_df is not None and not open_interest_df.empty:
                            derivatives_dfs['open_interest'] = open_interest_df
                            logging.info(f"Fetched {len(open_interest_df)} open interest data for {primary_symbol}.")
                        else:
                            logging.warning(f"No open interest fetched for {primary_symbol}.")
                    except Exception as e:
                        logging.warning(f"Failed to fetch open interest: {e}")
                else:
                    logging.info("Skipping open interest fetching as per configuration.")
                
                return dfp, feature_dfs, derivatives_dfs, all_dfs
            finally:
                # Properly close exchange connections
                if hasattr(exchange_instance, 'close'):
                    await exchange_instance.close()
        
        # Run the async function
        dfp, feature_dfs, derivatives_dfs, all_dfs = asyncio.run(fetch_all_data_async())
        
        if dfp is None:
            return

        # --- Fetch On-Chain Data ---
        from data.onchain import get_onchain_client
        onchain_dfs = {}
        onchain_config = config.get("onchain_data", {})
        if onchain_config.get("enabled", False):
            logging.info("Fetching on-chain data...")
            try:
                client = get_onchain_client(config)
                provider = onchain_config.get('provider', 'dune')
                
                queries_to_fetch = []
                if provider == 'dune':
                    queries_to_fetch = onchain_config.get('dune', {}).get('queries_to_fetch', [])
                elif provider == 'glassnode':
                    queries_to_fetch = onchain_config.get('glassnode', {}).get('metrics_to_fetch', [])

                for item in queries_to_fetch:
                    name = item.get('name')
                    query_id = item.get('query_id')
                    raw_sql = item.get('sql')
                    glassnode_path = item.get('path')
                    
                    df = pd.DataFrame()
                    if name and query_id: # Dune by query_id
                        logging.info(f"Fetching Dune data by ID: {name} (Query ID: {query_id})")
                        df = client.get_data(query_id=str(query_id))
                    elif name and raw_sql and hasattr(client, 'get_data_from_sql'): # Dune by raw SQL
                        logging.info(f"Fetching Dune data by raw SQL: {name}")
                        df = client.get_data_from_sql(sql_query=raw_sql)
                    elif name and glassnode_path: # Glassnode by path
                        logging.info(f"Fetching Glassnode data: {name} (Path: {glassnode_path})")
                        df = client.get_data(query_id=str(glassnode_path))

                    if not df.empty:
                            onchain_dfs[name] = df
                            logging.info(f"Successfully fetched {len(df)} rows for {name}.")
                    elif name:
                        logging.warning(f"No data returned for on-chain item: {name}")
            
            except Exception as e:
                logging.error(f"Failed to fetch on-chain data: {e}", exc_info=True)
        else:
            logging.info("Skipping on-chain data fetching as per configuration.")


        # Generate features using all available data
        dfm = generate_features(dfp, news_csv_path=news_csv_path, feature_dfs=feature_dfs, derivatives_dfs=derivatives_dfs, onchain_dfs=onchain_dfs)
        
        dfm.to_parquet(feature_cache_path)
        logging.info(f"SAVE CACHE: Features saved to {feature_cache_path}")

    # 确保返回的 DataFrame 有一个 DatetimeIndex
    if not isinstance(dfm.index, pd.DatetimeIndex):
        dfm = dfm.set_index('timestamp')

    # Build supervised learning data
    data = make_supervised(dfm, horizon=1, threshold=args.profit_threshold)
    
    non_feature_cols = ["ts", "dt", "y", "future_high", "future_low", "future_close", "future_ret", "date", "timestamp", "vol_ccy", "vol_ccy_quote", "confirm"]
    feature_cols = [c for c in data.columns if c not in non_feature_cols]
    
    # --- Data Cleaning ---
    # Replace infinite values with NaN, then drop all rows with any NaN in features or target.
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(subset=feature_cols + ["y"], inplace=True)
    
    if len(data) < 500:
        logging.warning(f"WARN 样本太少（{len(data)}）无法有效训练。")
        return

    # --- Feature Pre-selection ---
    try:
        if args.top_k_features and args.top_k_features > 0 and args.top_k_features < len(feature_cols):
            logging.info(f"--- Running feature pre-selection: Selecting top {args.top_k_features} features from {len(feature_cols)} --- ")
            X = data[feature_cols]
            y = data["y"]

            # --- NEW: Filter out constant features ---
            # Calculate variance for each feature. Dropna first to avoid NaN issues in variance calculation.
            # We only care about features that are constant across the *entire* dataset for pre-selection.
            variances = X.var()
            # Identify features with zero variance (constant features)
            constant_features = variances[variances == 0].index.tolist()
            
            if constant_features:
                logging.warning(f"WARN: Removing constant features before selection: {constant_features}")
                X = X.drop(columns=constant_features)
                # Update feature_cols for the selector
                feature_cols = [col for col in feature_cols if col not in constant_features]
                
                if len(feature_cols) < args.top_k_features:
                    logging.warning(f"WARN: After removing constant features, only {len(feature_cols)} remain. Adjusting top-k to {len(feature_cols)}.")
                    args.top_k_features = len(feature_cols)
                if len(feature_cols) == 0:
                    logging.error("CRITICAL: No features left after removing constant ones. Cannot perform feature selection.")
                    return # Exit gracefully if no features remain
            
            selector = SelectKBest(f_classif, k=args.top_k_features)
            selector.fit(X, y)
            
            selected_mask = selector.get_support()
            new_feature_cols = X.columns[selected_mask]
            
            logging.info(f"Selected features: {new_feature_cols.tolist()}")
            feature_cols = new_feature_cols.tolist() # Update feature_cols to the selected subset
    except Exception as e:
        logging.error(f"CRITICAL: Feature pre-selection failed: {e}", exc_info=True)
        return # Exit gracefully if selection fails
    # --- End of Feature Pre-selection ---

    # Split data: use recent portion for final validation
    val_size = max(1, int(len(data) * args.validation_frac))
    train_data = data.iloc[:-val_size]
    val_data = data.iloc[-val_size:]
    
    logging.info(f"Training on {len(train_data)} samples, validating on {len(val_data)} samples")

    # Train with improved cross-validation
    logging.debug("--- DEBUG: Calling improved train_evolve ---")
    best_score, best_params = train_evolve(
        train_data=train_data,
        val_data=val_data,
        feature_cols=feature_cols,
        out_dir=config["paths"]["model_dir"],
        n_trials=args.trials,
        model_list=model_list,
        profit_threshold=args.profit_threshold,
        confidence_threshold=args.confidence_threshold,
        stop_loss_pct=args.stop_loss_pct,
        max_drawdown_limit=args.max_drawdown,
        success_rate_threshold=args.success_rate_threshold,
        take_profit_pct=args.take_profit_pct,
        interval=interval,
        min_confidence=args.min_confidence,
        max_confidence=args.max_confidence
    )
    logging.debug("--- DEBUG: Returned from train_evolve ---")
    
    if best_score is None and best_params is None:
        logging.warning("Training finished without producing a valid model. Please check logs for details.")
    else:
        logging.info(f"OK 训练完成：best score={best_score:.4f}")
        logging.info(f"最佳参数： {json.dumps(best_params, ensure_ascii=False, indent=2)}")

def train_evolve(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
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
    min_confidence: float = 0.55, # New: minimum confidence threshold
    max_confidence: float = 0.70   # New: maximum confidence threshold
):
    logging.debug("--- DEBUG: Entered train_evolve (IMPROVED CLASSIFICATION MODE) ---")
    X_train = train_data[feature_cols]
    y_train = train_data["y"] # y is a classification label (0 or 1)

    # Final validation data
    X_val = val_data[feature_cols]
    y_val = val_data["y"]

    summary_log_path = os.path.join(out_dir, "improved_trials_summary.csv")
    if os.path.exists(summary_log_path):
        os.remove(summary_log_path)

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
        from utils.backtest import run_backtest  # 关键修复：在此处导入回测函数
        
        # Use more conservative model parameters to prevent overfitting
        model_name = trial.suggest_categorical("model", ["lgb"])  # Focus on LGB for now

        if model_name == "lgb":
            # More conservative parameters to prevent overfitting
            params = {
                "lgb_n_estimators": trial.suggest_int("lgb_n_estimators", 50, 300),  # Reduced max
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10),   # Reduced max
                "num_leaves": trial.suggest_int("num_leaves", 10, 80),               # Reduced max
                "max_depth": trial.suggest_int("max_depth", 3, 8),                  # Added max depth for regularization
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 50), # Added min_child_samples for regularization
                "subsample": trial.suggest_float("subsample", 0.7, 0.95),           # Added subsample for regularization
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9), # Added colsample for regularization
            }
            params["n_estimators"] = params.pop("lgb_n_estimators")
            
            # Add random state for reproducibility
            model = lgb.LGBMClassifier(random_state=42, verbose=-1, **params)

        # Use higher confidence threshold range
        confidence_thresh = trial.suggest_float("confidence_threshold", min_confidence, max_confidence)

        # Use TimeSeriesSplit with fewer splits to reduce variance in CV scores
        tscv = TimeSeriesSplit(n_splits=3)  # Reduced from 5 to reduce variance
        trading_periods_per_year = {"1H": 252 * 24}.get(interval, 252)
        annualization_factor = np.sqrt(trading_periods_per_year)

        all_returns_series = []
        all_success_rates = []
        all_trade_counts = []
        all_fold_sharpes = []

        for train_index, test_index in tscv.split(X_train):
            X_fold_train, X_fold_test = X_train.iloc[train_index], X_train.iloc[test_index]
            y_fold_train, y_fold_test = y_train.iloc[train_index], y_train.iloc[test_index]
            
            y_fold_train = y_fold_train.astype(int)

            model.fit(X_fold_train, y_fold_train)
            predictions = pd.Series(model.predict(X_fold_test), index=X_fold_test.index)
            probabilities = model.predict_proba(X_fold_test)[:, 1]

            _total_ret, _max_dd, success_rate, trade_count, returns_series = run_backtest(
                predictions=predictions,
                probabilities=probabilities,
                test_data=train_data.loc[X_fold_test.index],
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
            # Use a more balanced approach: still reward stability but don't overly penalize it
            stability_score = mean_sharpe - (0.5 * std_sharpe)  # Reduced penalty on std
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

    # Use pruner to avoid wasting time on clearly bad trials
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5))
    study.optimize(objective, n_trials=n_trials, n_jobs=1, callbacks=[save_trial_summary_callback])  # Reduced n_jobs to 1 for consistency

    if study.best_trial is None:
        logging.warning("Optuna study finished without a best trial.")
        return None, None
    
    best_score = study.best_value
    best_params = study.best_params

    # Final validation with best parameters
    logging.info("Performing final validation on holdout data...")
    
    # Import the backtest function
    from utils.backtest import run_backtest
    
    best_model_type = study.best_params.get("model", "lgb")
    if best_model_type == "lgb":
        params_for_final = {
            "n_estimators": study.best_params.get("lgb_n_estimators", 100),
            "learning_rate": study.best_params.get("learning_rate", 0.1),
            "num_leaves": study.best_params.get("num_leaves", 31),
            "max_depth": study.best_params.get("max_depth", 6),
            "min_child_samples": study.best_params.get("min_child_samples", 20),
            "subsample": study.best_params.get("subsample", 0.9),
            "colsample_bytree": study.best_params.get("colsample_bytree", 0.8),
        }
        
        final_model = lgb.LGBMClassifier(random_state=42, verbose=-1, **params_for_final)
    else:
        logging.error(f"Unsupported model type '{best_model_type}' for final training. Using LGBM.")
        final_model = lgb.LGBMClassifier(random_state=42, verbose=-1, n_estimators=100)

    # Train on full training data
    final_model.fit(X_train, y_train.astype(int))
    
    # Validate on holdout data
    val_predictions = pd.Series(final_model.predict(X_val), index=X_val.index)
    val_probabilities = final_model.predict_proba(X_val)[:, 1]
    
    final_return, final_max_dd, final_success_rate, final_trade_count, final_returns_series = run_backtest(
        predictions=val_predictions,
        probabilities=val_probabilities,
        test_data=val_data,
        confidence_threshold=study.best_params.get("confidence_threshold", 0.65),
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )
    
    # Calculate validation Sharpe
    if final_returns_series.std() != 0 and len(final_returns_series.loc[final_returns_series != 0]) >= 10:
        final_sharpe = (final_returns_series.mean() / final_returns_series.std()) * np.sqrt(trading_periods_per_year)
    else:
        final_sharpe = 0.0
    
    logging.info(f"Final validation results: Return={final_return:.4f}, Sharpe={final_sharpe:.2f}, SuccessRate={final_success_rate:.2%}, Trades={final_trade_count}")

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
            "val_return": float(final_return),
            "val_sharpe": float(final_sharpe),
            "val_success_rate": float(final_success_rate),
            "val_trade_count": int(final_trade_count),
            "params": _make_native_obj(study.best_trial.params or {}),
        }
        logging.info(f"Current run's best trial ({current_best_record['trial_number']}) resulted in stability score: {current_best_record['stability_score']:.4f}")
        logging.info(f"Validation metrics - Return: {current_best_record['val_return']:.4f}, Sharpe: {current_best_record['val_sharpe']:.2f}")
        
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
    
    # Save the final model on full dataset
    from joblib import dump
    import matplotlib.pyplot as plt
    
    os.makedirs(out_dir, exist_ok=True)
    
    model_path = os.path.join(out_dir, "improved_best_model.pkl")
    dump(final_model, model_path)
    logging.info(f"Improved all-time best model saved to: {model_path}")

    if hasattr(final_model, 'feature_importances_'):
        plot_path = os.path.join(out_dir, "improved_feature_importance.png")
        feature_imp = pd.Series(final_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        plt.figure(figsize=(10, 8))
        plt.title("Feature Importance (Improved Model)")
        feature_imp.plot(kind='barh')
        plt.tight_layout()
        plt.savefig(plot_path)
        logging.info(f"Feature importance plot saved to: {plot_path}")

    meta = {
        "model_type": f"{best_model_type}_classifier_improved",
        "feature_cols": feature_cols,
        "best_params": _make_native_obj(current_best_record.get("params", {})),
        "n_samples": int(len(train_data)),
        "best_score": float(current_best_record.get("stability_score", 0)),
        "val_return": float(current_best_record.get("val_return", 0)),
        "val_sharpe": float(current_best_record.get("val_sharpe", 0)),
        "training_timestamp": pd.Timestamp.utcnow().isoformat(),
        "source_run_timestamp": current_best_record.get("timestamp", "N/A")
    }
    metadata_path = os.path.join(out_dir, "improved_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logging.info(f"Metadata for improved model saved to: {metadata_path}")
    
    logging.debug("--- DEBUG: Exiting train_evolve ---")
    return best_score, best_params

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("--- SCRIPT FAILED WITH AN UNHANDLED EXCEPTION ---")
        print(traceback.format_exc())
        print("-------------------------------------------------")