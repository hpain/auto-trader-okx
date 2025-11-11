# research/evolve.py


import sys
import os



import json
import argparse
import logging
import hashlib

# This is fast and needs to be at the top for subsequent imports to work.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Main function to run the training and evolution process."""
    # Step 1: Setup parser
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--years", type=float, default=3.0, help="拉取多少年数据 (可以为小数, 如 0.5)")
    parser.add_argument("--models", type=str, default="lgb", help="使用的模型类型，逗号分隔")
    parser.add_argument("--trials", type=int, default=100, help="Optuna 搜索的 trial 数")
    parser.add_argument("--ignore-local", action="store_true", help="忽略本地CSV历史文件和特征缓存，直接重新生成")
    parser.add_argument("--top-k-features", type=int, default=20, help="在训练前预筛选出最重要的K个特征 (0表示禁用)")

    # --- Core Parameters ---
    parser.add_argument("--profit-threshold", type=float, default=0.005, help="每日最低收益目标")
    parser.add_argument("--confidence-threshold", type=float, default=0.95, help="执行交易的最低置信度")
    parser.add_argument("--stop-loss-pct", type=float, default=0.02, help="止损百分比 (例如 0.02 代表 2%%)")
    parser.add_argument("--take-profit-pct", type=float, default=0.05, help="止盈百分比 (例如 0.05 代表 5%%)")
    parser.add_argument("--max-drawdown", type=float, default=0.1, help="最大回撤限制 (例如 0.1 代表 10%%)")
    parser.add_argument("--success-rate-threshold", type=float, default=0.75, help="可接受的最低达标交易成功率")

    # Step 2: Parse arguments
    args = parser.parse_args()

    # Step 3: Heavy imports and setup
    from utils.logger import setup_script_logger
    setup_script_logger() # This will configure the root logger

    # --- Main logic begins ---
    logging.info("--- 策略参数 ---")
    logging.info(f"  拉取年数: {args.years}")
    logging.info(f"  收益目标: >= {args.profit_threshold:.2%}")
    logging.info(f"  止损线: {args.stop_loss_pct:.2%}")
    logging.info(f"  最大回撤限制: <= {args.max_drawdown:.2%}")
    logging.info(f"  预筛选特征数: {args.top_k_features if args.top_k_features > 0 else 'Disabled'}")
    logging.info("------------------")

    import pandas as pd
    import numpy as np
    from config import config
    from features.feature_engineering import generate_features, make_supervised
    from models.evolution import train_evolve

    from sklearn.feature_selection import SelectKBest, f_classif

    # --- Main logic continues ---
    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    symbol = config.get("trade", {}).get("symbol", "BTC-USDT")
    interval = config.get("trade", {}).get("interval", "1H")

    news_csv_path = os.path.join(config["paths"]["history_data_dir"], "sample_crypto_news.csv")

    feature_symbols = config.get("trading", {}).get("feature_pairs", [])
    primary_symbol = config.get("trading", {}).get("symbol", "BTC-USDT")
    
    # Ensure primary_symbol is in feature_symbols if it's not already there,
    # or handle it separately if it's always the main target.
    # For now, let's assume primary_symbol is the main one, and feature_symbols are additional.
    
    all_symbols_to_fetch = list(set([primary_symbol] + feature_symbols)) # Use set to avoid duplicates

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

        async def initialize_exchange_async():
            return await ExchangeFactory.create_exchange('aggregated', sandbox=False)
        
        exchange_instance = asyncio.run(initialize_exchange_async())
        
        # Dictionary to hold all fetched dataframes
        all_dfs = {}

        for sym in all_symbols_to_fetch:
            logging.info(f"Fetching historical data for {sym}...")
            df_current_symbol = exchange_instance.fetch_historical_data(symbol=sym, timeframe=interval, years=args.years)
            
            if df_current_symbol is None or df_current_symbol.empty:
                logging.warning(f"WARN: Price data for {sym} is empty after fetching. Skipping this symbol.")
                continue
            
            all_dfs[sym] = df_current_symbol
        
        if not all_dfs:
            logging.error("FAIL: No price data available after fetching all symbols. Aborting.")
            return

        # The primary symbol's data will be passed as the main df to generate_features
        # Other symbols' data will be passed as a dictionary
        dfp = all_dfs.get(primary_symbol)
        if dfp is None:
            logging.error(f"FAIL: Primary symbol {primary_symbol} data not found. Aborting.")
            return

        # Remove primary_symbol from feature_dfs to avoid redundant processing
        feature_dfs = {s: df for s, df in all_dfs.items() if s != primary_symbol}

        # --- Fetch Derivatives Data ---
        derivatives_dfs = {}
        fetch_funding_rates_flag = config.get("derivatives_data", {}).get("fetch_funding_rates", False)
        fetch_open_interest_flag = config.get("derivatives_data", {}).get("fetch_open_interest", False)

        # Calculate 'since' timestamp for historical derivatives data
        from datetime import datetime, timedelta
        since_dt = datetime.utcnow() - timedelta(days=args.years * 365.25)
        since_ms = int(since_dt.timestamp() * 1000)

        if fetch_funding_rates_flag:
            # Fetch funding rates for the primary symbol
            funding_rates_df = exchange_instance.fetch_funding_rates(symbol=primary_symbol, timeframe=interval, since=since_ms)
            if not funding_rates_df.empty:
                derivatives_dfs['funding_rates'] = funding_rates_df
                logging.info(f"Fetched {len(funding_rates_df)} funding rates for {primary_symbol}.")
            else:
                logging.warning(f"No funding rates fetched for {primary_symbol}.")
        else:
            logging.info("Skipping funding rates fetching as per configuration.")

        if fetch_open_interest_flag:
            # Fetch open interest for the primary symbol
            open_interest_df = exchange_instance.fetch_open_interest(symbol=primary_symbol, timeframe=interval, since=since_ms)
            if not open_interest_df.empty:
                derivatives_dfs['open_interest'] = open_interest_df
                logging.info(f"Fetched {len(open_interest_df)} open interest data for {primary_symbol}.")
            else:
                logging.warning(f"No open interest fetched for {primary_symbol}.")
        else:
            logging.info("Skipping open interest fetching as per configuration.")

        # --- Fetch On-Chain Data ---
        from data.onchain import get_onchain_client
        onchain_dfs = {}
        onchain_config = config.get("onchain_data", {})
        if onchain_config.get("enabled", False):
            logging.info("Fetching on-chain data...")
            try:
                # The config object is passed to the factory
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
                    # For Glassnode, the identifier is 'path'
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
                        df = client.get_data(query_id=str(glassnode_path)) # Glassnode client uses get_data with path

                    if not df.empty:
                            onchain_dfs[name] = df
                            logging.info(f"Successfully fetched {len(df)} rows for {name}.")
                    elif name:
                        logging.warning(f"No data returned for on-chain item: {name}")
            
            except Exception as e:
                logging.error(f"Failed to fetch on-chain data: {e}", exc_info=True)
        else:
            logging.info("Skipping on-chain data fetching as per configuration.")

        # Modify generate_features to accept multiple dataframes and derivatives data
        dfm = generate_features(dfp, news_csv_path=news_csv_path, feature_dfs=feature_dfs, derivatives_dfs=derivatives_dfs, onchain_dfs=onchain_dfs)
        dfm.to_parquet(feature_cache_path)
        logging.info(f"SAVE CACHE: Features saved to {feature_cache_path}")

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
            logging.info(f"--- Running feature pre-selection: Selecting top {args.top_k_features} features from {len(feature_cols)} ---")
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

    # --- Start Model Training ---
    logging.debug("--- DEBUG: Calling train_evolve ---")
    best_score, best_params = train_evolve(
        data=data,
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
    )
    logging.debug("--- DEBUG: Returned from train_evolve ---")
    
    if best_score is None and best_params is None:
        logging.warning("Training finished without producing a valid model. Please check logs for details.")
    else:
        logging.info(f"OK 训练完成：best score={best_score:.4f}")
        logging.info(f"最佳参数： {json.dumps(best_params, ensure_ascii=False, indent=2)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("--- SCRIPT FAILED WITH AN UNHANDLED EXCEPTION ---")
        print(traceback.format_exc())
        print("-------------------------------------------------")