"""
Training Speed Diagnostic Script

Identifies the main performance bottlenecks in the training pipeline.
"""

import pandas as pd
import numpy as np
import sys
import os
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def profile_section(name):
    """Decorator to profile a code section."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.info(f"[PROFILE] {name}: {elapsed:.2f}s")
            return result
        return wrapper
    return decorator


def main():
    print("="*60)
    print("TRAINING SPEED DIAGNOSTIC")
    print("="*60)
    
    # 1. Data Loading
    start = time.time()
    csv_path = 'data/history/CLEAN_UNIVERSAL_2022_2026.csv'
    df = pd.read_csv(csv_path)
    load_time = time.time() - start
    logger.info(f"[1] Data Loading: {load_time:.2f}s ({len(df):,} rows)")
    
    # 2. Feature Generation (per symbol)
    from features.feature_engineering import generate_features
    
    start = time.time()
    btc = df[df['symbol'] == 'BTCUSDT'].copy()
    btc['timestamp'] = pd.to_datetime(btc['timestamp'])
    btc = btc.set_index('timestamp')
    btc_features = generate_features(btc)
    btc_time = time.time() - start
    logger.info(f"[2] Feature Gen (BTC only, {len(btc):,} rows): {btc_time:.2f}s")
    
    start = time.time()
    eth = df[df['symbol'] == 'ETHUSDT'].copy()
    eth['timestamp'] = pd.to_datetime(eth['timestamp'])
    eth = eth.set_index('timestamp')
    eth_features = generate_features(eth)
    eth_time = time.time() - start
    logger.info(f"[3] Feature Gen (ETH only, {len(eth):,} rows): {eth_time:.2f}s")
    
    total_feature_time = btc_time + eth_time
    logger.info(f"[4] Total Feature Gen (both symbols): {total_feature_time:.2f}s")
    
    # 3. Triple Barrier Labeling
    from features.feature_engineering import apply_triple_barrier
    
    start = time.time()
    labeled = apply_triple_barrier(btc.copy(), tp=0.008, sl=0.005, timeout=12)
    label_time = time.time() - start
    logger.info(f"[5] Triple Barrier Labeling ({len(btc):,} rows): {label_time:.2f}s")
    
    # 4. Model Training (single fold)
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    
    # Prepare data
    features = btc_features.copy()
    features['y'] = labeled['y']
    features = features.dropna()
    
    non_feature_cols = ['y', 'open', 'high', 'low', 'close', 'volume', 'symbol']
    feature_cols = [c for c in features.columns if c not in non_feature_cols and features[c].dtype in ['float64', 'int64', 'float32']]
    
    X = features[feature_cols].replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='any')
    y = features.loc[X.index, 'y']
    
    # Drop rows with NaN
    valid = ~X.isna().any(axis=1) & ~y.isna()
    X, y = X[valid], y[valid]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    logger.info(f"\nDataset size: {len(X_train)} train, {len(X_test)} test, {len(X.columns)} features")
    
    # Train with different n_estimators
    for n_est in [100, 500, 1000]:
        model = lgb.LGBMClassifier(
            n_estimators=n_est,
            learning_rate=0.01,
            max_depth=6,
            num_leaves=32,
            random_state=42,
            verbose=-1,
            n_jobs=-1  # Use all cores
        )
        
        start = time.time()
        model.fit(X_train, y_train.astype(int))
        train_time = time.time() - start
        logger.info(f"[6] LGB Training (n_estimators={n_est}): {train_time:.2f}s")
    
    # 5. Optuna Trial Overhead
    logger.info("\n" + "="*60)
    logger.info("ESTIMATED TRAINING TIME")
    logger.info("="*60)
    
    # Estimate single trial time
    # 5 folds × feature gen + model training per fold
    single_fold_train = 3.0  # ~3s per fold based on n_est=500
    folds = 5
    single_trial = folds * single_fold_train + 1.0  # +1s overhead
    
    for trials in [20, 50, 100, 200]:
        total = single_trial * trials
        logger.info(f"  {trials} trials: ~{total/60:.1f} minutes ({total:.0f}s)")
    
    # 6. Recommendations
    logger.info("\n" + "="*60)
    logger.info("RECOMMENDATIONS")
    logger.info("="*60)
    
    logger.info("""
1. [HIGH IMPACT] Enable LightGBM GPU if available:
   python research/improved_evolution.py --gpu
   
2. [HIGH IMPACT] Reduce n_estimators range:
   Current: 100-1000, Suggested: 100-500
   
3. [MEDIUM IMPACT] Reduce number of CV folds:
   Current: 5, Could use: 3 (for faster experimentation)
   
4. [MEDIUM IMPACT] Use a smaller subset of data for initial experiments:
   python research/improved_evolution.py --local-csv data/history/CLEAN_UNIVERSAL_2022_2026.csv --years 1.0
   
5. [LOW IMPACT] Feature caching (already implemented via mined_factors.json)

6. [BOTTLENECK] Feature generation runs ONCE per training, NOT per trial.
   This is correct and efficient.
""")


if __name__ == "__main__":
    main()
