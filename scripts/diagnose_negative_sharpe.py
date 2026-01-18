"""
Quick diagnostic for negative Sharpe issue.
Checks if model predictions are inversely correlated with actual outcomes.
"""
import pandas as pd
import numpy as np
import sys
import os
import logging
import lightgbm as lgb
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    from features.feature_engineering import generate_features, apply_triple_barrier
    
    print("="*60)
    print("NEGATIVE SHARPE DIAGNOSTIC")
    print("="*60)
    
    # Load data
    df = pd.read_csv('data/history/CLEAN_UNIVERSAL_2022_2026.csv')
    btc = df[df['symbol'] == 'BTCUSDT'].copy()
    btc['timestamp'] = pd.to_datetime(btc['timestamp'])
    btc = btc.set_index('timestamp').sort_index()
    
    print(f"\nData: {len(btc)} rows")
    
    # Generate features
    print("Generating features...")
    features = generate_features(btc)
    
    # Apply Triple Barrier
    print("Applying Triple Barrier...")
    labeled = apply_triple_barrier(btc.copy(), tp=0.008, sl=0.005, timeout=12)
    
    # Merge
    data = features.copy()
    data['y'] = labeled['y']
    data = data.dropna()
    
    # Label distribution
    label_dist = data['y'].value_counts()
    print(f"\nLabel distribution:")
    print(f"  0 (no profit): {label_dist.get(0, 0)} ({label_dist.get(0, 0)/len(data)*100:.1f}%)")
    print(f"  1 (profit): {label_dist.get(1, 0)} ({label_dist.get(1, 0)/len(data)*100:.1f}%)")
    
    # Prepare features
    non_feature = ['y', 'open', 'high', 'low', 'close', 'volume', 'symbol', 
                   'future_high', 'future_low', 'future_close', 'future_ret']
    feature_cols = [c for c in data.columns if c not in non_feature and data[c].dtype in ['float64', 'int64', 'float32']]
    
    X = data[feature_cols].replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how='any')
    y = data.loc[X.index, 'y']
    
    valid = ~X.isna().any(axis=1)
    X, y = X[valid], y[valid]
    
    print(f"\nFinal: {len(X)} samples, {len(X.columns)} features")
    
    # Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Train two models: with and without class_weight
    print("\n" + "="*60)
    print("COMPARISON: class_weight='balanced' vs None")
    print("="*60)
    
    for cw_name, cw in [("None", None), ("balanced", "balanced")]:
        model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.01,
            max_depth=6,
            num_leaves=32,
            class_weight=cw,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train.astype(int))
        
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        pred_1_ratio = preds.mean()
        accuracy = (preds == y_test).mean()
        
        # When model predicts 1, what's the actual outcome?
        pred_1_mask = preds == 1
        if pred_1_mask.sum() > 0:
            when_pred_1_actual = y_test[pred_1_mask].mean()
        else:
            when_pred_1_actual = 0
        
        print(f"\n[class_weight={cw_name}]")
        print(f"  Prediction=1 ratio: {pred_1_ratio*100:.1f}%")
        print(f"  Accuracy: {accuracy*100:.1f}%")
        print(f"  When predicting 1, actual success rate: {when_pred_1_actual*100:.1f}%")
        print(f"  Prob mean: {probs.mean():.3f}, std: {probs.std():.3f}")
    
    # Check if inverse correlation exists
    print("\n" + "="*60)
    print("INVERSE CORRELATION CHECK")
    print("="*60)
    
    # Train with balanced
    model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.01, max_depth=6, 
        class_weight="balanced", random_state=42, verbose=-1
    )
    model.fit(X_train, y_train.astype(int))
    probs = model.predict_proba(X_test)[:, 1]
    
    # Compare high confidence predictions
    for thresh in [0.50, 0.55, 0.60]:
        high_conf = probs >= thresh
        if high_conf.sum() > 10:
            actual_sr = y_test[high_conf].mean()
            print(f"  prob >= {thresh}: {high_conf.sum()} predictions, actual success: {actual_sr*100:.1f}%")
    
    # Inverse check
    for thresh in [0.50, 0.45, 0.40]:
        low_conf = probs <= thresh
        if low_conf.sum() > 10:
            actual_sr = y_test[low_conf].mean()
            print(f"  prob <= {thresh}: {low_conf.sum()} predictions, if we bet AGAINST: {(1-actual_sr)*100:.1f}% success")
    
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    print("""
If "bet AGAINST" success rate > 50%, consider:
1. Invert the prediction logic (predict 0 when model says 1)
2. Remove class_weight='balanced' 
3. Re-examine Triple Barrier parameters
""")


if __name__ == "__main__":
    main()
