"""
Model Performance Diagnostic Script

This script helps understand why the model's stability score is low.
It runs a quick training session and logs detailed diagnostics.
"""

import pandas as pd
import numpy as np
import sys
import os
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def diagnose_triple_barrier_labels(df: pd.DataFrame, tp: float, sl: float, timeout: int):
    """Analyze Triple Barrier labeling outcomes."""
    from features.feature_engineering import apply_triple_barrier
    
    logger.info(f"\n{'='*60}")
    logger.info("TRIPLE BARRIER DIAGNOSIS")
    logger.info(f"{'='*60}")
    logger.info(f"Parameters: TP={tp:.3%}, SL={sl:.3%}, Timeout={timeout} bars")
    
    labeled = apply_triple_barrier(df.copy(), tp=tp, sl=sl, timeout=timeout)
    
    if 'y' not in labeled.columns:
        logger.error("No 'y' column generated!")
        return
    
    # Label distribution
    label_counts = labeled['y'].value_counts()
    total = len(labeled.dropna(subset=['y']))
    
    logger.info(f"\nLabel Distribution:")
    for label, count in sorted(label_counts.items()):
        pct = count / total * 100
        logger.info(f"  Label {int(label)}: {count:,} ({pct:.1f}%)")
    
    # Imbalance ratio
    if len(label_counts) >= 2:
        imbalance = label_counts.max() / label_counts.min()
        logger.info(f"\nClass Imbalance Ratio: {imbalance:.2f}")
        if imbalance > 3:
            logger.warning("WARNING: High class imbalance may hurt model performance!")
    
    return labeled


def diagnose_feature_predictiveness(X: pd.DataFrame, y: pd.Series):
    """Check if features have any correlation with target."""
    from scipy.stats import spearmanr
    
    logger.info(f"\n{'='*60}")
    logger.info("FEATURE PREDICTIVENESS DIAGNOSIS")
    logger.info(f"{'='*60}")
    
    # Calculate Spearman correlation for each feature
    correlations = []
    for col in X.columns:
        valid = X[col].notna() & y.notna()
        if valid.sum() < 100:
            continue
        try:
            corr, pval = spearmanr(X.loc[valid, col], y.loc[valid])
            if np.isfinite(corr):
                correlations.append({
                    'feature': col,
                    'correlation': corr,
                    'abs_correlation': abs(corr),
                    'p_value': pval
                })
        except:
            pass
    
    corr_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)
    
    # Top predictive features
    logger.info(f"\nTop 10 Most Predictive Features:")
    for _, row in corr_df.head(10).iterrows():
        sig = "*" if row['p_value'] < 0.05 else ""
        logger.info(f"  {row['feature']}: r={row['correlation']:.4f}{sig}")
    
    # Summary stats
    avg_abs_corr = corr_df['abs_correlation'].mean()
    max_abs_corr = corr_df['abs_correlation'].max()
    
    logger.info(f"\nCorrelation Summary:")
    logger.info(f"  Average |correlation|: {avg_abs_corr:.4f}")
    logger.info(f"  Max |correlation|: {max_abs_corr:.4f}")
    
    if max_abs_corr < 0.05:
        logger.warning("WARNING: No features have significant correlation with target!")
        logger.warning("This explains the low model performance.")
    
    return corr_df


def diagnose_model_predictions(model, X_test: pd.DataFrame, y_test: pd.Series):
    """Analyze model prediction distribution."""
    logger.info(f"\n{'='*60}")
    logger.info("MODEL PREDICTION DIAGNOSIS")
    logger.info(f"{'='*60}")
    
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    # Prediction distribution
    pred_counts = pd.Series(predictions).value_counts()
    logger.info(f"\nPrediction Distribution:")
    for pred, count in sorted(pred_counts.items()):
        pct = count / len(predictions) * 100
        logger.info(f"  Pred {int(pred)}: {count} ({pct:.1f}%)")
    
    # Probability distribution
    logger.info(f"\nProbability Statistics:")
    logger.info(f"  Mean: {np.mean(probabilities):.4f}")
    logger.info(f"  Std: {np.std(probabilities):.4f}")
    logger.info(f"  Min: {np.min(probabilities):.4f}")
    logger.info(f"  Max: {np.max(probabilities):.4f}")
    
    # How many would pass different confidence thresholds
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7]
    logger.info(f"\nTrades at Different Confidence Thresholds:")
    for thresh in thresholds:
        trades = ((predictions == 1) & (probabilities >= thresh)).sum()
        logger.info(f"  Threshold {thresh:.2f}: {trades} trades ({trades/len(predictions)*100:.1f}%)")
    
    # Accuracy
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, zero_division=0)
    rec = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    
    logger.info(f"\nClassification Metrics:")
    logger.info(f"  Accuracy: {acc:.4f}")
    logger.info(f"  Precision: {prec:.4f}")
    logger.info(f"  Recall: {rec:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    
    return predictions, probabilities


def main():
    """Run full diagnostic."""
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from features.feature_engineering import generate_features, apply_triple_barrier
    
    logger.info("="*60)
    logger.info("MODEL PERFORMANCE DIAGNOSTIC")
    logger.info("="*60)
    
    # Load data
    csv_path = 'data/history/CLEAN_UNIVERSAL_2022_2026.csv'
    logger.info(f"\nLoading data from {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Use only BTC for faster diagnosis
    btc = df[df['symbol'] == 'BTCUSDT'].copy()
    btc['timestamp'] = pd.to_datetime(btc['timestamp'])
    btc = btc.set_index('timestamp').sort_index()
    
    logger.info(f"Data: {len(btc)} rows from {btc.index.min()} to {btc.index.max()}")
    
    # Generate features
    logger.info("\nGenerating features...")
    features = generate_features(btc)
    logger.info(f"Generated {len(features.columns)} features")
    
    # Triple Barrier labeling with different parameters
    tp_sl_configs = [
        (0.008, 0.005, 12),  # Current
        (0.015, 0.010, 24),  # Wider TP/SL, longer timeout
        (0.010, 0.007, 18),  # Medium
    ]
    
    for tp, sl, timeout in tp_sl_configs:
        labeled = diagnose_triple_barrier_labels(btc, tp, sl, timeout)
    
    # Use the medium config for further analysis
    tp, sl, timeout = 0.010, 0.007, 18
    labeled = apply_triple_barrier(btc.copy(), tp=tp, sl=sl, timeout=timeout)
    
    # Merge features with labels
    data = features.copy()
    data['y'] = labeled['y']
    data = data.dropna(subset=['y'])
    
    # Define feature columns
    non_feature_cols = ['y', 'open', 'high', 'low', 'close', 'volume', 'symbol', 
                        'future_high', 'future_low', 'future_close', 'future_ret']
    feature_cols = [c for c in data.columns if c not in non_feature_cols and data[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    X = data[feature_cols].dropna(axis=1, how='all')
    y = data['y']
    
    # Clean data
    X = X.replace([np.inf, -np.inf], np.nan)
    valid_mask = ~X.isna().any(axis=1) & ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    logger.info(f"\nFinal dataset: {len(X)} samples, {len(X.columns)} features")
    
    # Feature predictiveness
    corr_df = diagnose_feature_predictiveness(X, y)
    
    # Train a simple model
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING SIMPLE MODEL")
    logger.info(f"{'='*60}")
    
    # Train/test split (time-based)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.01,
        max_depth=6,
        num_leaves=32,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train, y_train.astype(int))
    
    # Analyze predictions
    predictions, probabilities = diagnose_model_predictions(model, X_test, y_test)
    
    # Feature importance
    logger.info(f"\n{'='*60}")
    logger.info("TOP FEATURE IMPORTANCES")
    logger.info(f"{'='*60}")
    
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in importance.head(15).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']}")
    
    # Final recommendations
    logger.info(f"\n{'='*60}")
    logger.info("RECOMMENDATIONS")
    logger.info(f"{'='*60}")
    
    if corr_df['abs_correlation'].max() < 0.03:
        logger.info("1. Features have very low correlation with target.")
        logger.info("   -> Consider different feature engineering approaches")
        logger.info("   -> Consider regression instead of classification")
    
    if y.mean() < 0.3 or y.mean() > 0.7:
        logger.info("2. Labels are imbalanced.")
        logger.info("   -> Adjust Triple Barrier parameters")
        logger.info("   -> Consider class weights in training")
    
    probs_std = np.std(probabilities)
    if probs_std < 0.1:
        logger.info("3. Model probabilities have low variance.")
        logger.info("   -> Model is not confident in its predictions")
        logger.info("   -> Consider simpler model or better features")


if __name__ == "__main__":
    main()
