import sys
import os
import pandas as pd
import numpy as np
import logging
import pickle
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategies.transformer_strategy import TransformerStrategy, TimeSeriesTransformer
from features.feature_engineering import generate_features, apply_triple_barrier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Trainer")


def select_features(df, feature_cols, target_col, max_features=80):
    """
    Select the most informative features using statistical tests
    """
    if len(feature_cols) <= max_features:
        return feature_cols

    # Remove features with zero variance
    valid_features = []
    for col in feature_cols:
        if df[col].std() > 0:  # Only keep features with variance
            valid_features.append(col)

    if len(valid_features) <= max_features:
        return valid_features

    # Use SelectKBest to select top features based on ANOVA F-test
    selector = SelectKBest(score_func=f_classif, k=min(max_features, len(valid_features)))

    X = df[valid_features]
    y = df[target_col]

    # Handle any remaining NaN values
    X = X.fillna(X.mean())  # Fill with mean to avoid issues with SelectKBest

    try:
        X_selected = selector.fit_transform(X, y)
        selected_features = [valid_features[i] for i in selector.get_support(indices=True)]

        logger.info(f"Feature selection reduced from {len(valid_features)} to {len(selected_features)} features")
        return selected_features
    except Exception as e:
        logger.warning(f"Feature selection failed: {e}. Using first {max_features} features.")
        return valid_features[:max_features]


def train_main(args):
    # 1. Load Data
    data_path = args.data

    if not data_path:
        potential_files = [
            'data/history/CLEAN_UNIVERSAL_2022_2026.csv',
            'data/history/BTCUSDT_FULL_2020_2025.csv',
        ]
        for p in potential_files:
            if os.path.exists(p):
                data_path = p
                break

    if not data_path:
        logger.error("No data file found. Please provide --data argument.")
        return

    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Standardize columns
    df.columns = [c.lower() for c in df.columns]
    rename_map = {'ts': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}
    df.rename(columns=rename_map, inplace=True)

    if 'timestamp' in df.columns:
        if df['timestamp'].dtype == object:
            df['datetime'] = pd.to_datetime(df['timestamp'])
        else:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
    elif 'date' in df.columns:
        df.index = pd.to_datetime(df['date'])

    logger.info(f"Loaded {len(df)} rows.")

    # 2. Handle multi-symbol data
    if 'symbol' in df.columns:
        symbols = df['symbol'].unique()
        logger.info(f"Multi-symbol data: {symbols.tolist()}")
        # Use first symbol only (BTC)
        df = df[df['symbol'] == symbols[0]].copy()
        logger.info(f"Using {symbols[0]}: {len(df)} rows")

    # 3. Data Quality Improvements
    logger.info("Performing data quality improvements...")

    # Handle missing values in derived features
    # Forward fill for time series data
    for col in ['funding_rate', 'open_interest', 'open_interest_value']:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    # Handle other derived features
    for col in ['count_toptrader_long_short_ratio', 'sum_toptrader_long_short_ratio',
                'count_long_short_ratio', 'sum_taker_long_short_vol_ratio']:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    # Remove extreme outliers using IQR method
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['timestamp', 'datetime']]

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR  # Using 3*IQR for more tolerance
        upper_bound = Q3 + 3 * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    logger.info(f"After data cleaning: {len(df)} rows")

    # 4. Generate Features
    logger.info("Generating features...")
    df_features = generate_features(df)

    # 4. Triple Barrier Labeling (same as LGB for fair comparison)
    logger.info(f"Applying Triple Barrier: TP={args.tp}, SL={args.sl}, Timeout={args.timeout}")
    df_labeled = apply_triple_barrier(df_features, tp=args.tp, sl=args.sl, timeout=args.timeout)
    target_col = 'y'

    df_labeled.dropna(inplace=True)
    logger.info(f"Data after labeling: {len(df_labeled)}")

    # Label distribution
    label_counts = df_labeled[target_col].value_counts()
    logger.info(f"Labels: {label_counts.to_dict()}")

    # Calculate positive/negative class counts for balancing
    pos_count = label_counts.get(1, 0)
    neg_count = label_counts.get(0, 0)
    total_count = len(df_labeled)

    # Calculate pos_weight for balanced loss
    pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    logger.info(f"Label distribution - Pos: {pos_count} ({pos_count/total_count:.2%}), Neg: {neg_count} ({neg_count/total_count:.2%})")
    logger.info(f"Calculated pos_weight for loss: {pos_weight:.4f}")

    # Calculate baseline loss
    p_up = pos_count / total_count
    baseline = - (p_up * np.log(p_up + 1e-9) + (1-p_up) * np.log(1-p_up + 1e-9))
    logger.info(f"Baseline Loss: {baseline:.4f}")

    # 5. Split: 70% train, 10% val, 20% test
    train_end = int(len(df_labeled) * 0.7)
    val_end = int(len(df_labeled) * 0.8)

    train_df = df_labeled.iloc[:train_end]
    val_df = df_labeled.iloc[train_end:val_end]
    test_df = df_labeled.iloc[val_end:]

    logger.info(f"Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 7. Feature Selection with correlation analysis
    exclude = ['y', 'open', 'high', 'low', 'close', 'symbol'] + [c for c in df_labeled.columns if 'future' in c]
    feature_cols = [c for c in df_labeled.columns if c not in exclude and np.issubdtype(df_labeled[c].dtype, np.number)]
    logger.info(f"Initial features: {len(feature_cols)}")

    # Perform feature selection to reduce dimensionality
    feature_cols = select_features(train_df, feature_cols, target_col, max_features=80)
    logger.info(f"Selected features: {len(feature_cols)}")

    # 7. Scaling
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    os.makedirs('models', exist_ok=True)
    with open('models/scaler_transformer_tb.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    logger.info("Scaler saved")

    # 8. Initialize Optimized Transformer
    # Increase model capacity slightly to handle 135 features better
    strategy = TransformerStrategy(
        strategy_name="Transformer_Optimized",
        window_size=args.window,
        features=feature_cols,
        buy_threshold=0.6,
        sell_threshold=0.4
    )

    # Build model with increased capacity
    strategy.build_model(
        input_dim=len(feature_cols),
        d_model=args.d_model,  # Allow customization
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr  # Use configurable learning rate
    )

    # Use balanced loss function with calculated pos_weight
    import torch
    import torch.nn as nn
    strategy.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    strategy.scaler = scaler

    # 9. Prepare scaled data
    train_scaled = train_df.copy()
    train_scaled[feature_cols] = scaler.transform(train_df[feature_cols])

    val_scaled = val_df.copy()
    val_scaled[feature_cols] = scaler.transform(val_df[feature_cols])

    # 10. Train with validation
    logger.info(f"Training for {args.epochs} epochs...")

    from features.tensor_loader import create_lazy_loader
    import torch

    train_loader = create_lazy_loader(
        train_scaled[feature_cols],
        train_scaled[target_col],
        args.window,
        args.batch_size
    )
    val_loader = create_lazy_loader(
        val_scaled[feature_cols],
        val_scaled[target_col],
        args.window,
        args.batch_size
    )

    best_val_loss = float('inf')
    patience = args.patience  # Make patience configurable
    patience_counter = 0
    best_state = None

    # Add warmup scheduler
    from torch.optim.lr_scheduler import LambdaLR
    def warmup_lambda(current_step):
        warmup_steps = args.epochs * 0.1  # Warmup for first 10% of epochs
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    warmup_scheduler = LambdaLR(strategy.optimizer, lr_lambda=warmup_lambda)

    # Track metrics for better early stopping
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):
        # Training
        strategy.model.train()
        train_loss = 0
        train_batches = 0
        train_correct = 0
        train_total = 0
        train_tp = 0  # True positives
        train_fp = 0  # False positives
        train_fn = 0  # False negatives

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(strategy.device).float()
            batch_y = batch_y.to(strategy.device).float().unsqueeze(1)

            strategy.optimizer.zero_grad()
            outputs = strategy.model(batch_X)

            # Calculate loss
            loss = strategy.criterion(outputs, batch_y)
            loss.backward()

            # Adaptive gradient clipping based on loss magnitude
            grad_norm = torch.nn.utils.clip_grad_norm_(strategy.model.parameters(), max_norm=1.0)
            strategy.optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            # Calculate accuracy and other metrics
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_y.size(0)

            # Calculate TP, FP, FN for F1 score
            train_tp += ((preds == 1) & (batch_y == 1)).sum().item()
            train_fp += ((preds == 1) & (batch_y == 0)).sum().item()
            train_fn += ((preds == 0) & (batch_y == 1)).sum().item()

        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
        train_acc = train_correct / train_total if train_total > 0 else 0

        # Calculate train F1 score
        train_precision = train_tp / (train_tp + train_fp) if (train_tp + train_fp) > 0 else 0
        train_recall = train_tp / (train_tp + train_fn) if (train_tp + train_fn) > 0 else 0
        train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall) if (train_precision + train_recall) > 0 else 0

        # Validation
        strategy.model.eval()
        val_loss = 0
        val_batches = 0
        val_correct = 0
        val_total = 0
        val_tp = 0  # True positives
        val_fp = 0  # False positives
        val_fn = 0  # False negatives

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(strategy.device).float()
                batch_y = batch_y.to(strategy.device).float().unsqueeze(1)
                outputs = strategy.model(batch_X)

                # Calculate loss
                loss = strategy.criterion(outputs, batch_y)
                val_loss += loss.item()
                val_batches += 1

                # Calculate accuracy and other metrics
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)

                # Calculate TP, FP, FN for F1 score
                val_tp += ((preds == 1) & (batch_y == 1)).sum().item()
                val_fp += ((preds == 1) & (batch_y == 0)).sum().item()
                val_fn += ((preds == 0) & (batch_y == 1)).sum().item()

        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0

        # Calculate validation F1 score
        val_precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0
        val_recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0
        val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0

        # Update warmup scheduler first, then main scheduler
        if epoch < int(args.epochs * 0.1):  # Only apply warmup in first 10% of epochs
            warmup_scheduler.step()

        # LR Scheduler
        strategy.scheduler.step(avg_val_loss)
        current_lr = strategy.optimizer.param_groups[0]['lr']

        logger.info(f"Epoch {epoch+1}/{args.epochs} - Train: {avg_train_loss:.4f} (Acc: {train_acc:.4f}, F1: {train_f1:.4f}) - Val: {avg_val_loss:.4f} (Acc: {val_acc:.4f}, F1: {val_f1:.4f}) - LR: {current_lr:.6f}")

        # Early stopping based on validation F1 score (more appropriate for imbalanced data)
        # Also consider improvement in loss to avoid stopping too early on F1 fluctuations
        if val_f1 > best_val_f1 or (val_f1 >= best_val_f1 * 0.99 and avg_val_loss < best_val_loss):
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in strategy.model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch}, Best Val Loss: {best_val_loss:.4f}, Best Val Acc: {best_val_acc:.4f}, Best Val F1: {best_val_f1:.4f}")
                break

    # Restore best model
    if best_state is not None:
        strategy.model.load_state_dict(best_state)
        strategy.model.to(strategy.device)

    # 11. Save
    model_path = f'models/transformer_optimized_tb{int(args.tp*1000)}_{int(args.sl*1000)}.pth'
    strategy.save_model(model_path)
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Best Val Loss: {best_val_loss:.4f} (Baseline: {baseline:.4f}) at epoch {best_epoch}")
    logger.info(f"Best Val Accuracy: {best_val_acc:.4f}")
    logger.info(f"Best Val F1 Score: {best_val_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Optimized Transformer with Triple Barrier')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--window', type=int, default=60, help='Lookback window')
    parser.add_argument('--data', type=str, help='Path to CSV')
    parser.add_argument('--tp', type=float, default=0.010, help='Take Profit (1%)')
    parser.add_argument('--sl', type=float, default=0.007, help='Stop Loss (0.7%)')
    parser.add_argument('--timeout', type=int, default=24, help='Timeout in bars')

    # New optimization parameters
    parser.add_argument('--d_model', type=int, default=64, help='Transformer model dimension')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--patience', type=int, default=15, help='Patience for early stopping')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')

    args = parser.parse_args()

    try:
        train_main(args)
    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)


