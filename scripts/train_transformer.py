import sys
import os
import pandas as pd
import numpy as np
import logging
import pickle
import argparse
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategies.transformer_strategy import TransformerStrategy, TimeSeriesTransformer
from features.feature_engineering import generate_features
from features.proper_triple_barrier import apply_proper_triple_barrier_vectorized
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Trainer")


# Focal Loss Implementation
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [N, C] logits
        # targets: [N] class indices
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            # Alpha weighting
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha.gather(0, targets)
            loss = at * (1 - pt) ** self.gamma * ce_loss
        else:
            loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def select_features(df, feature_cols, target_col, max_features=80):
    """
    Select the most informative features using statistical tests
    For multiclass problems, use f_classif which works with integer targets
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
    # f_classif works with multiclass targets (integer labels)
    selector = SelectKBest(score_func=f_classif, k=min(max_features, len(valid_features)))

    X = df[valid_features]
    y = df[target_col].astype(int)  # Ensure target is integer for multiclass

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

    # 3.5 Enhanced Feature Engineering
    logger.info("Performing enhanced feature engineering...")

    # Add technical indicators that might help with prediction
    if 'close' in df.columns:
        # Add returns and volatility features
        df['returns'] = df['close'].pct_change()
        # Calculate daily volatility for dynamic labeling
        # Use a rolling window (e.g., 24 hours * 60 mins = 1440 bars if 1min, or custom window)
        # Assuming 5m bars, 1 day = 288 bars
        vol_window = 288
        df['volatility'] = df['returns'].rolling(window=vol_window).std()
        
        # Add rolling statistics
        df['close_ma_ratio'] = df['close'] / df['close'].rolling(window=20).mean()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()

        # Add momentum features
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1

        # Fill NaN values created by these operations
        df = df.fillna(method='bfill').fillna(method='ffill')

    logger.info(f"After enhanced feature engineering: {len(df)} rows")

    # 4. Generate Features
    logger.info("Generating features...")
    df_features = generate_features(df)

    # 4. Dynamic Triple Barrier Labeling
    logger.info(f"Applying Dynamic Proper Triple Barrier: TP={args.tp}x Vol, SL={args.sl}x Vol, Timeout={args.timeout}")
    
    # Calculate dynamic barriers based on volatility
    # Use a minimum floor for volatility to avoid zero barriers
    min_vol = 0.001 
    dynamic_vol = df_features['volatility'].clip(lower=min_vol).values
    
    # Calculate TP and SL arrays
    # Pass arrays to the vectorized function
    dynamic_tp = dynamic_vol * args.tp  # e.g., 1.5 * volatility
    dynamic_sl = dynamic_vol * args.sl  # e.g., 1.5 * volatility
    
    # Check max/min barrier size
    logger.info(f"Dynamic TP Stats: Mean={dynamic_tp.mean():.4f}, Max={dynamic_tp.max():.4f}, Min={dynamic_tp.min():.4f}")
    
    df_labeled = apply_proper_triple_barrier_vectorized(
        df_features, 
        tp=dynamic_tp, 
        sl=dynamic_sl, 
        timeout=args.timeout
    )
    target_col = 'y'

    df_labeled.dropna(inplace=True)
    logger.info(f"Data after labeling: {len(df_labeled)}")

    # Label distribution
    label_counts = df_labeled[target_col].value_counts()
    logger.info(f"Labels: {label_counts.to_dict()}")

    # Calculate class counts for balancing
    pos_count = label_counts.get(1, 0)  # Take profit cases
    neg_count = label_counts.get(0, 0)  # Stop loss cases
    neu_count = label_counts.get(-1, 0)  # Timeout cases
    total_count = len(df_labeled)

    logger.info(f"Label distribution - TP(Hit): {pos_count} ({pos_count/total_count:.2%}), SL(Hit): {neg_count} ({neg_count/total_count:.2%}), Timeout: {neu_count} ({neu_count/total_count:.2%})")

    # For CrossEntropyLoss, we need to map labels from {-1, 0, 1} to {0, 1, 2}
    # Update the labels in the dataset
    df_labeled['y_mapped'] = df_labeled[target_col].map({-1: 0, 0: 1, 1: 2})  # {timeout: 0, stop_loss: 1, take_profit: 2}
    target_col = 'y_mapped'  # Use the remapped target column

    # Recalculate counts with mapped labels
    mapped_label_counts = df_labeled[target_col].value_counts()
    pos_count = mapped_label_counts.get(2, 0)  # take_profit class
    neg_count = mapped_label_counts.get(1, 0)  # stop_loss class
    neu_count = mapped_label_counts.get(0, 0)  # timeout class
    
    # Calculate inverse frequency weights for Focal Loss alpha
    # Make weights sum to number of classes (3) for scale consistency
    weights = torch.tensor([
        total_count / (neu_count * 3) if neu_count > 0 else 1.0,
        total_count / (neg_count * 3) if neg_count > 0 else 1.0,
        total_count / (pos_count * 3) if pos_count > 0 else 1.0
    ], dtype=torch.float32)
    
    logger.info(f"Class Weights for Focal Loss: Timeout={weights[0]:.4f}, SL={weights[1]:.4f}, TP={weights[2]:.4f}")

    # Calculate baseline loss for 3-class problem using CrossEntropy
    p_tp = pos_count / total_count
    p_sl = neg_count / total_count
    p_timeout = neu_count / total_count
    # For CrossEntropy loss, baseline is the negative log-likelihood of always predicting the most frequent class
    max_class_prob = max(p_timeout, p_sl, p_tp)  # Use the most frequent class
    baseline = -np.log(max_class_prob + 1e-9)  # CrossEntropy baseline
    logger.info(f"Baseline Loss: {baseline:.4f}")

    # 5. Split: 70% train, 10% val, 20% test
    train_end = int(len(df_labeled) * 0.7)
    val_end = int(len(df_labeled) * 0.8)

    train_df = df_labeled.iloc[:train_end].copy()
    val_df = df_labeled.iloc[train_end:val_end].copy()
    test_df = df_labeled.iloc[val_end:].copy()

    logger.info(f"Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Verify that the mapped labels are correct (should be 0, 1, 2)
    logger.info(f"Target column after mapping: {target_col}")
    logger.info(f"Train labels range: {train_df[target_col].min()} to {train_df[target_col].max()}")
    
    # Ensure target column is integer type for CrossEntropyLoss
    train_df[target_col] = train_df[target_col].astype(int)
    val_df[target_col] = val_df[target_col].astype(int)
    test_df[target_col] = test_df[target_col].astype(int)

    # 7. Feature Selection with correlation analysis
    exclude = ['y', 'y_mapped', 'open', 'high', 'low', 'close', 'symbol'] + [c for c in df_labeled.columns if 'future' in c]
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

    # Use Focal Loss for 3-class imbalance
    # Move weights to the same device as the model
    class_weights = weights.to(strategy.device)
    
    # Use Focal Loss instead of standard CrossEntropy
    strategy.criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    logger.info("Using Focal Loss with gamma=2.0")

    strategy.scaler = scaler

    # 9. Prepare scaled data
    train_scaled = train_df.copy()
    train_scaled[feature_cols] = scaler.transform(train_df[feature_cols])

    val_scaled = val_df.copy()
    val_scaled[feature_cols] = scaler.transform(val_df[feature_cols])

    # 10. Training Setup
    logger.info(f"Training for {args.epochs} epochs [Dynamic Labeling]...")

    from features.tensor_loader import create_lazy_loader
    # import torch  <-- Removed redundant import

    # No sorting/curriculum learning - standard shuffle is better for stability
    train_scaled_enhanced = train_scaled.copy()
    
    # Convert target to long type for CrossEntropyLoss
    train_targets = train_scaled_enhanced[target_col].astype(int)
    val_targets = val_scaled[target_col].astype(int)

    train_loader = create_lazy_loader(
        train_scaled_enhanced[feature_cols],
        train_targets,
        args.window,
        args.batch_size
    )
    val_loader = create_lazy_loader(
        val_scaled[feature_cols],
        val_targets,
        args.window,
        args.batch_size
    )

    best_val_loss = float('inf')
    patience = args.patience
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
        total_grad_norm = 0  # Track gradient norms

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(strategy.device).float()  # Ensure inputs are float32
            # Convert targets to long for CrossEntropyLoss (expects class indices)
            # Ensure targets are flattened to 1D and converted to long
            batch_y = batch_y.to(strategy.device).squeeze().long()

            strategy.optimizer.zero_grad()
            outputs = strategy.model(batch_X)

            # Calculate loss - CrossEntropyLoss expects raw logits and class indices
            loss = strategy.criterion(outputs, batch_y)
            loss.backward()

            # Adaptive gradient clipping with dynamic threshold based on loss magnitude
            # Use a higher clip value initially and reduce it as training progresses
            adaptive_max_norm = max(1.0, 0.1 * loss.item()) if loss.item() > 0 else 1.0
            grad_norm = torch.nn.utils.clip_grad_norm_(strategy.model.parameters(), max_norm=min(adaptive_max_norm, 5.0))
            total_grad_norm += grad_norm.item()
            strategy.optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            # Calculate accuracy and other metrics for multiclass
            # Get predicted class (highest probability)
            _, predicted_classes = torch.max(outputs, 1)  # Get class with highest probability
            train_correct += (predicted_classes == batch_y).sum().item()
            train_total += batch_y.size(0)

            # Calculate TP, FP, FN for F1 score (focus on positive class - take profit = class 2)
            actual_take_profit = (batch_y == 2).float()
            pred_take_profit = (predicted_classes == 2).float()

            # Calculate TP, FP, FN for F1 score (for take profit class)
            train_tp += ((pred_take_profit == 1) & (actual_take_profit == 1)).sum().item()
            train_fp += ((pred_take_profit == 1) & (actual_take_profit == 0)).sum().item()
            train_fn += ((pred_take_profit == 0) & (actual_take_profit == 1)).sum().item()

        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
        avg_grad_norm = total_grad_norm / train_batches if train_batches > 0 else 0
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
                # Convert targets to long for CrossEntropyLoss (expects class indices)
                # Ensure targets are flattened to 1D and converted to long
                batch_y = batch_y.to(strategy.device).squeeze().long()
                outputs = strategy.model(batch_X)

                # Calculate loss - CrossEntropyLoss expects raw logits and class indices
                loss = strategy.criterion(outputs, batch_y)
                val_loss += loss.item()
                val_batches += 1

                # Calculate accuracy and other metrics for multiclass
                # Get predicted class (highest probability)
                _, predicted_classes = torch.max(outputs, 1)  # Get class with highest probability
                val_correct += (predicted_classes == batch_y).sum().item()
                val_total += batch_y.size(0)

                # Calculate TP, FP, FN for F1 score (focus on positive class - take profit = class 2)
                actual_take_profit = (batch_y == 2).float()
                pred_take_profit = (predicted_classes == 2).float()

                # Calculate TP, FP, FN for F1 score (for take profit class)
                val_tp += ((pred_take_profit == 1) & (actual_take_profit == 1)).sum().item()
                val_fp += ((pred_take_profit == 1) & (actual_take_profit == 0)).sum().item()
                val_fn += ((pred_take_profit == 0) & (actual_take_profit == 1)).sum().item()

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
        strategy.scheduler.step(epoch)  # Pass epoch for cosine annealing
        current_lr = strategy.optimizer.param_groups[0]['lr']

        logger.info(f"Epoch {epoch+1}/{args.epochs} - Train: {avg_train_loss:.4f} (Acc: {train_acc:.4f}, F1: {train_f1:.4f}) - Val: {avg_val_loss:.4f} (Acc: {val_acc:.4f}, F1: {val_f1:.4f}) - Grad: {avg_grad_norm:.2f} - LR: {current_lr:.6f}")

        # Early stopping logic (More Robust)
        
        # Check for new best
        is_better = False
        if avg_val_loss < best_val_loss:
            is_better = True
        
        if is_better:
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
    model_path = f'models/transformer_optimized_dynamic.pth'
    strategy.save_model(model_path)
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Best Val Loss: {best_val_loss:.4f} (Baseline: {baseline:.4f}) at epoch {best_epoch}")
    logger.info(f"Best Val Accuracy: {best_val_acc:.4f}")
    logger.info(f"Best Val F1 Score: {best_val_f1:.4f}")

    # Save metadata for the model
    import json
    metadata = {
        'model_type': 'Transformer_Optimized_Dynamic',
        'input_dim': len(feature_cols),
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'features': feature_cols,
        'best_val_f1': best_val_f1,
        'best_val_loss': best_val_loss,
        'baseline_loss': baseline,
        'training_date': pd.Timestamp.now().isoformat(),
        'parameters': vars(args)
    }

    metadata_path = f'models/transformer_optimized_dynamic_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Optimized Transformer with Dynamic Triple Barrier')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--window', type=int, default=60, help='Lookback window')
    parser.add_argument('--data', type=str, help='Path to CSV')
    # These are now MULTIPLIERS for volatility
    parser.add_argument('--tp', type=float, default=1.5, help='Take Profit (Multiplier of Volatility)')
    parser.add_argument('--sl', type=float, default=1.5, help='Stop Loss (Multiplier of Volatility)')
    parser.add_argument('--timeout', type=int, default=24, help='Timeout in bars')

    # New optimization parameters for 3-class problem
    parser.add_argument('--d_model', type=int, default=64, help='Transformer model dimension')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--patience', type=int, default=15, help='Patience for early stopping')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')

    args = parser.parse_args()

    try:
        train_main(args)
    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)


