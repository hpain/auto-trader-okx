import sys
import os
import pandas as pd
import numpy as np
import logging
import pickle
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategies.transformer_strategy import TransformerStrategy
from features.feature_engineering import generate_features, apply_triple_barrier
from sklearn.preprocessing import StandardScaler

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Trainer")


def train_main(args):
    # 1. Load Data
    data_path = args.data
    
    # Priority check if no path provided
    if not data_path:
        potential_files = [
            'data/history/CLEAN_UNIVERSAL_2022_2026.csv',     # Multi-asset dataset
            'data/history/BTCUSDT_FULL_2020_2025.csv',        # Default GPU
            'data/history/binance_BTCUSDT_1h_4y.csv',         # VPS Fallback
        ]
        for p in potential_files:
            if os.path.exists(p):
                data_path = p
                break
    
    if not data_path:
        logger.error("No data file found. Please provide --data argument.")
        return
    else:
        logger.info(f"Loading data from {data_path}...")

    df = pd.read_csv(data_path)
    
    # Standardize columns
    df.columns = [c.lower() for c in df.columns]
    
    # Handle mappings for compressed/abbreviated headers
    rename_map = {
        'ts': 'timestamp',
        'o': 'open',
        'h': 'high',
        'l': 'low',
        'c': 'close',
        'v': 'volume'
    }
    df.rename(columns=rename_map, inplace=True)

    # Handle timestamp
    if 'timestamp' in df.columns:
        if df['timestamp'].dtype == object: 
             df['datetime'] = pd.to_datetime(df['timestamp'])
        else:
             df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
    elif 'date' in df.columns:
        df.index = pd.to_datetime(df['date'])
    
    logger.info(f"Loaded {len(df)} rows.")

    # Handle multi-symbol data (e.g., CLEAN_UNIVERSAL with BTC+ETH)
    if 'symbol' in df.columns:
        symbols = df['symbol'].unique()
        logger.info(f"Multi-symbol data detected: {symbols.tolist()}")
        # Use only the first symbol for training (typically BTC)
        primary_symbol = symbols[0]
        df = df[df['symbol'] == primary_symbol].copy()
        logger.info(f"Using {primary_symbol} for training: {len(df)} rows")

    # 2. Generate Features
    logger.info("Generating features (including logical derivatives)...")
    df_features = generate_features(df)
    
    # 3. Labeling using Triple Barrier (IMPROVED from simple threshold)
    # Triple Barrier is superior because it considers:
    # - Take Profit (upside)
    # - Stop Loss (downside)  
    # - Timeout (time decay)
    logger.info(f"Applying Triple Barrier labeling (TP={args.tp}, SL={args.sl}, Timeout={args.timeout})...")
    df_labeled = apply_triple_barrier(df_features, tp=args.tp, sl=args.sl, timeout=args.timeout)
    target_col = 'y'
    
    # Drop NaNs
    df_labeled.dropna(inplace=True)
    logger.info(f"Data after labeling & dropping NaNs: {len(df_labeled)}")
    
    # Log label distribution
    label_counts = df_labeled[target_col].value_counts()
    logger.info(f"Label distribution: {label_counts.to_dict()}")

    # 4. Splitting & Scaling (70% train, 10% val, 20% test)
    train_end_idx = int(len(df_labeled) * 0.7)
    val_end_idx = int(len(df_labeled) * 0.8)
    
    train_df = df_labeled.iloc[:train_end_idx]
    val_df = df_labeled.iloc[train_end_idx:val_end_idx]
    test_df = df_labeled.iloc[val_end_idx:]  # Reserved for final evaluation
    
    logger.info(f"Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Select feature columns
    exclude = ['y'] + [c for c in df_labeled.columns if 'future' in c]
    feature_cols = [c for c in df_labeled.columns if c not in exclude and np.issubdtype(df_labeled[c].dtype, np.number)]
    
    logger.info(f"Training with {len(feature_cols)} features.")

    # Fit Scaler
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    
    # Save Scaler
    os.makedirs('models', exist_ok=True)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    logger.info("Scaler saved to models/scaler.pkl")

    # 5. Initialize Strategy & Model
    strategy = TransformerStrategy(
        strategy_name="Transformer_P2",
        window_size=args.window,
        features=feature_cols,
        buy_threshold=0.6,
        sell_threshold=0.4
    )
    
    # Note: NOT using pos_weight for Transformer as it causes overfitting
    # LGB also works better without it - the model should focus on precision
    # rather than trying to predict more class 1s
    strategy.build_model(input_dim=len(feature_cols))
    strategy.scaler = scaler
    
    # 6. Train
    logger.info(f"Starting Training for {args.epochs} epochs...")
    
    train_df_scaled = train_df.copy()
    train_df_scaled[feature_cols] = scaler.transform(train_df[feature_cols])
    
    # Prepare validation data if available
    val_df_scaled = None
    if len(val_df) > 0:
        val_df_scaled = val_df.copy()
        val_df_scaled[feature_cols] = scaler.transform(val_df[feature_cols])
    
    strategy.train_model(
        train_df_scaled, 
        target_col=target_col, 
        epochs=args.epochs, 
        batch_size=args.batch_size,
        val_df=val_df_scaled
    )
    
    # 7. Save Model
    model_path = f'models/transformer_v4_tb{int(args.tp*1000)}_{int(args.sl*1000)}.pth'
    strategy.save_model(model_path)
    logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Transformer Model with Triple Barrier Labeling')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--window', type=int, default=60, help='Lookback window size')
    parser.add_argument('--data', type=str, help='Path to CSV data file')
    
    # Triple Barrier parameters
    parser.add_argument('--tp', type=float, default=0.008, help='Take Profit threshold')
    parser.add_argument('--sl', type=float, default=0.005, help='Stop Loss threshold')
    parser.add_argument('--timeout', type=int, default=12, help='Timeout in bars')
    
    args = parser.parse_args()
    
    try:
        train_main(args)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)

