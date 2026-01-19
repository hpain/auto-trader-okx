import sys
import os
import pandas as pd
import numpy as np
import logging
import pickle

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategies.transformer_strategy import TransformerStrategy
from features.feature_engineering import generate_features, make_supervised
from sklearn.preprocessing import StandardScaler

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Trainer")

import argparse

def train_main(args):
    # 1. Load Data
    data_path = args.data
    
    # Priority check if no path provided
    if not data_path:
        potential_files = [
            'data/history/BTCUSDT_FULL_2020_2025.csv',       # Default GPU
            'data/history/binance_BTCUSDT_1h_4y.csv',         # VPS Fallback
            'data/history/BTCUSDT_FULL_2024_2025.csv',        # Short VPS Fallback
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

    if 'timestamp' in df.columns:
        if df['timestamp'].dtype == object: 
             df['datetime'] = pd.to_datetime(df['timestamp'])
        else:
             df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
    elif 'date' in df.columns:
        df.index = pd.to_datetime(df['date'])
    
    logger.info(f"Loaded {len(df)} rows.")

    # 2. Generate Features (P2-1 Logic)
    logger.info("Generating features (including logical derivatives)...")
    df_features = generate_features(df)
    
    # 3. Labeling (Target)
    horizon = 1
    threshold = 0.002 
    
    df_labeled = make_supervised(df_features, horizon=horizon, threshold=threshold)
    target_col = 'y'
    
    # Drop NaNs
    df_labeled.dropna(inplace=True)
    logger.info(f"Data after labeling & dropping NaNs: {len(df_labeled)}")

    # 4. Splitting & Scaling
    split_idx = int(len(df_labeled) * 0.8)
    train_df = df_labeled.iloc[:split_idx]
    
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
    
    strategy.build_model(input_dim=len(feature_cols))
    strategy.scaler = scaler
    
    # 6. Train
    logger.info(f"Starting Training for {args.epochs} epochs...")
    
    train_df_scaled = train_df.copy()
    train_df_scaled[feature_cols] = scaler.transform(train_df[feature_cols])
    
    strategy.train_model(train_df_scaled, target_col=target_col, epochs=args.epochs, batch_size=args.batch_size)
    
    # 7. Save Model
    strategy.save_model('models/transformer_v3.pth')
    logger.info("Model saved to models/transformer_v3.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Transformer Model')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--window', type=int, default=60, help='Lookback window size')
    parser.add_argument('--data', type=str, help='Path to CSV data file')
    
    args = parser.parse_args()
    
    try:
        train_main(args)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
