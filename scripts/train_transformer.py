import sys
import os
import pandas as pd
import logging
import time
import argparse

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.feature_engineering import generate_features
from strategies.transformer_strategy import TransformerStrategy

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def train_and_save():
    logger.info("--- Starting Full Transformer Training ---")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to training data CSV")
    parser.add_argument("--model_path", type=str, default="models/transformer_v1.pth", help="Path to save trained model")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    # 1. Config
    DATA_PATH = args.data
    MODEL_PATH = args.model_path
    EPOCHS = args.epochs
    BATCH_SIZE = 64
    
    # Ensure models dir exists
    os.makedirs('models', exist_ok=True)
    
    # 2. Load Data
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data file not found: {DATA_PATH}")
        return
        
    df = pd.read_csv(DATA_PATH)
    
    # Standardize Column Names
    df.rename(columns={'ts': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    logger.info(f"Loaded {len(df)} candles.")
    
    # 3. Features
    logger.info("Generating features...")
    df = generate_features(df)
    df.dropna(inplace=True)
    
    # Target (Next Close > Current Close)
    df['target_up'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # 4. Strategy & Model
    start_time = time.time()
    strategy = TransformerStrategy(window_size=60)
    
    # Build Model
    # Note: strategy.features is auto-set. input_dim = len(features)
    strategy.build_model(input_dim=len(strategy.features))
    
    if strategy.device == 'cpu':
        logger.warning("⚠️ Training on CPU! This might be slow. GPU is recommended for full training.")
    else:
        logger.info(f"🚀 Training on GPU: {strategy.device.upper()}")
        
    # 5. Train
    logger.info(f"Training for {EPOCHS} epochs...")
    strategy.train_model(df, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Training Complete in {duration:.2f} seconds.")
    
    # 6. Save
    strategy.save_model(MODEL_PATH)
    logger.info(f"✅ Model saved to: {os.path.abspath(MODEL_PATH)}")
    logger.info("Transfer this file to your cloud server for live trading.")

if __name__ == "__main__":
    train_and_save()
