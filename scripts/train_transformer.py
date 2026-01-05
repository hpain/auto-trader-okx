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

def train_main():
    # 1. Load Data
    data_path = 'data/history/binance_BTCUSDT_1h_4y.csv'
    logger.info(f"Loading data from {data_path}...")
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return

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
        # Check if timestamp is string (already parsed) or int (unix ms)
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
    # For training, we assume simulated derivatives if columns missing
    # But generate_features handles missing columns gracefully by imputing
    df_features = generate_features(df)
    
    # 3. Labeling (Target)
    # Target: 1 if return in next 24h > 1% (Volatile Up), 0 otherwise
    # Or simplified: Next candle return > 0.05%
    horizon = 1
    threshold = 0.002 # 0.2% per hour target
    
    df_labeled = make_supervised(df_features, horizon=horizon, threshold=threshold)
    target_col = 'y'
    
    # Drop NaNs
    df_labeled.dropna(inplace=True)
    logger.info(f"Data after labeling & dropping NaNs: {len(df_labeled)}")

    # 4. Splitting & Scaling
    split_idx = int(len(df_labeled) * 0.8)
    train_df = df_labeled.iloc[:split_idx]
    val_df = df_labeled.iloc[split_idx:]
    
    # Select feature columns (All except target and future_)
    exclude = ['y'] + [c for c in df_labeled.columns if 'future' in c]
    # Also exclude non-numeric
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
        window_size=60,
        features=feature_cols,
        buy_threshold=0.6,
        sell_threshold=0.4
    )
    
    strategy.build_model(input_dim=len(feature_cols))
    
    # Assign scaler to strategy for internal use
    strategy.scaler = scaler
    
    # 6. Train
    logger.info("Starting Training...")
    # Passing the FULL train_df because strategy handles scaling internally via self.scaler if logic matches,
    # BUT wait, train_model in strategy expects raw DF and doing lazy loading.
    # The current Strategy.train_model implementation DOES NOT scale automatically inside the loop! 
    # It assumes data passed is ready or simple.
    # HACK: We need to scale the data BEFORE passing to train_model, or update train_model to use scaler.
    # Let's scale the data here for training safety.
    
    train_df_scaled = train_df.copy()
    train_df_scaled[feature_cols] = scaler.transform(train_df[feature_cols])
    
    strategy.train_model(train_df_scaled, target_col=target_col, epochs=5, batch_size=64)
    
    # 7. Save Model
    strategy.save_model('models/transformer_v2.pth')
    logger.info("Model saved to models/transformer_v2.pth")

if __name__ == "__main__":
    try:
        train_main()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
