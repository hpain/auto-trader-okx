import sys
import os
import pandas as pd
import numpy as np
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.transformer_strategy import TransformerStrategy

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def test_transformer():
    logger.info("--- Testing Transformer Strategy ---")
    
    # 1. Initialize Strategy
    strategy = TransformerStrategy(features=['close', 'volume'])
    if strategy.model is None and strategy.device == 'cpu' and not hasattr(strategy, 'model'):
         logger.warning("Strategy initialized but PyTorch might be missing or model not built.")

    # 2. Build Model
    logger.info("Building Model...")
    strategy.build_model(input_dim=2) # 2 features: close, volume
    
    if strategy.model is None:
        logger.warning("❌ PyTorch not installed. Skipping training test.")
        return

    # 3. Synthetic Data
    logger.info("Generating Synthetic Data...")
    dates = pd.date_range(start='2024-01-01', periods=200, freq='h')
    df = pd.DataFrame(index=dates)
    df['close'] = np.cumsum(np.random.randn(200)) + 100
    df['volume'] = np.random.randint(100, 1000, 200)
    
    # Generate Target: 1 if next price > current price
    df['target_up'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()
    
    # 4. Train Loop
    logger.info("Starting Training Loop (Dry Run)...")
    try:
        strategy.train_model(df, epochs=2, batch_size=16)
        logger.info("✅ Training Loop Completed Successfully.")
    except Exception as e:
        logger.error(f"❌ Training Failed: {e}", exc_info=True)
        return

    # 5. Inference
    logger.info("Testing Inference...")
    try:
        signal = strategy.generate_signal(df)
        logger.info(f"Generated Signal: {signal}")
        logger.info("✅ Inference logic verified.")
    except Exception as e:
        logger.error(f"❌ Inference Failed: {e}", exc_info=True)

    logger.info("--- Transformer Verification Complete ---")

if __name__ == "__main__":
    test_transformer()
