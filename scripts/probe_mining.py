import sys
import os
import pandas as pd
import numpy as np
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.factor_mining import FactorMiner

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_factor_mining():
    logging.info("--- Starting Lightweight Factor Mining Probe ---")
    
    # 1. Generate Synthetic Data (Mocking a crypto OHLCV dataset)
    logging.info("Generating synthetic data...")
    dates = pd.date_range(start='2024-01-01', periods=500, freq='H')
    df = pd.DataFrame(index=dates)
    df['close'] = np.cumsum(np.random.randn(500)) + 100
    df['open'] = df['close'] + np.random.randn(500) * 0.1
    df['high'] = df[['open', 'close']].max(axis=1) + np.random.rand(500) * 0.2
    df['low'] = df[['open', 'close']].min(axis=1) - np.random.rand(500) * 0.2
    df['volume'] = np.random.randint(100, 1000, 500).astype(float)
    
    # Target: Next period return
    df['target'] = df['close'].shift(-1) / df['close'] - 1
    df = df.dropna()
    
    feature_names = ['open', 'high', 'low', 'close', 'volume']
    
    # 2. Initialize Miner
    logging.info("Initializing FactorMiner...")
    miner = FactorMiner(
        generations=2,          # Keep it small for speed
        population_size=100,     # Small population
        n_components=3,         # Find top 3 factors
        random_state=42
    )
    
    # 3. Fit (Run Genetic Programming)
    logging.info("Running Genetic Programming (Fit)...")
    try:
        miner.fit(df[feature_names], df['target'], feature_names=feature_names)
        logging.info("Fit complete.")
    except Exception as e:
        logging.error(f"Fit failed: {e}")
        return
    
    # 4. Extract and Save
    logging.info("Extracting best factors...")
    factors = miner.extract_best_factors(feature_names=feature_names, output_path="config/probe_mined_factors.json")
    
    if factors:
        logging.info(f"SUCCESS: Mined {len(factors)} factors.")
        for f in factors:
            logging.info(f"Factor: {f['name']} | Fitness: {f['fitness']:.4f} | Formula: {f['formula']}")
    else:
        logging.warning("No factors found.")

    logging.info("--- Probe Complete ---")

if __name__ == "__main__":
    test_factor_mining()
