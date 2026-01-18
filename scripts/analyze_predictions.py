
import logging
import pandas as pd
import numpy as np
import torch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from features.feature_engineering import generate_features
from strategies.transformer_strategy import TransformerStrategy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_analysis():
    logger.info("Starting prediction analysis...")
    
    # 1. Load Data
    price_path = "data/history/BTCUSDT_FULL_2024_2025.csv"
    metrics_path = "data/history/BTCUSDT_metrics_2024_2024.csv"
    
    if not os.path.exists(price_path):
        logger.error(f"Price data not found: {price_path}")
        return

    logger.info(f"Loading price data from {price_path}")
    df = pd.read_csv(price_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df.sort_index()
    
    # Load Metrics if available
    derivatives_dfs = {}
    if os.path.exists(metrics_path):
        logger.info(f"Loading metrics data from {metrics_path}")
        metrics_df = pd.read_csv(metrics_path)
        metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
        metrics_df.set_index('timestamp', inplace=True)
        # Rename for generate_features compatibility
        derivatives_dfs['okx'] = metrics_df
    
    # 2. Generate Features
    logger.info("Generating features...")
    # We use a dummy 200 row head to check if it works first? No, full run.
    # To save time, maybe take last 3000 rows.
    df = df.iloc[-3000:]
    
    try:
        full_df = generate_features(df, derivatives_dfs=derivatives_dfs)
        logger.info(f"Generated DataFrame shape: {full_df.shape}")
    except Exception as e:
        logger.error(f"Feature generation failed: {e}")
        return

    # 3. Load Model & Scaler
    strategy = TransformerStrategy(
        strategy_name="Analysis",
        window_size=120, 
        buy_threshold=0.60, 
        sell_threshold=0.40
    )
    
    # Ensure Torch is available
    if not strategy.model: 
        # Manually trigger build (load_model does it but let's be safe)
        # We need input_dim first. 
        # Scaler should tell us input dim?
        pass

    logger.info("Loading Scaler...")
    strategy.load_scaler("models/scaler.pkl")
    
    if not strategy.scaler:
        logger.error("Scaler failed to load.")
        return
        
    input_dim = strategy.scaler.n_features_in_
    logger.info(f"Scaler expects {input_dim} features.")
    
    logger.info("Loading Model...")
    strategy.load_model("models/transformer_v2.pth", input_dim=input_dim)
    
    if not strategy.model:
        logger.error("Model failed to load.")
        return

    # 4. Run Inference in Batch
    # Strategy 'generate_signal' typically runs on one window.
    # To check distribution, we want to slide the window across the last N rows.
    probs = []
    
    logger.info("Running inference on last 500 steps...")
    
    # We need sliding windows of size 120
    # Data is already generated features. 
    # We need to replicate the 'generate_signal' logic which does scaling inside.
    # But doing it row by row is slow.
    # IMPORTANT: generate_signal SCALES data inside using the scaler. 
    # So we pass the raw features (full_df) but limited to the window.
    
    # Optimization: Pre-scale entire DF? 
    # No, generate_signal logic is specific. Let's just call it N times.
    
    window_size = 120
    if len(full_df) < window_size + 10:
        logger.error("Not enough data.")
        return
        
    # Iterate through the last 500 valid start points
    start_indices = range(len(full_df) - 500, len(full_df))
    
    for i in start_indices:
        if i < window_size: continue
        
        # Slice window ending at i
        # generate_signal takes the whole DF and uses .iloc[-window:], so we pass a slice
        slice_df = full_df.iloc[:i+1] # up to current step
        
        # Hack: generate_signal looks at the END of the df passed.
        # But generate_signal calls scaler.transform(). 
        # If we pass a slice, it works.
        
        # Extract prob manually to avoid threshold logic hiding the raw value
        # Or just modify the strategy to return prob? 
        # No, I will copy-paste the inference part here for speed/access.
        
        # 4a. Select features
        if strategy.features:
             # Impute missing
             missing = [f for f in strategy.features if f not in slice_df.columns]
             if missing:
                for m in missing: slice_df[m] = 0.0
             target_df = slice_df[strategy.features].iloc[-window_size:]
        else:
             target_df = slice_df.iloc[-window_size:]
             
        # 4b. Scale
        try:
            scaled = strategy.scaler.transform(target_df)
            scaled = np.nan_to_num(scaled, nan=0.0)
            
            with torch.no_grad():
                inp = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(strategy.device)
                logits = strategy.model(inp)
                prob = torch.sigmoid(logits).item()
                probs.append(prob)
        except Exception as e:
            # logger.error(f"Inference error at {i}: {e}")
            pass
            
    # 5. Analyze Distribution
    if not probs:
        logger.error("No predictions generated.")
        return

    probs = np.array(probs)
    
    logger.info("-" * 40)
    logger.info(f"ANALYSIS RESULTS (N={len(probs)})")
    logger.info("-" * 40)
    logger.info(f"Mean Prob:   {probs.mean():.4f}")
    logger.info(f"Median Prob: {np.median(probs):.4f}")
    logger.info(f"Min Prob:    {probs.min():.4f}")
    logger.info(f"Max Prob:    {probs.max():.4f}")
    logger.info(f"Std Dev:     {probs.std():.4f}")
    
    # Percentiles
    for p in [10, 25, 50, 75, 90, 95]:
        logger.info(f"Percentile {p}: {np.percentile(probs, p):.4f}")
        
    logger.info("-" * 40)
    logger.info("POTENTIAL TRIGGERS at Thresholds:")
    for th in [0.55, 0.60, 0.65, 0.70]:
        count = (probs > th).sum()
        pct = (count / len(probs)) * 100
        logger.info(f"> {th:.2f}: {count} signals ({pct:.1f}%)")
        
    for th in [0.45, 0.40, 0.35, 0.30]:
        count = (probs < th).sum()
        pct = (count / len(probs)) * 100
        logger.info(f"< {th:.2f}: {count} signals ({pct:.1f}%)")

if __name__ == "__main__":
    run_analysis()
