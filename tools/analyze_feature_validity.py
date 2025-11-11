"""
Feature Validity Analysis Script

Run from project root:
    python tools/analyze_feature_validity.py

This script analyzes the predictive power of features on the most recent data segment
to diagnose feature decay.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif

# Setup paths and import necessary modules from the project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger
from config import config
from data.binance import get_klines_bian
from features.feature_engineering import generate_features, make_supervised
from utils.data_normalization import normalize_binance_df

# Features currently used by the best model (from metadata.json)
# If metadata changes, this list should be updated.
CURRENT_FEATURES = [
    "high", "vol", "return", "atr_14", "adx", "adx_neg", "volatility_10",
    "volatility_20", "vol_lag_2", "vol_lag_5", "vol_ma20", "vol_ratio",
    "bb_hband", "obv", "macd", "macd_signal", "sma10_to_sma50_spread_norm",
    "adx_x_rsi", "hour_of_day", "day_of_week"
]

def analyze_feature_validity(years=3, recent_pct=0.2):
    """
    Analyzes and prints the validity of features on the most recent data.

    Args:
        years (int): How many years of historical data to load.
        recent_pct (float): The percentage of recent data to analyze (e.g., 0.2 for the last 20%).
    """
    setup_logger()
    logging.info("--- Starting Feature Validity Analysis ---")

    # 1. Load data and generate features (simplified from evolve.py)
    logging.info(f"Loading {years} years of data...")
    symbol = config.get("trade", {}).get("symbol", "BTC-USDT")
    interval = config.get("trade", {}).get("interval", "1H")
    news_csv_path = os.path.join(config["paths"]["history_data_dir"], "sample_crypto_news.csv")

    dfp = get_klines_bian(None, symbol, interval, years=years, ignore_local=True) # Force refresh
    if dfp is None or dfp.empty:
        logging.error("Failed to load price data.")
        return

    dfp = normalize_binance_df(dfp)
    dfm = generate_features(dfp, news_csv_path=news_csv_path)
    
    # Using a default profit threshold for creating the target 'y'
    data = make_supervised(dfm, horizon=1, threshold=0.005)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Ensure target and features have no NaNs
    data.dropna(subset=CURRENT_FEATURES + ["y"], inplace=True)

    if len(data) < 100:
        logging.error("Not enough data to perform analysis.")
        return

    # 2. Isolate the most recent data segment
    segment_size = int(len(data) * recent_pct)
    if segment_size < 50:
        logging.error(f"Recent data segment is too small ({segment_size} samples). Try increasing years or recent_pct.")
        return
        
    recent_data = data.tail(segment_size)
    logging.info(f"Analyzing the most recent {segment_size} data points (last {recent_pct*100:.0f}% of data)...")

    X_recent = recent_data[CURRENT_FEATURES]
    y_recent = recent_data["y"]

    # 3. Calculate feature scores (using f_classif for consistency with evolve.py)
    try:
        f_scores, p_values = f_classif(X_recent, y_recent)
    except ValueError as e:
        logging.error(f"Could not calculate feature scores. Error: {e}")
        logging.error("This might happen if a feature has zero variance in the recent data segment.")
        # Print feature variance to help debug
        logging.info("Feature variances in recent data:\n" + str(X_recent.var().sort_values()))
        return

    # 4. Create and display the results
    results_df = pd.DataFrame({
        'feature': CURRENT_FEATURES,
        'f_score': f_scores,
        'p_value': p_values
    })
    results_df.sort_values(by='f_score', ascending=False, inplace=True)
    results_df.reset_index(drop=True, inplace=True)

    logging.info("\n--- Feature Validity Report (Recent Data) ---")
    print(results_df.to_string())
    
    highly_significant = results_df[results_df['p_value'] < 0.05]
    likely_irrelevant = results_df[results_df['p_value'] >= 0.05]

    logging.info(f"\nFound {len(highly_significant)} features that are still statistically significant (p < 0.05).")
    logging.info(f"Found {len(likely_irrelevant)} features that are likely irrelevant (p >= 0.05).")
    
    if not highly_significant.empty:
        logging.info("Top performing features:\n" + highly_significant.head().to_string())
    if not likely_irrelevant.empty:
        logging.info("Most irrelevant features:\n" + likely_irrelevant.head().to_string())

    logging.info("\n--- Analysis Complete ---")


if __name__ == "__main__":
    analyze_feature_validity()
