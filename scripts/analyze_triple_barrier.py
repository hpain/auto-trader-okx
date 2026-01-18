"""
Triple Barrier Parameter Analysis

Tests different TP/SL/Timeout combinations to find better label settings.
"""
import pandas as pd
import numpy as np
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.feature_engineering import apply_triple_barrier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    print("="*70)
    print("TRIPLE BARRIER PARAMETER ANALYSIS")
    print("="*70)
    
    # Load data
    df = pd.read_csv('data/history/CLEAN_UNIVERSAL_2022_2026.csv')
    btc = df[df['symbol'] == 'BTCUSDT'].copy()
    btc['timestamp'] = pd.to_datetime(btc['timestamp'])
    btc = btc.set_index('timestamp').sort_index()
    
    print(f"\nData: {len(btc)} rows, {btc.index.min()} to {btc.index.max()}")
    
    # Calculate market volatility
    returns = btc['close'].pct_change().dropna()
    hourly_vol = returns.std()
    daily_vol = hourly_vol * np.sqrt(24)
    
    print(f"\nMarket Volatility:")
    print(f"  Hourly: {hourly_vol*100:.3f}%")
    print(f"  Daily: {daily_vol*100:.2f}%")
    print(f"  Suggested TP (2x daily vol): {daily_vol*2*100:.2f}%")
    print(f"  Suggested SL (1x daily vol): {daily_vol*100:.2f}%")
    
    # Test different parameter combinations
    print("\n" + "="*70)
    print("TESTING DIFFERENT TP/SL/TIMEOUT COMBINATIONS")
    print("="*70)
    print(f"{'TP':>6} {'SL':>6} {'Timeout':>8} | {'Label 0':>8} {'Label 1':>8} {'Ratio':>6} | {'Avg Bars':>10}")
    print("-"*70)
    
    configs = [
        # Current (tight)
        (0.008, 0.005, 12),
        # Wider TP/SL
        (0.010, 0.007, 12),
        (0.012, 0.008, 12),
        (0.015, 0.010, 12),
        (0.020, 0.012, 12),
        # Longer timeout
        (0.010, 0.007, 24),
        (0.015, 0.010, 24),
        (0.020, 0.012, 24),
        # Asymmetric (wider TP)
        (0.015, 0.005, 12),
        (0.020, 0.007, 12),
        # Based on volatility
        (daily_vol * 1.5, daily_vol * 0.8, 24),
        (daily_vol * 2.0, daily_vol * 1.0, 24),
    ]
    
    for tp, sl, timeout in configs:
        try:
            labeled = apply_triple_barrier(btc.copy(), tp=tp, sl=sl, timeout=timeout)
            
            counts = labeled['y'].value_counts()
            label_0 = counts.get(0, 0)
            label_1 = counts.get(1, 0)
            total = label_0 + label_1
            ratio = label_1 / total if total > 0 else 0
            
            print(f"{tp*100:>5.2f}% {sl*100:>5.2f}% {timeout:>8}h | {label_0:>8,} {label_1:>8,} {ratio:>5.1%} |")
        except Exception as e:
            print(f"{tp*100:>5.2f}% {sl*100:>5.2f}% {timeout:>8}h | ERROR: {e}")
    
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print("""
For a healthy model, aim for:
- Label ratio between 40-60% (moderate imbalance)
- TP/SL based on actual market volatility

Current TP=0.8%/SL=0.5% may be too tight for 1H crypto data.
Consider using TP=1.5-2.0% / SL=0.8-1.0% with 24h timeout.
""")


if __name__ == "__main__":
    main()
