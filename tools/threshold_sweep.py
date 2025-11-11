import os
import sys
import argparse
import pandas as pd
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to sys.path to allow importing project modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from features.feature_engineering import make_supervised
from utils.backtest import run_backtest
from config import config

def threshold_sweep():
    """
    Performs a sweep of confidence thresholds to find the optimal value
    on a holdout dataset.
    """
    parser = argparse.ArgumentParser(description="Run a backtest with a sweep of confidence thresholds.")
    parser.add_argument("--model-path", type=str, default=os.path.join(config["paths"]["model_dir"], 'best_model.pkl'), help="Path to the trained model .pkl file.")
    parser.add_argument("--cache-path", type=str, default=os.path.join(config["paths"]["feature_cache_dir"], 'features_adf575c053.parquet'), help="Path to the feature cache .parquet file.")
    parser.add_argument("--holdout-pct", type=float, default=0.2, help="Percentage of data to use as the holdout test set.")
    parser.add_argument("--start-thresh", type=float, default=0.30, help="Start of threshold sweep.")
    parser.add_argument("--end-thresh", type=float, default=0.70, help="End of threshold sweep.")
    parser.add_argument("--step", type=float, default=0.02, help="Step for threshold sweep.")
    args = parser.parse_args()

    print(f"--- Threshold Sweep Configuration ---")
    print(f"Model: {args.model_path}")
    print(f"Data Cache: {args.cache_path}")
    print(f"Holdout Set: {args.holdout_pct:.0%}")
    print(f"Threshold Range: {args.start_thresh} to {args.end_thresh} (step: {args.step})")
    print("-------------------------------------\n")

    # --- 1. Load Data and Model ---
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model file not found at {args.model_path}")
        return
    if not os.path.exists(args.cache_path):
        print(f"ERROR: Data cache file not found at {args.cache_path}")
        return

    model = load(args.model_path)
    df = pd.read_parquet(args.cache_path)
    
    # --- 2. Prepare Supervised Dataset and Split ---
    sup_data = make_supervised(df, horizon=1, threshold=0.005)
    
    non_feature_cols = ["ts", "dt", "y", "future_high", "future_low", "future_close", "future_ret", "date", "timestamp", "vol_ccy", "vol_ccy_quote", "confirm"]
    feature_cols = [c for c in sup_data.columns if c not in non_feature_cols and c in model.feature_name_]
    
    sup_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    sup_data.dropna(subset=feature_cols + ["y"], inplace=True)

    n = len(sup_data)
    train_n = int(n * (1 - args.holdout_pct))
    test_set = sup_data.iloc[train_n:]
    
    X_test = test_set[feature_cols]
    
    print(f"Loaded {len(df)} raw data rows.")
    print(f"Supervised data has {len(sup_data)} rows after cleaning.")
    print(f"Using last {len(test_set)} rows for holdout test ({args.holdout_pct:.0%}).\n")

    # --- 3. Run Sweep ---
    thresholds = np.arange(args.start_thresh, args.end_thresh + args.step, args.step)
    results = []

    print("Running backtest for each threshold...")
    for thresh in thresholds:
        predictions = pd.Series(model.predict(X_test), index=X_test.index)
        probabilities = model.predict_proba(X_test)[:, 1]

        total_ret, max_dd, success_rate, trade_count, returns_series = run_backtest(
            predictions=predictions,
            probabilities=probabilities,
            test_data=test_set,
            confidence_threshold=thresh,
            stop_loss_pct=0.02, # Using default values
            take_profit_pct=0.05,
        )
        
        if returns_series.std() > 0 and len(returns_series.loc[returns_series != 0]) > 5:
            trading_periods_per_year = 252 * 24 # Assuming 1H interval
            annualization_factor = np.sqrt(trading_periods_per_year)
            sharpe = (returns_series.mean() / returns_series.std()) * annualization_factor
        else:
            sharpe = 0.0
            
        results.append({
            "threshold": thresh,
            "trade_count": trade_count,
            "total_return": total_ret,
            "sharpe_ratio": sharpe,
            "success_rate": success_rate,
            "max_drawdown": max_dd,
        })

    results_df = pd.DataFrame(results)
    
    # --- 4. Print and Save Results ---
    print("\n--- Sweep Results ---")
    print(results_df.round(4).to_string(index=False))
    
    # Save results to CSV
    results_dir = config["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(results_dir, f"threshold_sweep_{timestamp_str}.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # --- 5. Plot Results ---
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot Sharpe Ratio
    ax1.plot(results_df['threshold'], results_df['sharpe_ratio'], 'g-', marker='o', label='Sharpe Ratio')
    ax1.set_xlabel('Confidence Threshold')
    ax1.set_ylabel('Sharpe Ratio', color='g')
    ax1.tick_params('y', colors='g')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Create a second y-axis for Trade Count
    ax2 = ax1.twinx()
    ax2.plot(results_df['threshold'], results_df['trade_count'], 'b-', marker='x', label='Trade Count')
    ax2.set_ylabel('Trade Count', color='b')
    ax2.tick_params('y', colors='b')

    plt.title('Threshold Sweep: Sharpe Ratio vs. Trade Count')
    fig.tight_layout()
    
    plot_path = os.path.join(results_dir, f"threshold_sweep_{timestamp_str}.png")
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")
    # plt.show()

if __name__ == "__main__":
    threshold_sweep()
