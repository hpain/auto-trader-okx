
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import argparse
import logging

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.load_history import load_history_csv
from features.feature_engineering import generate_features
from config import config
from data.binance import get_klines_bian
from trader.okx_client import OKXClient
from utils.data_normalization import normalize_binance_df

def verify_features(args):
    """
    Loads data, generates features, and plots the Bayesian regime probabilities
    against the price to visually verify their behavior.
    """
    print("1. Loading historical data...")
    df_raw = None
    if args.ignore_local:
        print(f"   - --ignore-local is set. Fetching latest {args.years} year(s) of data from Binance...")
        try:
            client = OKXClient(**config["okx"]) # OKXClient can be used for general purposes
            symbol = config.get("trade", {}).get("symbol", "BTC-USDT")
            interval = config.get("trade", {}).get("interval", "1H")
            df_raw = get_klines_bian(client, symbol, interval, years=args.years, ignore_local=True)
            if df_raw is None or df_raw.empty:
                logging.error("Failed to fetch price data from Binance.")
                return
            df_raw = normalize_binance_df(df_raw)
            print("   - Data fetched and normalized successfully.")
        except Exception as e:
            logging.error(f"An error occurred while fetching data: {e}", exc_info=True)
            return
    else:
        print("   - Loading data from local CSV...")
        try:
            # Default data path if not fetching
            data_path = os.path.join(config["paths"]["history_data_dir"], f'binance_{symbol.replace("-", "")}_{interval}_{args.years}y.csv')
            df_raw = load_history_csv(data_path)
            if df_raw.empty:
                print(f"   - WARNING: Local data file is empty or not found at {data_path}.")
                print(f"   - Please run with --ignore-local --years {args.years} to fetch it.")
                return
            if 'ts' in df_raw.columns:
                df_raw = df_raw.set_index('ts')
            df_raw.sort_index(inplace=True)
            print(f"   - Data loaded successfully: {data_path}")
        except FileNotFoundError:
            print(f"   - ERROR: Data file not found at {data_path}. Please run with --ignore-local.")
            return

    print("2. Generating features, including Bayesian probabilities...")
    df_features = generate_features(df_raw, news_csv_path=None)
    print("   - Features generated.")

    # --- Start of Diagnosis Code ---
    import ta
    df_diag = df_features.copy()
    df_diag['ma_slope_20'] = ta.trend.SMAIndicator(df_diag["close"], window=20).sma_indicator().diff()
    df_diag['atr_norm_14'] = ta.volatility.AverageTrueRange(high=df_diag["high"], low=df_diag["low"], close=df_diag["close"], window=14).average_true_range() / df_diag['close']

    print("3. Analyzing feature distributions...")
    print("\n--- MA Slope (20-period) Distribution ---")
    print(df_diag['ma_slope_20'].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
    print("\n--- Normalized ATR (14-period) Distribution ---")
    print(df_diag['atr_norm_14'].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
    print("\n")
    # --- End of Diagnosis Code ---

    # Create a subset for visualization, now using the full dataframe
    df_subset = df_features.copy()

    print("4. Plotting results...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle('Visual Verification of Bayesian Regime Features', fontsize=16)

    ax1.plot(df_subset.index, df_subset['close'], label='Close Price', color='blue')
    ax1.set_ylabel('Price (USDT)')
    ax1.set_title('BTC/USDT Close Price')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    ax2.stackplot(df_subset.index,
                  df_subset['prob_bull'],
                  df_subset['prob_ranging'],
                  df_subset['prob_bear'],
                  labels=['Bullish', 'Ranging', 'Bearish'],
                  colors=['#2ca02c', '#ff7f0e', '#d62728'],
                  alpha=0.7)
    
    ax2.set_ylabel('Probability')
    ax2.set_title('Bayesian Regime Probabilities')
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.xlabel('Date')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = "bayesian_features_analysis.png"
    plt.savefig(output_path)
    
    print(f"5. Plot saved to: {os.path.abspath(output_path)}")
    print("\nDiagnosis script finished. Please check the generated image.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ignore-local", action="store_true", help="忽略本地CSV历史文件，拉取最新数据")
    parser.add_argument("--years", type=int, default=1, help="拉取多少年数据 (仅在 --ignore-local 生效时使用)")
    
    args = parser.parse_args()
    verify_features(args)
