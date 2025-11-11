import pandas as pd
import matplotlib.pyplot as plt
from strategies.moving_average import generate_signal
from trader.okx_client import OKXClient
from config import config # Use the new central config loader

def backtest(symbol, interval):
    """Performs a backtest using parameters from the central config."""
    # Load parameters from the new config structure
    params = config['strategy_params']['ma_crossover']['backtest']
    short = params['short_window']
    long = params['long_window']

    print(f"Running backtest for {symbol} with short={short}, long={long}")

    # NOTE: The data fetching logic here remains unchanged for now.
    # A future step in our roadmap is to make this use local historical data.
    client = OKXClient(**config["okx"])
    df = client.get_kline(symbol, interval)

    if df is None or df.empty:
        print(f"Could not retrieve data for {symbol}. Skipping backtest.")
        return

    df["ma_short"] = df["close"].rolling(short).mean()
    df["ma_long"] = df["close"].rolling(long).mean()
    df.dropna(inplace=True)

    df["signal"] = 0
    df.loc[df["ma_short"] > df["ma_long"], "signal"] = 1
    df.loc[df["ma_short"] <= df["ma_long"], "signal"] = -1
    df["strategy_ret"] = df["close"].pct_change() * df["signal"].shift()
    df["cum_ret"] = (1 + df["strategy_ret"]).cumprod()

    print(f"策略累计收益 (Cumulative Strategy Return): {df['cum_ret'].iloc[-1] - 1:.2%}")

    df[["close", "ma_short", "ma_long"]].plot(figsize=(12, 6), title=f"{symbol} Moving Average Strategy Backtest")
    plt.show()

if __name__ == "__main__":
    # Load symbols to test from the new config structure
    symbols_to_test = config['backtest']['symbols_to_test']
    interval_to_test = config['trading']['interval']

    for s in symbols_to_test:
        print(f"\n🔍 Backtesting symbol: {s}")
        try:
            backtest(symbol=s, interval=interval_to_test)
        except Exception as e:
            print(f"  -> An error occurred during backtest for {s}: {e}")
