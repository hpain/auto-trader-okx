import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_signal(df):
    if df is None or len(df) < 20:
        return 'hold'

    df['ma_short'] = df['close'].rolling(window=5).mean()
    df['ma_long'] = df['close'].rolling(window=20).mean()

    df = df.dropna()

    if len(df) < 2:
        return 'hold'

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    if prev['ma_short'] < prev['ma_long'] and curr['ma_short'] > curr['ma_long']:
        return 'buy'
    elif prev['ma_short'] > prev['ma_long'] and curr['ma_short'] < curr['ma_long']:
        return 'sell'
    else:
        return 'hold'

def plot_moving_averages(df, symbol="BTC-USDT"):
    plt.figure(figsize=(12,6))
    plt.plot(df['close'], label='Close Price', color='blue')
    plt.plot(df['ma_short'], label='Short MA', color='orange')
    plt.plot(df['ma_long'], label='Long MA', color='green')
    plt.title(f"{symbol} Price with Moving Averages")
    plt.xlabel("Data Points")
    plt.ylabel("Price")
    plt.legend()

    os.makedirs("charts", exist_ok=True)
    file_path = f"charts/{symbol}_ma_chart.png"
    plt.savefig(file_path)
    plt.close()
    print(f"均线走势图已保存: {file_path}")
