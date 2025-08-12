import pandas as pd

def generate_signal(df):
    if df is None or len(df) < 20:
        # 数据本身就不够长
        return 'hold'

    df['ma_short'] = df['close'].rolling(window=5).mean()
    df['ma_long'] = df['close'].rolling(window=20).mean()

    # 删除所有包含 NaN 的行
    df = df.dropna()

    if len(df) < 2:
        # 删除 NaN 后行数还不够
        return 'hold'

    # 提取最后两行用于比较
    prev = df.iloc[-2]
    curr = df.iloc[-1]

    if prev['ma_short'] < prev['ma_long'] and curr['ma_short'] > curr['ma_long']:
        return 'buy'
    elif prev['ma_short'] > prev['ma_long'] and curr['ma_short'] < curr['ma_long']:
        return 'sell'
    else:
        return 'hold'

import matplotlib.pyplot as plt
import os

def generate_signal(df):
    # 检查数据是否足够
    if len(df) < 2:
        return "hold"
    
    if df['ma_short'].iloc[-2] < df['ma_long'].iloc[-2] and df['ma_short'].iloc[-1] > df['ma_long'].iloc[-1]:
        return "buy"
    elif df['ma_short'].iloc[-2] > df['ma_long'].iloc[-2] and df['ma_short'].iloc[-1] < df['ma_long'].iloc[-1]:
        return "sell"
    else:
        return "hold"

def plot_moving_averages(df, symbol="BTC-USDT"):
    """
    画出价格和短/长期均线
    """
    plt.figure(figsize=(12,6))
    plt.plot(df['close'], label='Close Price', color='blue')
    plt.plot(df['ma_short'], label='Short MA', color='orange')
    plt.plot(df['ma_long'], label='Long MA', color='green')
    plt.title(f"{symbol} Price with Moving Averages")
    plt.xlabel("Data Points")
    plt.ylabel("Price")
    plt.legend()

    # 确保保存目录存在
    os.makedirs("charts", exist_ok=True)
    file_path = f"charts/{symbol}_ma_chart.png"
    plt.savefig(file_path)
    plt.close()
    print(f"📊 均线走势图已保存: {file_path}")

