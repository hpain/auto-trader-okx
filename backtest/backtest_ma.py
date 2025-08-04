import pandas as pd
import matplotlib.pyplot as plt
from strategies.moving_average import generate_signal
from trader.okx_client import OKXClient
import yaml

def backtest(symbol="BTC-USDT", interval="1h", short=7, long=25):
    with open("config/settings.yaml", "r") as f:
        config = yaml.safe_load(f)
    client = OKXClient(**config["okx"])
    df = client.get_kline(symbol, interval)

    df["ma_short"] = df["close"].rolling(short).mean()
    df["ma_long"] = df["close"].rolling(long).mean()
    df.dropna(inplace=True)

    df["signal"] = 0
    df.loc[df["ma_short"] > df["ma_long"], "signal"] = 1
    df.loc[df["ma_short"] <= df["ma_long"], "signal"] = -1
    df["strategy_ret"] = df["close"].pct_change() * df["signal"].shift()
    df["cum_ret"] = (1 + df["strategy_ret"]).cumprod()

    print(f"策略累计收益: {df['cum_ret'].iloc[-1] - 1:.2%}")

    df[["close", "ma_short", "ma_long"]].plot(figsize=(12, 6), title=f"{symbol} 移动平均策略")
    plt.show()

if __name__ == "__main__":
    for s in ["BTC-USDT", "ETH-USDT", "SOL-USDT", "KDA-USDT", "CFX-USDT", "DOT-USDT"]:
        print(f"\n🔍 回测交易对: {s}")
        backtest(symbol=s)