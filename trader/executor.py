'''
from strategies.moving_average import generate_signal, plot_moving_averages
from trader.okx_client import OKXClient
import yaml
'''
'''
def run():
    with open("config/settings.yaml", "r") as f:
        config = yaml.safe_load(f)
    client = OKXClient(**config["okx"])
    df = client.get_kline(config["trade"]["symbol"], config["trade"]["interval"])
    signal = generate_signal(df)

    if signal in ["buy", "sell"]:
        client.mock_order(signal, config["trade"]["symbol"], config["trade"]["order_size"])
    else:
        print("暂无交易信号")

'''
'''
from data.okx import fetch_ohlcv  # 假设你在这里获取数据

def run():
    from config import config
    from data.okx import get_klines

    client = OKXClient(**config["okx"])
    symbol = config["trade"]["symbol"]
    interval = config["trade"]["interval"]
    df = get_klines(client, symbol, interval)
    
    if df is None or df.empty:
        print("❌ API 返回数据为空，终止运行")
        return
    
    # 计算均线
    df['ma_short'] = df['close'].rolling(window=5).mean()
    df['ma_long'] = df['close'].rolling(window=20).mean()

    # 生成交易信号
    signal = generate_signal(df)
    print(f"📈 生成交易信号: {signal}")

    # 画图保存
    plot_moving_averages(df, symbol)
    '''

'''
    # ✅ 打印数据结构和后几行
    print("\n📊 市场数据预览：")
    print(df.tail(5))           # 打印后5行
    print("\n数据总行数:", len(df))
    print("字段列表:", df.columns.tolist())

   
'''



# trader/executor.py
import json, os
import pandas as pd
from joblib import load

from config import config
from data.okx import get_klines
from strategies.moving_average import generate_signal as ma_signal, plot_moving_averages

def _load_model():
    try:
        pipe = load("models/best_model.pkl")
        with open("models/metadata.json","r",encoding="utf-8") as f:
            meta = json.load(f)
        return pipe, meta
    except Exception:
        return None, None

def run():
    symbol = config.get("trade", {}).get("symbol", "BTC-USDT")
    interval = config.get("trade", {}).get("interval", "1H")

    df = get_klines(None, symbol, interval, limit=400)
    if df is None or df.empty:
        print("🚫 获取数据失败，终止运行")
        return

    # 先用MA画图备用
    df['ma_short'] = df['close'].rolling(window=5, min_periods=5).mean()
    df['ma_long']  = df['close'].rolling(window=20, min_periods=20).mean()
    plot_moving_averages(df, symbol)

    pipe, meta = _load_model()
    if not pipe or not meta:
        print("⚠️ 未找到训练模型，使用MA策略信号。")
        sig = ma_signal(df)
        print(f"📈 交易信号(MA): {sig}")
        return

    feature_cols = meta["feature_cols"]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"⚠️ 模型特征缺失: {missing}，回退MA。")
        sig = ma_signal(df)
        print(f"📈 交易信号(MA): {sig}")
        return

    # 模型推断
    x = df[feature_cols].tail(1)
    try:
        proba = float(pipe.predict_proba(x)[:,1][0])
    except Exception as e:
        print(f"⚠️ 模型推断失败: {e}，回退MA。")
        sig = ma_signal(df)
        print(f"📈 交易信号(MA): {sig}")
        return

    buy_th = meta["best_params"].get("buy_th", 0.55)
    sell_th = meta["best_params"].get("sell_th", 0.45)

    if proba > buy_th:
        sig = "buy"
    elif proba < sell_th:
        # 若你不做空，这里可以改成hold
        sig = "hold"
    else:
        sig = "hold"

    print(f"🧠 模型概率={proba:.3f} | buy_th={buy_th:.2f} sell_th={sell_th:.2f}")
    print(f"📈 交易信号(Model): {sig}")



