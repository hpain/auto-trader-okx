# trader/executor.py
import json, os, logging
import pandas as pd
from joblib import load

from config import config
from data.okx import get_klines
from strategies.moving_average import generate_signal as ma_signal, plot_moving_averages
from trader.okx_client import OKXClient


def _load_model():
    try:
        model_dir = config["paths"]["model_dir"]
        model_path = os.path.join(model_dir, "best_model.pkl")
        meta_path = os.path.join(model_dir, "metadata.json")
        pipe = load(model_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return pipe, meta
    except FileNotFoundError:
        # This is an expected error if the model is not trained yet.
        # No need to log it as an error, the calling function will handle it.
        return None, None
    except Exception:
        # This catches all other unexpected errors during model loading,
        # like a corrupted file. We log it for debugging.
        logging.error("Failed to load model due to an unexpected error:", exc_info=True)
        return None, None


def run():
    # Instantiate OKX client
    client = OKXClient(
        api_key=config["okx"]["api_key"],
        secret_key=config["okx"]["secret_key"],
        passphrase=config["okx"]["passphrase"],
        flag=config["okx"]["flag"],
    )

    symbol = config.get("trade", {}).get("symbol", "BTC-USDT")
    interval = config.get("trade", {}).get("interval", "1H")
    quantity = config.get("trade", {}).get("quantity", 0.001)

    df = get_klines(client, symbol, interval, limit=400)
    if df is None or df.empty:
        print("获取数据失败，终止运行")
        return

    # 先用MA画图备用
    df['ma_short'] = df['close'].rolling(window=5, min_periods=5).mean()
    df['ma_long'] = df['close'].rolling(window=20, min_periods=20).mean()
    plot_moving_averages(df, symbol)

    pipe, meta = _load_model()
    if not pipe or not meta:
        print("未找到训练模型，使用MA策略信号。")
        sig = ma_signal(df)
        print(f"交易信号(MA): {sig}")
        if sig in ["buy", "sell"]:
            client.place_order(symbol, sig, quantity)
        return

    feature_cols = meta["feature_cols"]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"模型特征缺失: {missing}，回退MA。")
        sig = ma_signal(df)
        print(f"交易信号(MA): {sig}")
        if sig in ["buy", "sell"]:
            client.place_order(symbol, sig, quantity)
        return

    # 模型推断
    x = df[feature_cols].tail(1)
    try:
        proba = float(pipe.predict_proba(x)[:, 1][0])
    except Exception as e:
        print(f"模型推断失败: {e}，回退MA。")
        sig = ma_signal(df)
        print(f"交易信号(MA): {sig}")
        if sig in ["buy", "sell"]:
            client.place_order(symbol, sig, quantity)
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

    print(f"模型概率={proba:.3f} | buy_th={buy_th:.2f} sell_th={sell_th:.2f}")
    print(f"交易信号(Model): {sig}")
    if sig in ["buy", "sell"]:
        client.place_order(symbol, sig, quantity)