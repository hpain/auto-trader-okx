import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime, timedelta
from trader.okx_client import OKXClient
import yaml
import os
import joblib

def extract_features(df: pd.DataFrame):
    df["return"] = df["close"].pct_change()
    df["volatility"] = df["return"].rolling(5).std()
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["ma_ratio"] = df["ma_5"] / df["ma_20"]
    df["volume"] = df["vol"].astype(float)
    df = df.dropna()
    return df

def label_data(df):
    future_return = df["close"].pct_change().shift(-1)
    df["label"] = (future_return > 0).astype(int)
    return df.dropna()

def train_model(symbol="BTC-USDT", interval="1h"):
    with open("config/settings.yaml", "r") as f:
        config = yaml.safe_load(f)
    client = OKXClient(**config["okx"])
    df = client.get_kline(symbol, interval)
    df = extract_features(df)
    df = label_data(df)

    features = ["return", "volatility", "ma_ratio", "volume"]
    X = df[features]
    y = df["label"]

    model = lgb.LGBMClassifier(n_estimators=100)
    model.fit(X, y)

    os.makedirs("models/checkpoints", exist_ok=True)
    joblib.dump(model, f"models/checkpoints/{symbol.replace('-', '_')}_model.pkl")
    print(f"✅ 模型已训练并保存：{symbol}")

def predict(symbol="BTC-USDT", interval="1h"):
    with open("config/settings.yaml", "r") as f:
        config = yaml.safe_load(f)
    client = OKXClient(**config["okx"])
    df = client.get_kline(symbol, interval)
    df = extract_features(df)

    features = ["return", "volatility", "ma_ratio", "volume"]
    model_path = f"models/checkpoints/{symbol.replace('-', '_')}_model.pkl"

    if not os.path.exists(model_path):
        print(f"⚠️ 找不到模型文件，请先运行 train_model()")
        return None

    model = joblib.load(model_path)
    X_latest = df[features].iloc[-1:]
    proba = model.predict_proba(X_latest)[0]
    print(f"📈 [{symbol}] 预测上涨概率: {proba[1]:.2%}")
    return "buy" if proba[1] > 0.55 else "sell" if proba[1] < 0.45 else "hold"

if __name__ == "__main__":
    train_model("BTC-USDT")
    signal = predict("BTC-USDT")
    print("模型建议操作：", signal)