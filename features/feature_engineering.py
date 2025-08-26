# features/feature_engineering.py
import pandas as pd
import numpy as np
import ta  # technical analysis indicators

def add_tech_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # 需要列: ['open','high','low','close','vol']
    for c in ["open","high","low","close","vol"]:
        if c not in out.columns:
            raise ValueError(f"price df missing column {c}")
    # 常见指标
    out["rsi_14"] = ta.momentum.RSIIndicator(out["close"], window=14).rsi()
    out["ema_12"] = ta.trend.EMAIndicator(out["close"], window=12).ema_indicator()
    out["ema_26"] = ta.trend.EMAIndicator(out["close"], window=26).ema_indicator()
    macd = ta.trend.MACD(out["close"])
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["atr_14"] = ta.volatility.AverageTrueRange(out["high"], out["low"], out["close"], window=14).average_true_range()
    out["roc_10"] = ta.momentum.ROCIndicator(out["close"], window=10).roc()
    return out

def make_supervised(
    df: pd.DataFrame, horizon: int = 1, threshold: float = 0.005
) -> pd.DataFrame:
    """
    构建监督学习数据集.
    y = 1 如果未来收益 >= threshold, 否则 y = 0.
    """
    # 计算未来N个周期的最高价和最低价，用于止损和判断潜在收益
    df["future_high"] = df["high"].shift(-horizon)
    df["future_low"] = df["low"].shift(-horizon)
    df["future_close"] = df["close"].shift(-horizon)

    # 目标变量：未来收盘价是否达到目标阈值
    df["y"] = ((df["future_close"] / df["close"] - 1) >= threshold).astype(int)
    
    df.dropna(subset=["future_high", "future_low", "future_close", "y"], inplace=True)
    return df

def merge_price_and_sentiment(price_df: pd.DataFrame, daily_sent_df: pd.DataFrame) -> pd.DataFrame:
    """
    将日级情绪聚合对齐到K线：
    - 取K线时间的date作为merge key（UTC无tz）
    - 用前向填充方式把日情绪扩展到当日所有bar
    """
    p = price_df.copy()
    p["date"] = p.index.floor("D")
    sent = daily_sent_df.copy()
    # 左连接，再前向填充
    m = p.merge(sent, how="left", left_on="date", right_index=True)
    m[["sent_mean","sent_median","count"]] = m[["sent_mean","sent_median","count"]].fillna(method="ffill")
    m.drop(columns=["date"], inplace=True)
    return m
