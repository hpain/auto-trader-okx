import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    给K线数据添加常见技术指标因子
    需要 df 包含列: ['open','high','low','close','vol']
    """

    df = df.copy()

    # === 移动平均线
    for win in [5, 10, 20, 50]:
        df[f"sma_{win}"] = df["close"].rolling(win).mean()
        df[f"ema_{win}"] = df["close"].ewm(span=win, adjust=False).mean()

    # === 动量指标
    df["return"] = df["close"].pct_change()
    df["cum_return"] = (1 + df["return"]).cumprod()

    # RSI
    window = 14
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window).mean()
    avg_loss = pd.Series(loss).rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # 布林带
    rolling_mean = df["close"].rolling(20).mean()
    rolling_std = df["close"].rolling(20).std()
    df["boll_upper"] = rolling_mean + 2 * rolling_std
    df["boll_lower"] = rolling_mean - 2 * rolling_std

    # 波动率 (ATR)
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # 成交量因子
    df["vol_ma20"] = df["vol"].rolling(20).mean()
    df["vol_ratio"] = df["vol"] / (df["vol_ma20"] + 1e-9)

    # 去掉缺失值
    df = df.dropna().reset_index(drop=True)

    return df
