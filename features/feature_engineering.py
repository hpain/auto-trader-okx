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
    
    # --- 新增指标 ---
    # 布林带
    bollinger = ta.volatility.BollingerBands(close=out["close"], window=20, window_dev=2)
    out["bb_mavg"] = bollinger.bollinger_mavg()
    out["bb_hband"] = bollinger.bollinger_hband()
    out["bb_lband"] = bollinger.bollinger_lband()

    # 随机振荡器
    stoch = ta.momentum.StochasticOscillator(high=out["high"], low=out["low"], close=out["close"], window=14, smooth_window=3)
    out["stoch_k"] = stoch.stoch()
    out["stoch_d"] = stoch.stoch_signal()

    return out

def make_supervised(
    df: pd.DataFrame, horizon: int = 1, threshold: float = 0.005
) -> pd.DataFrame:
    """
    构建监督学习数据集。

    此函数会创建一个新的 DataFrame，其中包含：
    1. 原始特征。
    2. 目标变量 'y'：如果未来 `horizon` 周期的收盘价涨幅 >= `threshold`，则为 1，否则为 0。
    3. 未来价格 'future_*'：可用于后续的回测分析（如止盈止损），但不应作为模型的训练特征，以防数据泄露。

    Args:
        df: 包含价格数据的 DataFrame。
        horizon: 预测未来的时间周期数。
        threshold: 定义“上涨”的收益率阈值。

    Returns:
        一个新的 DataFrame，包含特征、目标变量 'y' 和未来价格信息。
    """
    # 创建副本以避免修改原始 DataFrame
    out_df = df.copy()

    # 计算未来N个周期的价格，用于生成目标和回测分析
    # 警告：这些 'future_*' 列不应作为模型训练的特征！
    out_df["future_high"] = out_df["high"].shift(-horizon)
    out_df["future_low"] = out_df["low"].shift(-horizon)
    out_df["future_close"] = out_df["close"].shift(-horizon)

    # 目标变量：未来收盘价是否达到目标阈值
    out_df["y"] = ((out_df["future_close"] / out_df["close"] - 1) >= threshold).astype(int)
    
    # 移除因 shift 操作在末尾产生的 NaN 值，这些行无法用于训练或回测
    out_df = out_df.dropna(subset=["future_high", "future_low", "future_close", "y"])
    return out_df

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
