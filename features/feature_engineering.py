# features/feature_engineering.py
import pandas as pd
import numpy as np
import ta

# Import news processing functions
from data.news import load_news_from_csv, aggregate_daily_sentiment

def generate_features(df: pd.DataFrame, news_csv_path: str = None) -> pd.DataFrame:
    """
    Generates a streamlined, high-importance set of features.
    """
    out = df.copy()
    
    # Check for required columns
    for col in ["open", "high", "low", "close", "vol"]:
        if col not in out.columns:
            raise ValueError(f"DataFrame must contain column: {col}")

    # --- Base Features (Required for other calculations) ---
    # These are kept as they are fundamental or have high importance.
    out["return"] = out["close"].pct_change()
    if len(out) >= 14:
        out["atr_14"] = ta.volatility.AverageTrueRange(high=out["high"], low=out["low"], close=out["close"], window=14).average_true_range()
        out["rsi_14"] = ta.momentum.RSIIndicator(out["close"], window=14).rsi()
        adx_indicator = ta.trend.ADXIndicator(high=out["high"], low=out["low"], close=out["close"], window=14)
        out["adx"] = adx_indicator.adx()
        out["adx_pos"] = adx_indicator.adx_pos()
        out["adx_neg"] = adx_indicator.adx_neg()
    else:
        out["atr_14"] = np.nan
        out["rsi_14"] = np.nan
        out["adx"] = np.nan
        out["adx_pos"] = np.nan
        out["adx_neg"] = np.nan

    if len(out) >= 20:
        out['vol_ma20'] = out['vol'].rolling(20).mean()
        out['vol_ratio'] = out['vol'] / (out['vol_ma20'] + 1e-9)
        bollinger = ta.volatility.BollingerBands(close=out["close"], window=20, window_dev=2)
        out["bb_mavg"] = bollinger.bollinger_mavg()
        out["bb_hband"] = bollinger.bollinger_hband()
        out["bb_lband"] = bollinger.bollinger_lband()
        # Base SMAs needed for ratio calculation later
        out["sma_20"] = ta.trend.SMAIndicator(out["close"], window=20).sma_indicator()
    else:
        out['vol_ma20'] = np.nan
        out['vol_ratio'] = np.nan
        out["bb_mavg"] = np.nan
        out["bb_hband"] = np.nan
        out["bb_lband"] = np.nan
        out["sma_20"] = np.nan

    if len(out) >= 50:
        out["sma_10"] = ta.trend.SMAIndicator(out["close"], window=10).sma_indicator()
        out["sma_50"] = ta.trend.SMAIndicator(out["close"], window=50).sma_indicator()
        out["ema_5"] = ta.trend.EMAIndicator(out["close"], window=5).ema_indicator()
        out["ema_20"] = ta.trend.EMAIndicator(out["close"], window=20).ema_indicator()
        out["ema_50"] = ta.trend.EMAIndicator(out["close"], window=50).ema_indicator()
    else:
        out["sma_10"] = np.nan
        out["sma_50"] = np.nan
        out["ema_5"] = np.nan
        out["ema_20"] = np.nan
        out["ema_50"] = np.nan

    # --- High Importance Standalone Features ---
    if len(out) >= 1:
        out["obv"] = ta.volume.OnBalanceVolumeIndicator(close=out["close"], volume=out["vol"]).on_balance_volume()
    else:
        out["obv"] = np.nan

    if len(out) >= 10:
        out["roc_10"] = ta.momentum.ROCIndicator(out["close"], window=10).roc()
    else:
        out["roc_10"] = np.nan

    if len(out) >= 14:
        out["williams_r"] = ta.momentum.WilliamsRIndicator(high=out["high"], low=out["low"], close=out["close"], lbp=14).williams_r()
        stoch = ta.momentum.StochasticOscillator(high=out["high"], low=out["low"], close=out["close"], window=14, smooth_window=3)
        out["stoch_k"] = stoch.stoch()
        out["stoch_d"] = stoch.stoch_signal()
    else:
        out["williams_r"] = np.nan
        out["stoch_k"] = np.nan
        out["stoch_d"] = np.nan

    if len(out) >= 26:
        macd = ta.trend.MACD(out["close"])
        out["macd"] = macd.macd()
        out["macd_signal"] = macd.macd_signal()
    else:
        out["macd"] = np.nan
        out["macd_signal"] = np.nan

    # --- High Importance Interaction & Advanced Features ---
    if 'sma_20' in out.columns:
        out['close_to_sma20_ratio'] = out['close'] / out['sma_20']
    if 'ema_50' in out.columns:
        out['close_to_ema50_ratio'] = out['close'] / out['ema_50']
    if 'ema_5' in out.columns and 'ema_20' in out.columns:
        out['ema5_to_ema20_ratio'] = out['ema_5'] / out['ema_20']
    if 'sma_10' in out.columns and 'sma_50' in out.columns:
        out['sma10_to_sma50_spread_norm'] = (out['sma_10'] - out['sma_50']) / out['close']
    if 'roc_10' in out.columns and 'atr_14' in out.columns and not out['atr_14'].isnull().all():
        out['roc_10_norm_by_atr'] = out['roc_10'] / (out['atr_14'] + 1e-9)
    if 'adx' in out.columns and 'rsi_14' in out.columns:
        out['adx_x_rsi'] = out['adx'] * out['rsi_14']

    # Time-based Features
    out['hour_of_day'] = out.index.hour

    # --- News Sentiment Features (Kept as dummy for now) ---
    if news_csv_path:
        try:
            news_df = load_news_from_csv(news_csv_path)
            daily_sent_df = aggregate_daily_sentiment(news_df)
            out = merge_price_and_sentiment(out, daily_sent_df)
            sent_cols = ["sent_mean", "sent_median", "count"]
            for col in sent_cols:
                if col in out.columns:
                    out[col] = out[col].fillna(0)
        except FileNotFoundError:
            out["sent_mean"] = 0.0
            out["sent_median"] = 0.0
            out["count"] = 0
        except Exception:
            out["sent_mean"] = 0.0
            out["sent_median"] = 0.0
            out["count"] = 0
    else:
        out["sent_mean"] = 0.0
        out["sent_median"] = 0.0
        out["count"] = 0

    # Drop the simple MA features that are now encoded in ratios/spreads
    out = out.drop(columns=["sma_5", "sma_10", "sma_20", "sma_50", "ema_5", "ema_10", "ema_20", "ema_50"], errors='ignore')

    return out

def make_supervised(
    df: pd.DataFrame, horizon: int = 1, threshold: float = 0.005
) -> pd.DataFrame:
    """
    构建监督学习数据集（回归模式）。

    此函数会创建一个新的 DataFrame，其中包含：
    1. 原始特征。
    2. 目标变量 'y'：现在是未来 `horizon` 周期的实际连续回报率。
    3. 未来价格 'future_*'：可用于后续的回测分析（如止盈止损），但不应作为模型的训练特征，以防数据泄露。

    Args:
        df: 包含价格数据的 DataFrame。
        horizon: 预测未来的时间周期数。
        threshold: 此参数在回归模式下不再用于定义'y'，但为了保持函数签名一致性暂时保留。

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

    # 目标变量：未来的实际连续回报率
    out_df["y"] = (out_df["future_close"] / out_df["close"]) - 1
    
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
    m[["sent_mean","sent_median","count"]] = m[["sent_mean","sent_median","count"]].ffill()
    m.drop(columns=["date"], inplace=True)
    return m