# data/news.py
import pandas as pd
from datetime import timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

def load_news_from_csv(path: str) -> pd.DataFrame:
    """
    读取本地CSV新闻文件，要求至少包含：
    - timestamp: 可解析时间（UTC或本地均可，将统一为日期粒度）
    - title: 文本标题
    可选: source, ticker 等
    """
    df = pd.read_csv(path)
    if "timestamp" not in df.columns or "title" not in df.columns:
        raise ValueError("news csv must contain 'timestamp' and 'title' columns.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.dropna(subset=["timestamp", "title"], inplace=True)
    return df

def score_sentiment(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    s = _analyzer.polarity_scores(text)
    return s["compound"]

def aggregate_daily_sentiment(news_df: pd.DataFrame, tz_convert=None) -> pd.DataFrame:
    """
    将新闻聚合到“日”级别：对同一天的标题情绪求均值/中位数/计数等。
    返回索引为日期的DataFrame：['sent_mean','sent_median','count']
    """
    df = news_df.copy()
    if tz_convert:
        df["timestamp"] = df["timestamp"].dt.tz_convert(tz_convert)
    df["date"] = df["timestamp"].dt.floor("D")
    df["sent"] = df["title"].astype(str).map(score_sentiment)
    agg = df.groupby("date").agg(sent_mean=("sent","mean"),
                                 sent_median=("sent","median"),
                                 count=("sent","size")).sort_index()
    return agg
