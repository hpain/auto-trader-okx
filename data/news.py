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

def fetch_crypto_news(api_key: str, query: str = "crypto", page_size: int = 100) -> pd.DataFrame:
    """
    Fetches crypto news from newsapi.org.
    Requires a NewsAPI key.
    """
    import requests
    print(f"📡 Fetching news for query: {query}...")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "apiKey": api_key,
        "pageSize": page_size,
        "sortBy": "publishedAt",
        "language": "en"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data["status"] != "ok" or "articles" not in data:
            print(f"❌ News API error: {data.get('message', 'Unknown error')}")
            return pd.DataFrame()

        articles = data["articles"]
        if not articles:
            print("⚠️ No news articles found.")
            return pd.DataFrame()

        df = pd.DataFrame(articles)
        df = df[["publishedAt", "title", "source"]]
        df.rename(columns={"publishedAt": "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["source"] = df["source"].apply(lambda x: x.get("name") if isinstance(x, dict) else x)


        print(f"✅ Successfully fetched {len(df)} news articles.")
        return df

    except requests.exceptions.RequestException as e:
        print(f"❌ Network request failed: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Failed to process news data: {e}")
        return pd.DataFrame()
