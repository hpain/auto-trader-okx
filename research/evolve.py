# research/evolve.py
import sys
import os, json
import pandas as pd

# 将项目根目录加入 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from data.okx import get_klines
from data.news import load_news_from_csv, aggregate_daily_sentiment
from features.feature_engineering import add_tech_indicators, merge_price_and_sentiment, make_supervised
from models.evolution import train_evolve
from trader.okx_client import OKXClient

NEWS_CSV = os.getenv("NEWS_CSV_PATH", "news_sample.csv")  # 先支持本地CSV，列: timestamp,title

def main():
    symbol = config.get("trade", {}).get("symbol","BTC-USDT")
    interval = config.get("trade", {}).get("interval","1H")

    # 价格数据
    client = OKXClient(**config["okx"])
    dfp = get_klines(client, symbol, interval, max_candles=10000)
    if dfp is None or dfp.empty:
        print("❌ 价格数据为空")
        return

    # 技术指标
    dfp = add_tech_indicators(dfp)

    # 新闻情绪（可选）
    if os.path.exists(NEWS_CSV):
        news_df = load_news_from_csv(NEWS_CSV)
        daily_sent = aggregate_daily_sentiment(news_df)
        dfm = merge_price_and_sentiment(dfp, daily_sent)
    else:
        print(f"⚠️ 未找到新闻CSV: {NEWS_CSV}，将仅使用技术指标。")
        dfm = dfp.copy()
        dfm["sent_mean"] = 0.0
        dfm["sent_median"] = 0.0
        dfm["count"] = 0.0

    # 构建监督学习数据
    data = make_supervised(dfm, horizon=1)

    # 特征列
    feature_cols = [
        "open","high","low","close","vol",
        "rsi_14","ema_12","ema_26","macd","macd_signal","atr_14","roc_10",
        "sent_mean","sent_median","count"
    ]
    # 过滤缺失
    data = data.dropna(subset=feature_cols + ["y"]).copy()
    if len(data) < 300:
        print(f"⚠️ 样本太少（{len(data)}）无法有效训练，建议拉更长周期或换小周期。")
        return

    best_score, best_params = train_evolve(data, feature_cols, out_dir="models", n_trials=30)
    print(f"✅ 训练完成：best sharpe={best_score:.3f}")
    print("最佳参数：", json.dumps(best_params, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
