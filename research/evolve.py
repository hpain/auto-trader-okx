# research/evolve.py
import sys, os, json, argparse
import pandas as pd

# 将项目根目录加入 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from data.okx import get_klines
from data.news import load_news_from_csv, aggregate_daily_sentiment
from features.feature_engineering import add_tech_indicators, merge_price_and_sentiment, make_supervised
from models.evolution import train_evolve
from trader.okx_client import OKXClient
from research.features_utils import add_features

NEWS_CSV = os.getenv("NEWS_CSV_PATH", "news_sample.csv")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=3, help="拉取多少年数据（默认3年）")
    parser.add_argument("--models", type=str, default="logreg,rf,lgb", help="使用的模型类型，逗号分隔")
    parser.add_argument("--trials", type=int, default=50, help="Optuna 搜索的 trial 数（默认50）")
    args = parser.parse_args()

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]

    symbol = config.get("trade", {}).get("symbol", "BTC-USDT")
    interval = config.get("trade", {}).get("interval", "1H")

    print(f"拉取年数：{args.years}")
    # === 拉取价格数据 ===
    client = OKXClient(**config["okx"])
    dfp = get_klines(client, symbol, interval, years=args.years)
    if dfp is None or dfp.empty:
        print("❌ 价格数据为空")
        return

    # === 技术指标 & 特征 ===
    dfp = add_tech_indicators(dfp)
    dfp = add_features(dfp)
    print(f"📊 已添加因子，最终样本数: {len(dfp)}")

    # === 新闻情绪因子（可选） ===
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

    # === 构建监督学习数据 ===
    data = make_supervised(dfm, horizon=1)
    feature_cols = [
        "open","high","low","close","vol",
        "rsi_14","ema_12","ema_26","macd","macd_signal","atr_14","roc_10",
        "sent_mean","sent_median","count"
    ]

    data = data.dropna(subset=feature_cols + ["y"]).copy()
    if len(data) < 300:
        print(f"⚠️ 样本太少（{len(data)}）无法有效训练，建议拉更长周期或换小周期。")
        return

    # === 训练 ===
    best_score, best_params = train_evolve(
        data, feature_cols,
        out_dir="models",
        n_trials=args.trials
    )
    print(f"✅ 训练完成：best sharpe={best_score:.3f}")
    print("最佳参数：", json.dumps(best_params, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
