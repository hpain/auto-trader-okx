# research/evolve.py
import sys, os, json, argparse, logging
import pandas as pd
import hashlib

# 将项目根目录加入 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from data.binance import get_klines_bian
from data.news import load_news_from_csv, aggregate_daily_sentiment
from features.feature_engineering import generate_features, merge_price_and_sentiment, make_supervised
from models.evolution import train_evolve
from trader.okx_client import OKXClient
from utils.data_normalization import normalize_binance_df

NEWS_CSV = os.getenv("NEWS_CSV_PATH", "news_sample.csv")

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--years", type=int, default=3, help="拉取多少年数据")
    parser.add_argument("--models", type=str, default="rf,lgb", help="使用的模型类型，逗号分隔")
    parser.add_argument("--trials", type=int, default=100, help="Optuna 搜索的 trial 数")
    parser.add_argument("--ignore-local", action="store_true", help="忽略本地CSV历史文件和特征缓存，直接重新生成")
    parser.add_argument("--patience", type=int, default=10, help="Optuna early stopping patience (当前未使用，可移除)")
    
    # --- 核心参数 ---
    parser.add_argument("--profit-threshold", type=float, default=0.005, help="每日最低收益目标")
    parser.add_argument("--confidence-threshold", type=float, default=0.95, help="执行交易的最低置信度")
    parser.add_argument("--stop-loss-pct", type=float, default=0.02, help="止损百分比 (例如 0.02 代表 2%%)")
    parser.add_argument("--max-drawdown", type=float, default=0.1, help="最大回撤限制 (例如 0.1 代表 10%%)")
    parser.add_argument("--success-rate-threshold", type=float, default=0.75, help="可接受的最低达标交易成功率")

    args = parser.parse_args()

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    symbol = config.get("trade", {}).get("symbol", "BTC-USDT")
    interval = config.get("trade", {}).get("interval", "1H")

    logging.info("--- 策略参数 ---")
    logging.info(f"  拉取年数: {args.years}")
    logging.info(f"  收益目标: >= {args.profit_threshold:.2%}")
    logging.info(f"  置信度门槛: >= {args.confidence_threshold:.2%}")
    logging.info(f"  可接受成功率: >= {args.success_rate_threshold:.2%}")
    logging.info(f"  止损线: {args.stop_loss_pct:.2%}")
    logging.info(f"  最大回撤限制: <= {args.max_drawdown:.2%}")
    logging.info("------------------")

    # --- Feature Caching Logic ---
    config_str = f"{symbol}-{interval}-{args.years}"
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:10]
    cache_dir = "data/cache"
    os.makedirs(cache_dir, exist_ok=True)
    feature_cache_path = os.path.join(cache_dir, f"features_{config_hash}.parquet")

    if os.path.exists(feature_cache_path) and not args.ignore_local:
        logging.info(f"✅ CACHE: Found feature cache, loading from {feature_cache_path}")
        dfm = pd.read_parquet(feature_cache_path)
    else:
        logging.info("⏳ CACHE: No feature cache found or --ignore-local is set, running full data pipeline...")
        # === Data Loading and Processing ===
        client = OKXClient(**config["okx"])
        dfp = get_klines_bian(client, symbol, interval, years=args.years, ignore_local=args.ignore_local)
        if dfp is None or dfp.empty:
            logging.info("❌ Price data is empty")
            return
        dfp = normalize_binance_df(dfp)
        dfp = generate_features(dfp)
        
        if os.path.exists(NEWS_CSV):
            news_df = load_news_from_csv(NEWS_CSV)
            daily_sent = aggregate_daily_sentiment(news_df)
            dfm = merge_price_and_sentiment(dfp, daily_sent)
        else:
            logging.info(f"⚠️ News CSV not found: {NEWS_CSV}, using only technical indicators.")
            dfm = dfp.copy()
            for col in ["sent_mean", "sent_median", "count"]:
                if col not in dfm.columns:
                    dfm[col] = 0.0
        
        dfm.to_parquet(feature_cache_path)
        logging.info(f"💾 CACHE: Features saved to {feature_cache_path}")

    # === 构建监督学习数据 ===
    data = make_supervised(dfm, horizon=1, threshold=args.profit_threshold)
    
    # 动态确定特征列
    non_feature_cols = ["ts", "dt", "y", "future_high", "future_low", "future_close", "future_ret", "date", "timestamp", "vol_ccy", "vol_ccy_quote", "confirm"]
    feature_cols = [c for c in data.columns if c not in non_feature_cols]
    
    data = data.dropna(subset=feature_cols + ["y"]).copy()
    
    if len(data) < 500:
        logging.info(f"⚠️ 样本太少（{len(data)}）无法有效训练。")
        return

    # === 训练 (传入所有新参数) ===
    best_score, best_params = train_evolve(
        data=data,
        feature_cols=feature_cols,
        out_dir="models",
        n_trials=args.trials,
        patience=args.patience,
        model_list=model_list,
        profit_threshold=args.profit_threshold,
        confidence_threshold=args.confidence_threshold,
        stop_loss_pct=args.stop_loss_pct,
        max_drawdown_limit=args.max_drawdown,
        success_rate_threshold=args.success_rate_threshold, # 传入新参数
        interval=interval,
    )
    logging.info(f"✅ 训练完成：best score={best_score:.4f}")
    logging.info(f"最佳参数： {json.dumps(best_params, ensure_ascii=False, indent=2)}")

if __name__ == "__main__":
    main()