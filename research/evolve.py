# research/evolve.py
import sys, os, json, argparse
import pandas as pd

# 将项目根目录加入 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from data.okx import get_klines_bian
from data.news import load_news_from_csv, aggregate_daily_sentiment
from features.feature_engineering import add_tech_indicators, merge_price_and_sentiment, make_supervised
from models.evolution import train_evolve
from trader.okx_client import OKXClient
from research.features_utils import add_features
from utils.data_normalization import normalize_binance_df

NEWS_CSV = os.getenv("NEWS_CSV_PATH", "news_sample.csv")

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--years", type=int, default=3, help="拉取多少年数据")
    parser.add_argument("--models", type=str, default="rf,lgb", help="使用的模型类型，逗号分隔")
    parser.add_argument("--trials", type=int, default=100, help="Optuna 搜索的 trial 数")
    parser.add_argument("--ignore-local", action="store_true", help="忽略本地CSV历史文件，直接从服务器全量拉取")
    parser.add_argument("--patience", type=int, default=10, help="Optuna early stopping patience (当前未使用，可移除)")
    
    # --- 核心参数 ---
    parser.add_argument("--profit-threshold", type=float, default=0.005, help="每日最低收益目标")
    parser.add_argument("--confidence-threshold", type=float, default=0.95, help="执行交易的最低置信度")
    parser.add_argument("--stop-loss-pct", type=float, default=0.02, help="止损百分比 (例如 0.02 代表 2%%)")
    parser.add_argument("--max-drawdown", type=float, default=0.1, help="最大回撤限制 (例如 0.1 代表 10%%)")
    # --- 新增：可配置的成功率阈值 ---
    parser.add_argument("--success-rate-threshold", type=float, default=0.75, help="可接受的最低达标交易成功率")

    args = parser.parse_args()

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    symbol = config.get("trade", {}).get("symbol", "BTC-USDT")
    interval = config.get("trade", {}).get("interval", "1H")

    print("--- 策略参数 ---")
    print(f"  拉取年数: {args.years}")
    print(f"  收益目标: >= {args.profit_threshold:.2%}")
    print(f"  置信度门槛: >= {args.confidence_threshold:.2%}")
    print(f"  可接受成功率: >= {args.success_rate_threshold:.2%}")
    print(f"  止损线: {args.stop_loss_pct:.2%}")
    print(f"  最大回撤限制: <= {args.max_drawdown:.2%}")
    print("------------------")

    # === 数据拉取与处理 ===
    client = OKXClient(**config["okx"])
    dfp = get_klines_bian(client, symbol, interval, years=args.years, ignore_local=args.ignore_local)
    if dfp is None or dfp.empty:
        print("❌ 价格数据为空")
        return
    dfp = normalize_binance_df(dfp)
    dfp = add_tech_indicators(dfp)
    dfp = add_features(dfp)
    
    if os.path.exists(NEWS_CSV):
        news_df = load_news_from_csv(NEWS_CSV)
        daily_sent = aggregate_daily_sentiment(news_df)
        dfm = merge_price_and_sentiment(dfp, daily_sent)
    else:
        print(f"⚠️ 未找到新闻CSV: {NEWS_CSV}，将仅使用技术指标。")
        dfm = dfp.copy()
        dfm["sent_mean"] = 0.0; dfm["sent_median"] = 0.0; dfm["count"] = 0.0

    # === 构建监督学习数据 ===
    data = make_supervised(dfm, horizon=1, threshold=args.profit_threshold)
    
    # 动态确定特征列
    non_feature_cols = ["ts", "dt", "y", "future_high", "future_low", "future_close", "future_ret", "date"]
    feature_cols = [c for c in data.columns if c not in non_feature_cols]
    
    data = data.dropna(subset=feature_cols + ["y"]).copy()
    
    if len(data) < 500:
        print(f"⚠️ 样本太少（{len(data)}）无法有效训练。")
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
    )
    print(f"✅ 训练完成：best score={best_score:.4f}")
    print("最佳参数：", json.dumps(best_params, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
