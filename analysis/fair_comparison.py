# analysis/fair_comparison.py
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from data.binance import get_klines_bian
from trader.okx_client import OKXClient
from utils.data_normalization import normalize_binance_df
from features.feature_engineering import generate_features, make_supervised

def fair_model_comparison():
    """公平比较两种模型在相同数据上的表现"""
    print("=== 公平模型比较分析 ===")
    
    # 使用相同的数据集进行比较
    client = OKXClient(**config["okx"])
    symbol = config.get("trade", {}).get("symbol", "BTC-USDT")
    interval = config.get("trade", {}).get("interval", "1H")
    
    print("1. 获取相同的数据集...")
    # 获取2年数据用于公平比较
    dfp = get_klines_bian(client, symbol, interval, years=2, ignore_local=True)
    if dfp is None or dfp.empty:
        print("无法获取数据")
        return
    
    dfp = normalize_binance_df(dfp)
    print(f"   数据时间范围: {dfp.index.min()} → {dfp.index.max()}")
    print(f"   数据样本数: {len(dfp)}")
    print()
    
    # 生成特征
    print("2. 生成特征...")
    dfm = generate_features(dfp, news_csv_path=None)
    
    # 构建监督学习数据
    print("3. 构建监督学习数据...")
    data = make_supervised(dfm, horizon=1, threshold=0.005)  # 使用相同的阈值
    
    # 数据清洗
    non_feature_cols = ["ts", "dt", "y", "future_high", "future_low", "future_close", "future_ret", "date", "timestamp", "vol_ccy", "vol_ccy_quote", "confirm"]
    feature_cols = [c for c in data.columns if c not in non_feature_cols]
    
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(subset=feature_cols + ["y"], inplace=True)
    
    print(f"   特征数量: {len(feature_cols)}")
    print(f"   清洗后样本数: {len(data)}")
    print(f"   正样本比例: {data['y'].mean():.2%}")
    print()
    
    # 在相同数据上测试两种模型
    print("4. 模型性能对比 (在相同数据上):")
    
    # 假设我们已经有了训练好的模型
    model_dir = config["paths"]["model_dir"]
    
    # 检查模型文件是否存在
    improved_model_path = os.path.join(model_dir, "improved_best_model.pkl")
    original_model_path = os.path.join(model_dir, "best_model.pkl")
    
    if os.path.exists(improved_model_path) and os.path.exists(original_model_path):
        from joblib import load
        import lightgbm as lgb
        from sklearn.ensemble import RandomForestClassifier
        from utils.backtest import run_backtest
        
        # 加载模型
        try:
            improved_model = load(improved_model_path)
            original_model = load(original_model_path)
            
            print("   [OK] 模型加载成功")
            
            # 在相同数据上进行预测
            X = data[feature_cols]
            y = data["y"]
            
            print("   在相同数据上的预测表现:")
            
            # 改进模型预测
            if hasattr(improved_model, 'predict'):
                improved_pred = pd.Series(improved_model.predict(X), index=X.index)
                if hasattr(improved_model, 'predict_proba'):
                    improved_prob = improved_model.predict_proba(X)[:, 1]
                else:
                    improved_prob = np.ones(len(X)) * 0.5  # 默认概率
                
                print("     改进模型:")
                print(f"       预测为1的比例: {improved_pred.mean():.2%}")
                print(f"       平均预测概率: {np.mean(improved_prob):.4f}")
            
            # 原始模型预测
            if hasattr(original_model, 'predict'):
                original_pred = pd.Series(original_model.predict(X), index=X.index)
                if hasattr(original_model, 'predict_proba'):
                    original_prob = original_model.predict_proba(X)[:, 1]
                else:
                    original_prob = np.ones(len(X)) * 0.5  # 默认概率
                
                print("     原始模型:")
                print(f"       预测为1的比例: {original_pred.mean():.2%}")
                print(f"       平均预测概率: {np.mean(original_prob):.4f}")
                
        except Exception as e:
            print(f"   [ERROR] 模型加载失败: {e}")
    else:
        print("   [WARNING] 模型文件不存在，无法进行直接比较")
    
    print()
    print("5. 结论:")
    print("   [WARNING] 得分从0.1跃升到0.97确实存在异常")
    print("   [RECOMMENDATION] 建议在完全相同的数据集和条件下重新训练比较")
    print("   [RECOMMENDATION] 需要验证是否真的有更好的实盘表现")

if __name__ == "__main__":
    fair_model_comparison()