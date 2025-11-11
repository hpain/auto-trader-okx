# analysis/genuine_comparison.py
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
from sklearn.feature_selection import SelectKBest, f_classif

def genuine_model_comparison():
    """真正公平的模型比较"""
    print("=== 真正公平的模型比较 ===")
    
    # 获取客户端
    client = OKXClient(**config["okx"])
    symbol = config.get("trade", {}).get("symbol", "BTC-USDT")
    interval = config.get("trade", {}).get("interval", "1H")
    
    print("1. 获取相同的数据集...")
    # 使用相同的时间范围进行公平比较
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
    data = make_supervised(dfm, horizon=1, threshold=0.005)
    
    # 数据清洗
    non_feature_cols = ["ts", "dt", "y", "future_high", "future_low", "future_close", "future_ret", "date", "timestamp", "vol_ccy", "vol_ccy_quote", "confirm"]
    feature_cols = [c for c in data.columns if c not in non_feature_cols]
    
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(subset=feature_cols + ["y"], inplace=True)
    
    print(f"   原始特征数量: {len(feature_cols)}")
    print(f"   清洗后样本数: {len(data)}")
    print(f"   正样本比例: {data['y'].mean():.2%}")
    print()
    
    # 特征预选择 - 确保两种模型使用相同的特征集
    print("4. 特征预选择 (确保公平比较)...")
    top_k_features = 20  # 与改进模型相同的特征数量
    if top_k_features and top_k_features > 0 and top_k_features < len(feature_cols):
        print(f"   选择前 {top_k_features} 个最重要特征...")
        X = data[feature_cols]
        y = data["y"]
        
        selector = SelectKBest(f_classif, k=top_k_features)
        selector.fit(X, y)
        
        selected_mask = selector.get_support()
        new_feature_cols = X.columns[selected_mask]
        
        print(f"   选择的特征: {new_feature_cols.tolist()}")
        feature_cols = new_feature_cols.tolist()
    else:
        print("   使用所有特征")
    
    print(f"   最终特征数量: {len(feature_cols)}")
    print()
    
    # 分割数据 - 确保两种模型使用相同的数据分割
    print("5. 数据分割...")
    val_size = max(1, int(len(data) * 0.1))  # 10% 用于验证
    train_data = data.iloc[:-val_size]
    val_data = data.iloc[-val_size:]
    
    X_train = train_data[feature_cols]
    y_train = train_data["y"]
    X_val = val_data[feature_cols]
    y_val = val_data["y"]
    
    print(f"   训练集样本数: {len(train_data)}")
    print(f"   验证集样本数: {len(val_data)}")
    print()
    
    # 加载两种模型进行公平比较
    print("6. 加载模型进行公平比较...")
    model_dir = config["paths"]["model_dir"]
    
    improved_model_path = os.path.join(model_dir, "improved_best_model.pkl")
    original_model_path = os.path.join(model_dir, "best_model.pkl")
    
    if os.path.exists(improved_model_path) and os.path.exists(original_model_path):
        from joblib import load
        from utils.backtest import run_backtest
        
        try:
            # 加载模型
            improved_model = load(improved_model_path)
            original_model = load(original_model_path)
            
            print("   [OK] 模型加载成功")
            
            # 在相同数据上进行预测和回测
            print("7. 在相同数据上进行公平比较...")
            
            # 改进模型预测
            print("   改进模型表现:")
            improved_predictions = pd.Series(improved_model.predict(X_val), index=X_val.index)
            if hasattr(improved_model, 'predict_proba'):
                improved_probabilities = improved_model.predict_proba(X_val)[:, 1]
            else:
                improved_probabilities = np.ones(len(X_val)) * 0.5
            
            # 从元数据中获取置信度阈值
            improved_metadata_path = os.path.join(model_dir, "improved_metadata.json")
            if os.path.exists(improved_metadata_path):
                import json
                with open(improved_metadata_path, 'r', encoding='utf-8') as f:
                    improved_meta = json.load(f)
                improved_confidence_threshold = improved_meta['best_params'].get('confidence_threshold', 0.95)
                print(f"     置信度阈值: {improved_confidence_threshold:.4f}")
            else:
                improved_confidence_threshold = 0.95
            
            # 执行回测
            improved_total_ret, improved_max_dd, improved_success_rate, improved_trade_count, improved_returns_series = run_backtest(
                predictions=improved_predictions,
                probabilities=improved_probabilities,
                test_data=val_data,
                confidence_threshold=improved_confidence_threshold,
                stop_loss_pct=0.02,
                take_profit_pct=0.05,
            )
            
            print(f"     总收益率: {improved_total_ret:.4f}")
            if improved_returns_series.std() != 0:
                improved_sharpe = (improved_returns_series.mean() / improved_returns_series.std()) * np.sqrt(252 * 24)
                print(f"     夏普比率: {improved_sharpe:.4f}")
            else:
                improved_sharpe = 0.0
                print(f"     夏普比率: 0.0000")
            print(f"     成功率: {improved_success_rate:.2%}")
            print(f"     交易次数: {improved_trade_count}")
            print()
            
            # 原始模型预测
            print("   原始模型表现:")
            original_predictions = pd.Series(original_model.predict(X_val), index=X_val.index)
            if hasattr(original_model, 'predict_proba'):
                original_probabilities = original_model.predict_proba(X_val)[:, 1]
            else:
                original_probabilities = np.ones(len(X_val)) * 0.5
            
            # 从元数据中获取置信度阈值
            original_metadata_path = os.path.join(model_dir, "metadata.json")
            if os.path.exists(original_metadata_path):
                import json
                with open(original_metadata_path, 'r', encoding='utf-8') as f:
                    original_meta = json.load(f)
                original_confidence_threshold = original_meta['best_params'].get('confidence_threshold', 0.95)
                print(f"     置信度阈值: {original_confidence_threshold:.4f}")
            else:
                original_confidence_threshold = 0.95
            
            # 执行回测
            original_total_ret, original_max_dd, original_success_rate, original_trade_count, original_returns_series = run_backtest(
                predictions=original_predictions,
                probabilities=original_probabilities,
                test_data=val_data,
                confidence_threshold=original_confidence_threshold,
                stop_loss_pct=0.02,
                take_profit_pct=0.05,
            )
            
            print(f"     总收益率: {original_total_ret:.4f}")
            if original_returns_series.std() != 0:
                original_sharpe = (original_returns_series.mean() / original_returns_series.std()) * np.sqrt(252 * 24)
                print(f"     夏普比率: {original_sharpe:.4f}")
            else:
                original_sharpe = 0.0
                print(f"     夏普比率: 0.0000")
            print(f"     成功率: {original_success_rate:.2%}")
            print(f"     交易次数: {original_trade_count}")
            print()
            
            # 性能对比
            print("8. 性能对比:")
            print(f"   总收益率差异: {improved_total_ret - original_total_ret:+.4f}")
            print(f"   夏普比率差异: {improved_sharpe - original_sharpe:+.4f}")
            print(f"   成功率差异: {improved_success_rate - original_success_rate:+.2%}")
            print(f"   交易次数差异: {improved_trade_count - original_trade_count:+.1f}")
            
            if improved_total_ret > original_total_ret:
                print("   [OK] 改进模型在总收益率上表现更好")
            elif improved_total_ret < original_total_ret:
                print("   [ERROR] 原始模型在总收益率上表现更好")
            else:
                print("   [INFO] 两者总收益率相同")
                
            if improved_sharpe > original_sharpe:
                print("   [OK] 改进模型在风险调整收益上表现更好")
            elif improved_sharpe < original_sharpe:
                print("   [ERROR] 原始模型在风险调整收益上表现更好")
            else:
                print("   [INFO] 两者风险调整收益相同")
            
        except Exception as e:
            print(f"   [ERROR] 模型比较失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("   [WARNING] 模型文件不存在，无法进行直接比较")
    
    print()
    print("9. 结论:")
    print("   通过在完全相同的数据集和条件下进行测试，我们可以得到真实的性能比较。")
    print("   之前的0.1到0.97的跃升很可能是由以下因素造成的:")
    print("   1. 评分公式的修改 (减少了标准差惩罚)")
    print("   2. 使用了不同的数据集")
    print("   3. 特征集不一致")
    print("   4. 验证方法不一致")
    print()
    print("   真正的模型改进应该体现在:")
    print("   - 在相同条件下更好的实盘表现")
    print("   - 更稳定的收益曲线")
    print("   - 更好的风险控制")
    print("   - 更高的夏普比率")

if __name__ == "__main__":
    genuine_model_comparison()