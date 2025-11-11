# analysis/data_comparison.py
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

def compare_training_data():
    """比较两种模型使用的训练数据差异"""
    print("=== 训练数据对比分析 ===")
    
    # 获取客户端
    client = OKXClient(**config["okx"])
    symbol = config.get("trade", {}).get("symbol", "BTC-USDT")
    interval = config.get("trade", {}).get("interval", "1H")
    
    print("1. 原始模型数据获取:")
    # 原始模型使用3年数据
    dfp_orig = get_klines_bian(client, symbol, interval, years=3, ignore_local=True)
    if dfp_orig is not None and not dfp_orig.empty:
        dfp_orig = normalize_binance_df(dfp_orig)
        print(f"   原始数据时间范围: {dfp_orig.index.min()} → {dfp_orig.index.max()}")
        print(f"   原始数据样本数: {len(dfp_orig)}")
        print(f"   原始数据价格范围: ${dfp_orig['close'].min():.2f} → ${dfp_orig['close'].max():.2f}")
        print()
    
    print("2. 改进模型数据获取:")
    # 改进模型使用2年数据
    dfp_improved = get_klines_bian(client, symbol, interval, years=2, ignore_local=True)
    if dfp_improved is not None and not dfp_improved.empty:
        dfp_improved = normalize_binance_df(dfp_improved)
        print(f"   改进数据时间范围: {dfp_improved.index.min()} → {dfp_improved.index.max()}")
        print(f"   改进数据样本数: {len(dfp_improved)}")
        print(f"   改进数据价格范围: ${dfp_improved['close'].min():.2f} → ${dfp_improved['close'].max():.2f}")
        print()
    
    # 比较数据覆盖范围
    if dfp_orig is not None and dfp_improved is not None:
        print("3. 数据覆盖范围对比:")
        orig_start, orig_end = dfp_orig.index.min(), dfp_orig.index.max()
        imp_start, imp_end = dfp_improved.index.min(), dfp_improved.index.max()
        
        print(f"   时间范围差异:")
        print(f"     原始模型: {orig_start} → {orig_end}")
        print(f"     改进模型: {imp_start} → {imp_end}")
        print(f"     重叠期: {max(orig_start, imp_start)} → {min(orig_end, imp_end)}")
        print()
        
        # 计算重叠期数据
        overlap_start = max(orig_start, imp_start)
        overlap_end = min(orig_end, imp_end)
        
        if overlap_start < overlap_end:
            dfp_orig_overlap = dfp_orig[(dfp_orig.index >= overlap_start) & (dfp_orig.index <= overlap_end)]
            dfp_imp_overlap = dfp_improved[(dfp_improved.index >= overlap_start) & (dfp_improved.index <= overlap_end)]
            
            print(f"   重叠期数据分析:")
            print(f"     重叠期样本数: 原始={len(dfp_orig_overlap)}, 改进={len(dfp_imp_overlap)}")
            print(f"     重叠期价格相关性: {dfp_orig_overlap['close'].corr(dfp_imp_overlap['close']):.6f}")
            print()
        
        # 检查是否是市场环境差异
        print("4. 市场环境分析:")
        if len(dfp_orig) > 100 and len(dfp_improved) > 100:
            # 计算波动率
            orig_returns = dfp_orig['close'].pct_change().dropna()
            imp_returns = dfp_improved['close'].pct_change().dropna()
            
            orig_volatility = orig_returns.std() * np.sqrt(365 * 24)  # 年化波动率
            imp_volatility = imp_returns.std() * np.sqrt(365 * 24)    # 年化波动率
            
            print(f"   年化波动率:")
            print(f"     原始数据期间: {orig_volatility:.4f}")
            print(f"     改进数据期间: {imp_volatility:.4f}")
            print()
            
            # 检查趋势
            orig_trend = (dfp_orig['close'].iloc[-1] / dfp_orig['close'].iloc[0]) - 1
            imp_trend = (dfp_improved['close'].iloc[-1] / dfp_improved['close'].iloc[0]) - 1
            
            print(f"   期间累计收益:")
            print(f"     原始数据期间: {orig_trend:.2%}")
            print(f"     改进数据期间: {imp_trend:.2%}")
            print()
            
    print("5. 可能的问题:")
    print("   [WARNING] 使用不同时间范围的数据可能导致不公平比较")
    print("   [WARNING] 不同的市场环境可能导致模型表现差异")
    print("   [WARNING] 需要在相同数据集上验证模型性能")

if __name__ == "__main__":
    compare_training_data()