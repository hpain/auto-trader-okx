# analysis/market_regime_detection.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import config
from data.binance import get_klines_bian
from exchange.okx_exchange import OKXExchange
from utils.data_normalization import normalize_binance_df
import ta

import warnings
warnings.filterwarnings('ignore')

def detect_regime_change(df: pd.DataFrame, split_ratio: float = 0.5) -> dict:
    """
    分析给定的市场数据，检测市场模式是否发生显著变化。

    :param df: 包含OHLCV和'returns'列的DataFrame。
    :param split_ratio: 用于分割早期和近期数据的比例。
    :return: 一个包含分析结果的字典。
    """
    if df is None or df.empty or len(df) < 120: # 至少需要120个数据点来保证分析的可靠性
        return {"regime_changed": False, "reason": "Not enough data", "details": {}}

    # 计算所需指标
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=20).std()

    # 分割时段
    mid_point = int(len(df) * split_ratio)
    early_period = df.iloc[:mid_point]
    late_period = df.iloc[mid_point:]

    if early_period.empty or late_period.empty or len(early_period) < 20 or len(late_period) < 20:
        return {"regime_changed": False, "reason": "Not enough data after split for reliable calculation", "details": {}}

    # 计算两个时段的统计数据
    early_mean_ret = early_period['returns'].mean()
    late_mean_ret = late_period['returns'].mean()
    early_std_ret = early_period['returns'].std()
    late_std_ret = late_period['returns'].std()
    early_mean_vol = early_period['volatility'].dropna().mean()
    late_mean_vol = late_period['volatility'].dropna().mean()

    # 计算变化
    mean_change = abs(late_mean_ret - early_mean_ret)
    vol_change = abs(late_mean_vol - early_mean_vol)
    trend_change = abs(late_std_ret - early_std_ret)

    # 定义阈值
    mean_threshold = 0.0002  # 稍微放宽以避免在稳定市场中误报
    vol_threshold = 0.001
    trend_threshold = 0.001

    # 判断哪个变化最显著
    changes = {
        "Mean return": mean_change / mean_threshold if mean_threshold > 0 else 0,
        "Volatility (rolling mean)": vol_change / vol_threshold if vol_threshold > 0 else 0,
        "Volatility (period std)": trend_change / trend_threshold if trend_threshold > 0 else 0,
    }

    # 找到变化最大的那个
    most_significant_change_key = max(changes, key=changes.get)
    most_significant_change_value = changes[most_significant_change_key]

    regime_changed = False
    reason = "No significant change detected."

    if most_significant_change_value > 1.0:
        regime_changed = True
        # 构建更详细的原因
        if most_significant_change_key == "Mean return":
            reason = f"Mean return change ({mean_change:.6f}) exceeded threshold ({mean_threshold})."
        elif most_significant_change_key == "Volatility (rolling mean)":
            reason = f"Volatility (rolling mean) change ({vol_change:.6f}) exceeded threshold ({vol_threshold})."
        elif most_significant_change_key == "Volatility (period std)":
            reason = f"Volatility (period std) change ({trend_change:.6f}) exceeded threshold ({trend_threshold})."


    return {
        "regime_changed": regime_changed,
        "reason": reason,
        "details": {
            "mean_return_change": mean_change,
            "volatility_change": vol_change,
            "trend_change": trend_change,
            "thresholds": {
                "mean": mean_threshold,
                "volatility": vol_threshold,
                "trend": trend_threshold
            }
        }
    }

def analyze_market_regime():
    """
    用于独立运行的分析函数，获取数据、进行分析、打印结果并绘图。
    """
    print("=== 市场模式对比分析 ===")
    
    # 加载数据
    # 注意：这里为了演示，仍然保留了数据获取部分
    client = OKXExchange(
        api_key=config["okx"].get("api_key"),
        api_secret=config["okx"].get("secret_key"),
        passphrase=config["okx"].get("passphrase"),
        sandbox=config["okx"].get("flag") == '0'
    )
    symbol = config.get("trade", {}).get("symbol", "BTC-USDT")
    interval = config.get("trade", {}).get("interval", "1H")

    df = get_klines_bian(client, symbol, interval, years=2)
    
    if df is None or df.empty:
        print("无法获取数据，请检查网络连接和API设置")
        return
        
    df = normalize_binance_df(df)

    # 调用新的核心函数进行分析
    regime_result = detect_regime_change(df.copy()) # 传递副本以防修改

    # 打印结果
    print(f"\n--- 分析结果 ---")
    if regime_result['regime_changed']:
        print(f"⚠️  检测到市场模式可能发生变化！")
        print(f"   原因: {regime_result['reason']}")
    else:
        print(f"✅ 未检测到明显的市场模式变化。")
    
    print("\n--- 详细指标 ---")
    for key, value in regime_result['details'].items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value:.6f}")
    
    # 可视化部分
    print("\n--- 生成可视化图表 ---")
    df['volatility'] = df['returns'].rolling(window=20).std() # 重新计算以确保存在
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    axes[0].plot(df.index, df['close'], label='Close Price', alpha=0.7)
    axes[0].set_title('Price Over Time')
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(df.index, df['volatility'], label='20-period Volatility', color='orange')
    axes[1].set_title('Volatility Over Time')
    axes[1].legend()
    axes[1].grid(True)
    returns_rolling = df['returns'].rolling(window=30).mean()
    axes[2].plot(df.index, returns_rolling, label='30-period Average Returns', color='green')
    axes[2].set_title('Rolling Average Returns Over Time')
    axes[2].legend()
    axes[2].grid(True)
    plt.tight_layout()
    plt.savefig('market_regime_analysis.png')
    # plt.show() # 在自动化脚本中通常不显示图像
    print("可视化图表已保存为 'market_regime_analysis.png'")

if __name__ == "__main__":
    analyze_market_regime()
