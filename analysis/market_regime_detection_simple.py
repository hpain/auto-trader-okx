# analysis/market_regime_detection_simple.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config

def analyze_market_regime_simple():
    """使用已有的缓存数据进行市场模式分析"""
    print("=== 市场模式对比分析 ===")
    
    # 检查是否有缓存的特征数据
    cache_dir = config["paths"]["feature_cache_dir"]
    import glob
    
    # 获取所有的特征缓存文件
    cache_files = glob.glob(os.path.join(cache_dir, "features_*.parquet"))
    
    if not cache_files:
        print("未找到特征缓存文件，请先运行训练脚本生成数据")
        return
    
    # 使用最新的特征文件
    latest_file = max(cache_files, key=os.path.getctime)
    print(f"使用特征文件: {os.path.basename(latest_file)}")
    
    # 读取特征数据
    df = pd.read_parquet(latest_file)
    
    # 检查是否包含价格数据
    if 'close' not in df.columns:
        print("特征数据中不包含价格信息，无法进行市场模式分析")
        return
    
    # 移除可能存在的无穷大值和 NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['close'])
    
    if len(df) < 100:  # 确保有足够的数据点
        print("数据点太少，无法进行有意义的分析")
        return
    
    # 计算技术指标用于分析
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=20).std()
    df['high_low_pct'] = (df['high'] - df['low']) / df['close'] if 'high' in df.columns and 'low' in df.columns else pd.Series([0] * len(df), index=df.index)
    
    # 分时段统计
    mid_point = len(df) // 2
    early_period = df.iloc[:mid_point]
    late_period = df.iloc[mid_point:]
    
    print(f"数据总长度: {len(df)}")
    print()
    
    print("早期时段 (前50%):")
    if len(early_period) > 0:
        print(f"  日期范围: {early_period.index[0]} 到 {early_period.index[-1]}")
        returns_mean = early_period['returns'].mean()
        returns_std = early_period['returns'].std()
        volatility_mean = early_period['volatility'].dropna().mean()
        print(f"  平均收益率: {returns_mean:.6f}")
        print(f"  收益率标准差: {returns_std:.6f}")
        print(f"  平均波动率: {volatility_mean:.6f}")
        early_direction_changes = len(early_period[early_period['returns'] * early_period['returns'].shift(1) < 0]) / len(early_period) if len(early_period['returns'].dropna()) > 1 else 0
        print(f"  价格方向变化率: {early_direction_changes:.2%}")
    else:
        print("  无数据")
    print()
    
    print("近期时段 (后50%):")
    if len(late_period) > 0:
        print(f"  日期范围: {late_period.index[0]} 到 {late_period.index[-1]}")
        returns_mean = late_period['returns'].mean()
        returns_std = late_period['returns'].std()
        volatility_mean = late_period['volatility'].dropna().mean()
        print(f"  平均收益率: {returns_mean:.6f}")
        print(f"  收益率标准差: {returns_std:.6f}")
        print(f"  平均波动率: {volatility_mean:.6f}")
        late_direction_changes = len(late_period[late_period['returns'] * late_period['returns'].shift(1) < 0]) / len(late_period) if len(late_period['returns'].dropna()) > 1 else 0
        print(f"  价格方向变化率: {late_direction_changes:.2%}")
    else:
        print("  无数据")
    print()
    
    # 市场模式变化的指标
    if len(early_period) > 0 and len(late_period) > 0:
        mean_change = abs(late_period['returns'].mean() - early_period['returns'].mean())
        early_vol_mean = early_period['volatility'].dropna().mean()
        late_vol_mean = late_period['volatility'].dropna().mean()
        vol_change = abs(late_vol_mean - early_vol_mean) if not (pd.isna(early_vol_mean) or pd.isna(late_vol_mean)) else 0
        trend_change = abs(late_period['returns'].std() - early_period['returns'].std())
        
        print("=== 市场模式变化指标 ===")
        print(f"收益率均值变化: {mean_change:.6f}")
        print(f"波动率变化: {vol_change:.6f}")
        print(f"收益率标准差变化: {trend_change:.6f}")
        
        if mean_change > 0.0001 or vol_change > 0.001 or trend_change > 0.001:
            print()
            print("警告: 检测到市场模式可能发生变化！")
            print("  - 建议使用更近的历史数据进行训练")
            print("  - 考虑实施滚动训练机制")
            print("  - 重新评估特征重要性")
        else:
            print()
            print("OK: 未检测到明显的市场模式变化")
        
        # 更详细的分析
        print("\n=== 详细分析 ===")
        early_skew = early_period['returns'].dropna().skew()
        late_skew = late_period['returns'].dropna().skew()
        early_kurt = early_period['returns'].dropna().kurtosis()
        late_kurt = late_period['returns'].dropna().kurtosis()
        print(f"早期收益率偏度: {early_skew:.4f}")
        print(f"近期收益率偏度: {late_skew:.4f}")
        print(f"早期收益率峰度: {early_kurt:.4f}")
        print(f"近期收益率峰度: {late_kurt:.4f}")

    # 创建可视化
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))

    # 价格走势图
    axes[0].plot(df.index, df['close'], label='Close Price', alpha=0.7)
    axes[0].set_title('Price Over Time')
    axes[0].legend()
    axes[0].grid(True)

    # 波动率变化
    if 'volatility' in df.columns:
        axes[1].plot(df.index, df['volatility'], label='20-period Volatility', color='orange')
        axes[1].set_title('Volatility Over Time')
        axes[1].legend()
        axes[1].grid(True)

    # 收益率变化
    if 'returns' in df.columns:
        returns_rolling = df['returns'].rolling(window=30).mean()
        axes[2].plot(df.index, returns_rolling, label='30-period Average Returns', color='green')
        axes[2].set_title('Rolling Average Returns Over Time')
        axes[2].legend()
        axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('market_regime_analysis.png')
    plt.show()
    
    print("\n可视化图表已保存为 'market_regime_analysis.png'")

if __name__ == "__main__":
    analyze_market_regime_simple()