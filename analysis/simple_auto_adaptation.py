# analysis/simple_auto_adaptation.py
import pandas as pd
import numpy as np
import sys
import os
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from data.binance import get_klines_bian
from trader.okx_client import OKXClient
from utils.data_normalization import normalize_binance_df

class SimpleAutoAdaptation:
    def __init__(self):
        """初始化简单自适应系统"""
        self.config = config
        self.model_dir = config["paths"]["model_dir"]
        self.performance_history_file = os.path.join(self.model_dir, "performance_history.json")
        
    def detect_regime_change(self, df, window_size=60):
        """
        检测市场模式变化
        """
        # 计算滚动统计量
        df['returns'] = df['close'].pct_change()
        
        # 使用窗口对比，检查统计特性是否有显著变化
        mid_point = len(df) // 2
        early_period = df.iloc[:mid_point]['returns'].dropna()
        late_period = df.iloc[mid_point:]['returns'].dropna()
        
        if len(early_period) < 10 or len(late_period) < 10:
            return False, 0.0
        
        # 检查多个指标
        mean_change = abs(late_period.mean() - early_period.mean()) / (abs(early_period.mean()) + 1e-8)
        std_change = abs(late_period.std() - early_period.std()) / (early_period.std() + 1e-8)
        skew_change = abs(late_period.skew() - early_period.skew())
        kurt_change = abs(late_period.kurtosis() - early_period.kurtosis())
        
        # 综合判断
        change_score = (mean_change + std_change + skew_change/10 + kurt_change/10) / 4
        
        # 设定阈值判断是否有显著变化
        has_change = change_score > 0.1
        
        return has_change, change_score
    
    def suggest_parameters(self, has_change, change_score):
        """
        根据市场模式变化建议参数
        """
        if not has_change:
            # 默认保守参数
            return {
                'profit_threshold': 0.005,
                'confidence_threshold': 0.60,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.05,
                'n_trials': 50,
                'use_recent_data': False
            }
        
        # 根据变化程度调整参数
        # 变化越大，越保守，使用更近期的数据
        confidence_threshold = max(0.55, 0.65 - change_score * 0.2)
        profit_threshold = max(0.003, 0.005 - change_score * 0.001)
        
        # 增加试验次数以找到更优参数
        n_trials = min(100, 50 + int(change_score * 30))
        
        return {
            'profit_threshold': profit_threshold,
            'confidence_threshold': confidence_threshold,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.05,
            'n_trials': n_trials,
            'use_recent_data': True  # 使用更近期的数据
        }
    
    def print_recommendations(self, has_change, change_score, params):
        """
        打印推荐参数
        """
        print("=== 市场模式变化检测结果 ===")
        if has_change:
            print(f"[OK] 检测到市场模式变化")
            print(f"  变化得分: {change_score:.3f}")
        else:
            print("[NO] 未检测到显著市场模式变化")
        
        print("\n=== 推荐参数 ===")
        print(f"盈利阈值: {params['profit_threshold']:.4f}")
        print(f"置信度阈值: {params['confidence_threshold']:.3f}")
        print(f"止损百分比: {params['stop_loss_pct']:.3f}")
        print(f"止盈百分比: {params['take_profit_pct']:.3f}")
        print(f"优化试验次数: {params['n_trials']}")
        print(f"使用近期数据: {'是' if params['use_recent_data'] else '否'}")
        
        if params['use_recent_data']:
            print("\n建议:")
            print("- 使用更近期的数据进行训练 (最近1年而不是2年)")
            print("- 增加模型正则化以避免过拟合")
            print("- 降低置信度阈值以增加交易机会")
            print("- 增加试验次数以找到更优参数")
        else:
            print("\n建议:")
            print("- 可以使用完整历史数据进行训练")
            print("- 保持当前参数设置")
            print("- 定期监测市场模式变化")
    
    def run_simple_adaptation(self):
        """
        执行简单的自适应流程
        """
        print("=== 简单自适应系统启动 ===")
        
        # 加载数据
        symbol = config.get("trade", {}).get("symbol", "BTC-USDT")
        interval = config.get("trade", {}).get("interval", "1H")
        
        # 获取最近2年的数据
        client = OKXClient(**config["okx"])
        df = get_klines_bian(client, symbol, interval, years=2, ignore_local=True)
        
        if df is None or df.empty:
            print("无法加载数据，退出自适应流程")
            return
        
        df = normalize_binance_df(df)
        
        # 检测市场模式变化
        print("正在检测市场模式变化...")
        has_change, change_score = self.detect_regime_change(df)
        
        # 获取建议参数
        params = self.suggest_parameters(has_change, change_score)
        
        # 打印推荐
        self.print_recommendations(has_change, change_score, params)
        
        print("\n=== 自适应系统完成 ===")

if __name__ == "__main__":
    auto_system = SimpleAutoAdaptation()
    auto_system.run_simple_adaptation()