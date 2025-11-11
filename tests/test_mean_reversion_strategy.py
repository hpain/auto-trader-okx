"""
测试均值回归策略
"""
import unittest
import pandas as pd
import numpy as np
from strategies.mean_reversion_strategy import MeanReversionStrategy


class TestMeanReversionStrategy(unittest.TestCase):
    
    def setUp(self):
        """设置测试环境"""
        self.config = {
            'short_window': 5,
            'long_window': 10,
            'zscore_threshold': 1.0,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70
        }
        self.strategy = MeanReversionStrategy("TestMeanReversion", self.config)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.strategy.short_window, 5)
        self.assertEqual(self.strategy.long_window, 10)
        self.assertEqual(self.strategy.zscore_threshold, 1.0)
        self.assertEqual(self.strategy.rsi_period, 14)
    
    def test_calculate_rsi(self):
        """测试RSI计算"""
        # 创建一些价格数据
        prices = pd.Series([100, 102, 101, 99, 103, 105, 104, 102, 100, 98])
        rsi = self.strategy.calculate_rsi(prices, period=5)
        
        # RSI应该在0-100之间，或者为NaN（前几个值由于数据不足会是NaN）
        # 检查非NaN值是否在范围内
        non_nan_values = rsi.dropna()
        if len(non_nan_values) > 0:
            self.assertTrue((non_nan_values >= 0).all())
            self.assertTrue((non_nan_values <= 100).all())
    
    def test_calculate_zscore(self):
        """测试Z-score计算"""
        # 创建一些价格数据
        prices = pd.Series([100, 102, 101, 99, 103, 105, 104, 102, 100, 98] * 10)  # 重复以获得足够的数据点
        zscore = self.strategy.calculate_zscore(prices, window=10)
        
        # Z-score可以是任意值，但我们检查是否计算成功
        self.assertEqual(len(zscore), len(prices))
        self.assertFalse(zscore.isna().all())  # 不应该全是NaN
    
    def test_generate_signals_basic(self):
        """测试基本信号生成"""
        # 创建测试数据
        data = pd.DataFrame({
            'open': [100, 101, 102, 99, 98],
            'high': [103, 104, 105, 101, 100],
            'low': [98, 99, 100, 97, 96],
            'close': [102, 103, 101, 98, 97],
            'volume': [1000, 1200, 800, 1500, 1100]
        })
        
        signals = self.strategy.generate_signals(data)
        
        # 检查输出格式
        self.assertIn('signal', signals.columns)
        self.assertIn('position', signals.columns)
        self.assertEqual(len(signals), len(data))
        
        # 信号值应该是-1, 0, 或1
        valid_signals = set([-1, 0, 1])
        actual_signals = set(signals['signal'].unique())
        self.assertTrue(actual_signals.issubset(valid_signals))
    
    def test_strategy_info(self):
        """测试策略信息获取"""
        info = self.strategy.get_strategy_info()
        
        self.assertEqual(info['name'], "TestMeanReversion")
        self.assertEqual(info['type'], 'mean_reversion')
        self.assertIn('parameters', info)
        self.assertIn('description', info)


if __name__ == '__main__':
    unittest.main()