"""
测试套利策略
"""
import unittest
import pandas as pd
import numpy as np
from strategies.arbitrage_strategy import ArbitrageStrategy


class TestArbitrageStrategy(unittest.TestCase):
    
    def setUp(self):
        """设置测试环境"""
        self.config = {
            'spread_threshold': 0.005,
            'zscore_threshold': 2.0,
            'min_volume': 1000,
            'spread_lookback': 20
        }
        self.strategy = ArbitrageStrategy("TestArbitrage", self.config)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.strategy.spread_threshold, 0.005)
        self.assertEqual(self.strategy.zscore_threshold, 2.0)
        self.assertEqual(self.strategy.min_volume, 1000)
        self.assertEqual(self.strategy.spread_lookback, 20)
    
    def test_calculate_zscore(self):
        """测试Z-score计算"""
        data = pd.Series([1, 2, 3, 4, 5, 4, 3, 2, 1, 2] * 10)  # 重复以获得足够的数据点
        zscore = self.strategy.calculate_zscore(data, window=5)
        
        self.assertEqual(len(zscore), len(data))
        # 检查是否有有效的Z-score值（不是全部NaN）
        self.assertFalse(zscore.isna().all())
    
    def test_calculate_spread(self):
        """测试价差计算"""
        price_a = pd.Series([100, 101, 102, 103, 104])
        price_b = pd.Series([99, 100, 101, 102, 103])
        
        spread = self.strategy.calculate_spread(price_a, price_b)
        
        # 价差应该是百分比形式，且有合理的值
        self.assertEqual(len(spread), 5)
        # 验证价差计算逻辑：(A-B)/((A+B)/2)
        expected_first = (100 - 99) / ((100 + 99) / 2)
        self.assertAlmostEqual(spread.iloc[0], expected_first, places=6)
    
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
        
        self.assertEqual(info['name'], "TestArbitrage")
        self.assertEqual(info['type'], 'arbitrage')
        self.assertIn('parameters', info)
        self.assertIn('description', info)


if __name__ == '__main__':
    unittest.main()