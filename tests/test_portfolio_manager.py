"""
测试投资组合管理模块
"""
import unittest
import pandas as pd
import numpy as np
from trader.portfolio_manager import PortfolioManager, get_portfolio_manager


class TestPortfolioManager(unittest.TestCase):
    
    def setUp(self):
        """设置测试环境"""
        self.config = {
            'multi_asset': {
                'symbols': ['BTC-USDT', 'ETH-USDT'],
                'allocation_strategy': 'equal',
                'max_assets': 5,
                'rebalance_frequency_days': 7,
                'per_asset_risk_limit': 0.10,
                'overall_risk_limit': 0.15
            }
        }
        self.portfolio_manager = PortfolioManager(self.config)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.portfolio_manager.allocation_strategy, 'equal')
        self.assertEqual(self.portfolio_manager.max_assets, 5)
        self.assertEqual(self.portfolio_manager.rebalance_frequency_days, 7)
        self.assertIn('BTC-USDT', self.portfolio_manager.asset_data)
        self.assertIn('ETH-USDT', self.portfolio_manager.asset_data)
    
    def test_calculate_target_allocations_equal(self):
        """测试等权重分配"""
        market_data = {
            'BTC-USDT': pd.DataFrame({'close': [50000, 50010, 50020]}),
            'ETH-USDT': pd.DataFrame({'close': [3000, 3010, 3020]})
        }
        
        allocations = self.portfolio_manager.calculate_target_allocations(market_data)
        
        # 应该是等权重分配
        expected_weight = 1.0 / 2  # 两个资产
        self.assertEqual(allocations['BTC-USDT'], expected_weight)
        self.assertEqual(allocations['ETH-USDT'], expected_weight)
    
    def test_should_rebalance(self):
        """测试是否需要再平衡"""
        # 刚启动不需要再平衡
        self.assertFalse(self.portfolio_manager.should_rebalance())
        
        # 修改最后再平衡时间为一周前
        import datetime
        self.portfolio_manager.last_rebalance_time = \
            datetime.datetime.utcnow() - datetime.timedelta(days=8)
        
        # 现在应该需要再平衡
        self.assertTrue(self.portfolio_manager.should_rebalance())
    
    def test_update_position(self):
        """测试更新持仓"""
        self.portfolio_manager.update_position('BTC-USDT', 0.5, 50000)
        
        btc_data = self.portfolio_manager.asset_data['BTC-USDT']
        self.assertEqual(btc_data['position'], 0.5)
        self.assertEqual(btc_data['value'], 25000)  # 0.5 * 50000
    
    def test_get_portfolio_summary(self):
        """测试投资组合摘要"""
        # 更新一些持仓
        self.portfolio_manager.update_position('BTC-USDT', 0.5, 50000)
        self.portfolio_manager.update_position('ETH-USDT', 2.0, 3000)
        
        summary = self.portfolio_manager.get_portfolio_summary()
        
        self.assertEqual(summary['total_value'], 31000)  # 25000 + 6000
        self.assertIn('BTC-USDT', summary['assets'])
        self.assertIn('ETH-USDT', summary['assets'])
    
    def test_check_risk_limits(self):
        """测试风险限制检查"""
        # 更新持仓到超过风险限制
        self.portfolio_manager.update_position('BTC-USDT', 100, 50000)  # 大量BTC
        self.portfolio_manager.update_position('ETH-USDT', 1, 3000)      # 少量ETH
        
        is_over_risk, warnings = self.portfolio_manager.check_risk_limits()
        
        # BTC的权重远超风险限制
        self.assertTrue(is_over_risk)
        self.assertGreater(len(warnings), 0)


if __name__ == '__main__':
    import datetime  # 为了测试需要导入
    unittest.main()