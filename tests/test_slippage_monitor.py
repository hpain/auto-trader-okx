"""
测试滑点监控模块
"""
import unittest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np
from trader.slippage_monitor import SlippageMonitor, get_slippage_monitor


class MockExchange:
    """模拟交易所接口"""
    def get_ticker(self, symbol):
        return {
            'quote_volume_24h': 2000000  # 足够的交易量
        }
    
    def get_orderbook(self, symbol, depth=1):
        return {
            'bids': [[50000, 10]],  # 买一价
            'asks': [[50010, 10]]   # 卖一价
        }


class TestSlippageMonitor(unittest.TestCase):
    
    def setUp(self):
        """设置测试环境"""
        self.config = {
            'slippage_monitoring': {
                'max_slippage_pct': 0.02,  # 2%
                'volume_threshold': 1000000,  # 100万
                'bid_ask_spread_threshold': 0.005  # 0.5%
            }
        }
        self.slippage_monitor = SlippageMonitor(self.config)
    
    def test_calculate_slippage_buy(self):
        """测试买入时的滑点计算"""
        expected_price = 50000
        executed_price = 50050  # 更高，对买入是不利滑点
        side = 'buy'
        
        slippage = self.slippage_monitor.calculate_slippage(expected_price, executed_price, side)
        
        # (50050 - 50000) / 50000 = 0.001 (只有不利滑点)
        self.assertEqual(slippage, 0.001)
    
    def test_calculate_slippage_sell(self):
        """测试卖出时的滑点计算"""
        expected_price = 50000
        executed_price = 49950  # 更低，对卖出是不利滑点
        side = 'sell'
        
        slippage = self.slippage_monitor.calculate_slippage(expected_price, executed_price, side)
        
        # (50000 - 49950) / 50000 = 0.001 (只有不利滑点)
        self.assertEqual(slippage, 0.001)
    
    def test_is_excessive_slippage(self):
        """测试是否过度滑点"""
        # 滑点在阈值内
        is_excessive = self.slippage_monitor.is_excessive_slippage(50000, 50050, 'buy')
        self.assertFalse(is_excessive)
        
        # 滑点超出阈值
        is_excessive = self.slippage_monitor.is_excessive_slippage(50000, 60000, 'buy')
        self.assertTrue(is_excessive)
    
    def test_check_liquidity_conditions_ok(self):
        """测试流动性条件检查 - 正常情况"""
        exchange_mock = MockExchange()
        
        is_ok, msg = self.slippage_monitor.check_liquidity_conditions('BTC-USDT', exchange_mock)
        
        self.assertTrue(is_ok)
        self.assertIn('Liquidity OK', msg)
    
    def test_monitor_trade_execution(self):
        """测试交易执行监控"""
        exchange_mock = MockExchange()
        
        result = self.slippage_monitor.monitor_trade_execution(
            symbol='BTC-USDT',
            side='buy',
            quantity=0.1,
            expected_price=50000,
            executed_price=50010,
            exchange_interface=exchange_mock
        )
        
        self.assertEqual(result['symbol'], 'BTC-USDT')
        self.assertEqual(result['side'], 'buy')
        self.assertEqual(result['quantity'], 0.1)
        self.assertGreaterEqual(result['slippage_pct'], 0)  # 不利滑点应为正数
        self.assertIn('action_taken', result)
    
    def test_get_slippage_summary(self):
        """测试滑点汇总"""
        exchange_mock = MockExchange()
        
        # 添加几个历史交易
        self.slippage_monitor.monitor_trade_execution(
            symbol='BTC-USDT',
            side='buy',
            quantity=0.1,
            expected_price=50000,
            executed_price=50010,
            exchange_interface=exchange_mock
        )
        
        summary = self.slippage_monitor.get_slippage_summary(hours=24)
        
        self.assertIn('total_trades', summary)
        self.assertIn('avg_slippage_pct', summary)
        self.assertIn('max_slippage_pct', summary)


if __name__ == '__main__':
    unittest.main()