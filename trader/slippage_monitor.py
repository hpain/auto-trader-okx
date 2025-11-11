"""
滑点监控模块，用于检测和控制交易中的滑点
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import pandas as pd


class SlippageMonitor:
    """
    滑点监控器，用于检测交易执行中的滑点情况
    """
    
    def __init__(self, config: Dict):
        self.config = config.get('slippage_monitoring', {})
        self.max_slippage_pct = self.config.get('max_slippage_pct', 0.02)  # 2%的最大滑点
        self.volume_threshold = self.config.get('volume_threshold', 1000000)  # 最小交易量阈值
        self.bid_ask_spread_threshold = self.config.get('bid_ask_spread_threshold', 0.005)  # 0.5%的买卖差价阈值
        
        self.logger = logging.getLogger(__name__)
        self.trade_history = []  # 存储交易历史用于滑点分析
        
    def calculate_slippage(self, expected_price: float, executed_price: float, side: str) -> float:
        """
        计算滑点百分比
        
        Args:
            expected_price: 预期成交价格
            executed_price: 实际成交价格
            side: 交易方向 ('buy' 或 'sell')
            
        Returns:
            滑点百分比 (正数表示不利滑点)
        """
        if expected_price <= 0:
            return 0.0
            
        if side.lower() == 'buy':
            # 买入时，实际价格高于预期价格为不利滑点
            slippage = (executed_price - expected_price) / expected_price
        elif side.lower() == 'sell':
            # 卖出时，实际价格低于预期价格为不利滑点
            slippage = (expected_price - executed_price) / expected_price
        else:
            self.logger.warning(f"Invalid trade side: {side}")
            return 0.0
            
        return max(0, slippage)  # 只返回不利滑点（正数）
    
    def is_excessive_slippage(self, expected_price: float, executed_price: float, side: str) -> bool:
        """
        检查是否发生了过度滑点
        
        Args:
            expected_price: 预期成交价格
            executed_price: 实际成交价格
            side: 交易方向
            
        Returns:
            True 如果滑点超过阈值
        """
        slippage_pct = self.calculate_slippage(expected_price, executed_price, side)
        return slippage_pct > self.max_slippage_pct
    
    def check_liquidity_conditions(self, symbol: str, exchange_interface) -> Tuple[bool, str]:
        """
        检查流动性条件
        
        Args:
            symbol: 交易对符号
            exchange_interface: 交易所接口实例
            
        Returns:
            (是否满足流动性条件, 原因描述)
        """
        try:
            # 获取24小时交易量
            ticker = exchange_interface.get_ticker(symbol)
            if not ticker:
                return False, f"Could not fetch ticker data for {symbol}"
                
            volume_24h = ticker.get('quote_volume_24h', 0)  # 24小时报价货币交易量
            if volume_24h < self.volume_threshold:
                return False, f"Low liquidity: 24h volume {volume_24h} < threshold {self.volume_threshold}"
            
            # 获取市场深度并计算买卖差价
            depth = exchange_interface.get_orderbook(symbol, depth=1)
            if not depth or 'bids' not in depth or 'asks' not in depth:
                return False, f"Could not fetch order book for {symbol}"
                
            best_bid = depth['bids'][0][0] if depth['bids'] else 0
            best_ask = depth['asks'][0][0] if depth['asks'] else 0
            
            if best_bid == 0 or best_ask == 0:
                return False, f"Invalid bid/ask prices for {symbol}"
                
            mid_price = (best_bid + best_ask) / 2
            spread_pct = (best_ask - best_bid) / mid_price
            
            if spread_pct > self.bid_ask_spread_threshold:
                return False, f"High bid-ask spread: {spread_pct:.4f} > threshold {self.bid_ask_spread_threshold}"
                
            return True, f"Liquidity OK: volume {volume_24h}, spread {spread_pct:.4f}"
            
        except Exception as e:
            self.logger.error(f"Error checking liquidity for {symbol}: {e}")
            return False, f"Error checking liquidity: {e}"
    
    def monitor_trade_execution(self, symbol: str, side: str, quantity: float, 
                              expected_price: float, executed_price: float, 
                              exchange_interface) -> Dict:
        """
        监控单笔交易执行情况
        
        Args:
            symbol: 交易对
            side: 交易方向
            quantity: 交易数量
            expected_price: 预期价格
            executed_price: 实际成交价格
            exchange_interface: 交易所接口
            
        Returns:
            监控结果字典
        """
        result = {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'expected_price': expected_price,
            'executed_price': executed_price,
            'timestamp': datetime.utcnow().isoformat(),
            'slippage_pct': 0.0,
            'is_excessive_slippage': False,
            'liquidity_ok': True,
            'warning': '',
            'action_taken': 'proceed'  # 'proceed', 'warn', 'cancel'
        }
        
        # 计算滑点
        slippage_pct = self.calculate_slippage(expected_price, executed_price, side)
        result['slippage_pct'] = slippage_pct
        result['is_excessive_slippage'] = self.is_excessive_slippage(expected_price, executed_price, side)
        
        # 检查流动性
        liquidity_ok, liquidity_msg = self.check_liquidity_conditions(symbol, exchange_interface)
        result['liquidity_ok'] = liquidity_ok
        
        # 记录交易历史
        self.trade_history.append(result.copy())
        
        # 根据滑点和流动性情况决定是否采取行动
        if result['is_excessive_slippage']:
            result['warning'] = f"Excessive slippage detected: {slippage_pct:.4f} > threshold {self.max_slippage_pct}"
            result['action_taken'] = 'warn'
            self.logger.warning(result['warning'])
        elif not liquidity_ok:
            result['warning'] = f"Liquidity concerns: {liquidity_msg}"
            result['action_taken'] = 'warn'  # 可以改为 'cancel' 以完全取消交易
            self.logger.warning(result['warning'])
        else:
            self.logger.debug(f"Trade executed within acceptable parameters: slippage={slippage_pct:.4f}")
        
        return result
    
    def get_slippage_summary(self, hours: int = 24) -> Dict:
        """
        获取指定时间范围内的滑点汇总
        
        Args:
            hours: 时间范围（小时）
            
        Returns:
            滑点汇总信息
        """
        if not self.trade_history:
            return {
                'total_trades': 0,
                'avg_slippage_pct': 0.0,
                'max_slippage_pct': 0.0,
                'excessive_slippage_count': 0,
                'period_hours': hours
            }
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_trades = [
            trade for trade in self.trade_history
            if datetime.fromisoformat(trade['timestamp']) > cutoff_time
        ]
        
        if not recent_trades:
            return {
                'total_trades': 0,
                'avg_slippage_pct': 0.0,
                'max_slippage_pct': 0.0,
                'excessive_slippage_count': 0,
                'period_hours': hours
            }
        
        slippage_values = [trade['slippage_pct'] for trade in recent_trades]
        excessive_count = sum(1 for trade in recent_trades if trade['is_excessive_slippage'])
        
        return {
            'total_trades': len(recent_trades),
            'avg_slippage_pct': sum(slippage_values) / len(slippage_values) if slippage_values else 0.0,
            'max_slippage_pct': max(slippage_values) if slippage_values else 0.0,
            'excessive_slippage_count': excessive_count,
            'period_hours': hours
        }
    
    def should_restrict_trading(self, symbol: str, exchange_interface) -> Tuple[bool, str]:
        """
        检查是否应该限制对某个交易对的交易
        
        Args:
            symbol: 交易对
            exchange_interface: 交易所接口
            
        Returns:
            (是否应该限制交易, 原因)
        """
        # 检查流动性
        liquidity_ok, liquidity_msg = self.check_liquidity_conditions(symbol, exchange_interface)
        if not liquidity_ok:
            return True, f"Liquidity issue: {liquidity_msg}"
        
        # 检查近期滑点历史
        slippage_summary = self.get_slippage_summary(hours=1)  # 检查过去1小时
        if slippage_summary['total_trades'] > 5:  # 如果有足够的交易样本
            if slippage_summary['avg_slippage_pct'] > self.max_slippage_pct * 1.5:  # 如果平均滑点超过阈值的1.5倍
                return True, f"High average slippage in past hour: {slippage_summary['avg_slippage_pct']:.4f}"
            if slippage_summary['excessive_slippage_count'] > slippage_summary['total_trades'] * 0.3:  # 如果30%以上的交易有过度滑点
                return True, f"Too many excessive slippage trades: {slippage_summary['excessive_slippage_count']}/{slippage_summary['total_trades']}"
        
        return False, "Trading allowed"


# 全局实例
_slippage_monitor = None


def get_slippage_monitor(config: Dict) -> SlippageMonitor:
    """
    获取或创建滑点监控器实例
    """
    global _slippage_monitor
    if _slippage_monitor is None:
        _slippage_monitor = SlippageMonitor(config)
    return _slippage_monitor