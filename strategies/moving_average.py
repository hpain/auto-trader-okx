"""
移动平均线策略实现
"""
import logging
from typing import Dict, Any
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy


class MovingAverageStrategy(BaseStrategy):
    """
    移动平均线策略
    基于短期和长期移动平均线的交叉来生成交易信号
    """
    
    def __init__(self, strategy_name: str = "MovingAverageStrategy", config: Dict[str, Any] = None):
        super().__init__(strategy_name, config or {})
        
        # 策略参数
        self.short_window = self.config.get('short_window', 5)
        self.long_window = self.config.get('long_window', 20)
        
        self.logger.info(f"MovingAverageStrategy initialized with parameters:")
        self.logger.info(f"  Short window: {self.short_window}")
        self.logger.info(f"  Long window: {self.long_window}")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            data: 包含价格数据的数据框
            
        Returns:
            包含交易信号的数据框
        """
        if not self.validate_data(data):
            # 返回带有默认信号的数据框
            signals = pd.DataFrame(index=data.index)
            signals['signal'] = 0  # 默认无操作信号
            signals['position'] = 0
            return signals
        
        # 预处理数据
        processed_data = self.prepare_data(data)
        
        # 计算移动平均线
        close_prices = processed_data['close']
        short_ma = close_prices.rolling(window=self.short_window).mean()
        long_ma = close_prices.rolling(window=self.long_window).mean()
        
        # 初始化信号数据框
        signals = pd.DataFrame(index=processed_data.index)
        signals['signal'] = 0  # 默认无操作信号
        signals['position'] = 0
        
        # 生成交易信号
        # 买入信号：短期均线上穿长期均线
        buy_signals = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
        
        # 卖出信号：短期均线下穿长期均线
        sell_signals = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
        
        # 设置信号
        signals.loc[buy_signals, 'signal'] = 1  # 买入信号
        signals.loc[sell_signals, 'signal'] = -1  # 卖出信号
        
        # 计算仓位
        signals['position'] = signals['signal'].replace(0, np.nan).ffill().fillna(0)
        
        # 添加辅助指标到结果中
        signals['short_ma'] = short_ma
        signals['long_ma'] = long_ma
        
        self.logger.debug(f"Generated {len(signals[signals['signal'] != 0])} trading signals out of {len(signals)} total periods")
        
        return signals
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        获取策略信息
        
        Returns:
            策略相关信息的字典
        """
        return {
            'name': self.strategy_name,
            'type': 'moving_average',
            'parameters': {
                'short_window': self.short_window,
                'long_window': self.long_window
            },
            'description': 'A moving average crossover strategy that generates buy signals when short MA crosses above long MA and sell signals when short MA crosses below long MA.'
        }
    
    def generate_signal(self, data: pd.DataFrame) -> int:
        """
        为单个时间点生成信号（兼容旧接口）
        """
        signals_df = self.generate_signals(data)
        if not signals_df.empty:
            # 返回最后一个信号
            return int(signals_df['signal'].iloc[-1])
        return 0


def generate_signal(data: pd.DataFrame) -> int:
    """
    Standalone function to generate a signal using default MovingAverageStrategy.
    This provides backward compatibility for modules expecting a simple function.
    """
    strategy = MovingAverageStrategy()
    return strategy.generate_signal(data)