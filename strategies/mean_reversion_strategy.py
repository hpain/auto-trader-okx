"""
均值回归策略实现
"""
import logging
from typing import Dict, Any
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """
    均值回归策略
    该策略基于价格偏离移动平均线的程度进行交易决策
    """
    
    def __init__(self, strategy_name: str = "MeanReversionStrategy", config: Dict[str, Any] = None):
        super().__init__(strategy_name, config or {})
        
        # 策略参数
        self.short_window = self.config.get('short_window', 10)  # 短期移动平均线窗口
        self.long_window = self.config.get('long_window', 30)    # 长期移动平均线窗口
        self.zscore_threshold = self.config.get('zscore_threshold', 2.0)  # Z-score阈值
        self.oversold_threshold = self.config.get('oversold_threshold', -2.0)  # 超卖阈值
        self.overbought_threshold = self.config.get('overbought_threshold', 2.0)  # 超买阈值
        self.rsi_period = self.config.get('rsi_period', 14)  # RSI周期
        self.rsi_oversold = self.config.get('rsi_oversold', 30)  # RSI超卖阈值
        self.rsi_overbought = self.config.get('rsi_overbought', 70)  # RSI超买阈值
        
        self.logger.info(f"MeanReversionStrategy initialized with parameters:")
        self.logger.info(f"  Short window: {self.short_window}")
        self.logger.info(f"  Long window: {self.long_window}")
        self.logger.info(f"  Z-score threshold: {self.zscore_threshold}")
        self.logger.info(f"  RSI period: {self.rsi_period}")
    
    def calculate_bollinger_bands(self, data: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """
        计算布林带
        
        Args:
            data: 价格数据
            window: 移动平均窗口
            num_std: 标准差倍数
            
        Returns:
            包含布林带的数据框
        """
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        bb_df = pd.DataFrame({
            'bb_middle': rolling_mean,
            'bb_upper': upper_band,
            'bb_lower': lower_band
        })
        
        # 计算价格相对于布林带的位置
        bb_df['bb_position'] = (data - bb_df['bb_lower']) / (bb_df['bb_upper'] - bb_df['bb_lower'])
        
        return bb_df
    
    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """
        计算RSI指标
        
        Args:
            data: 价格数据
            period: RSI周期
            
        Returns:
            RSI序列
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # 处理除零和NaN情况
        rs = gain / loss
        rs = rs.replace([np.inf, -np.inf], np.nan)  # 将无穷大替换为NaN
        rsi = 100 - (100 / (1 + rs))
        
        # RSI应在0-100之间，处理异常值
        rsi = np.clip(rsi, 0, 100)
        
        return rsi
    
    def calculate_zscore(self, data: pd.Series, window: int = 20) -> pd.Series:
        """
        计算Z-score
        
        Args:
            data: 价格数据
            window: 计算窗口
            
        Returns:
            Z-score序列
        """
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        
        zscore = (data - rolling_mean) / rolling_std
        
        return zscore
    
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
        
        # 计算技术指标
        close_prices = processed_data['close']
        
        # 计算移动平均线
        short_ma = close_prices.rolling(window=self.short_window).mean()
        long_ma = close_prices.rolling(window=self.long_window).mean()
        
        # 计算Z-score（基于长期移动平均）
        zscore = self.calculate_zscore(close_prices, window=self.long_window)
        
        # 计算RSI
        rsi = self.calculate_rsi(close_prices, period=self.rsi_period)
        
        # 计算布林带
        bb_data = self.calculate_bollinger_bands(close_prices)
        
        # 初始化信号数据框
        signals = pd.DataFrame(index=processed_data.index)
        signals['signal'] = 0  # 默认无操作信号
        signals['position'] = 0
        
        # 生成交易信号
        # 买入信号条件：
        # 1. Z-score低于超卖阈值（价格远低于均值）
        # 2. RSI低于超卖阈值
        # 3. 价格低于布林带下轨
        buy_condition = (
            (zscore < self.oversold_threshold) &
            (rsi < self.rsi_oversold) &
            (close_prices < bb_data['bb_lower'])
        )
        
        # 卖出信号条件：
        # 1. Z-score高于超买阈值（价格远高于均值）
        # 2. RSI高于超买阈值
        # 3. 价格高于布林带上轨
        sell_condition = (
            (zscore > self.overbought_threshold) &
            (rsi > self.rsi_overbought) &
            (close_prices > bb_data['bb_upper'])
        )
        
        # 超买超卖回补条件：
        # 1. 持有多头仓位且当前处于超买状态
        # 2. 持有空头仓位且当前处于超卖状态
        close_long_condition = (
            (signals['position'].shift(1, fill_value=0) == 1) &
            ((zscore > -0.5) | (rsi > 50) | (close_prices > bb_data['bb_middle']))
        )
        
        close_short_condition = (
            (signals['position'].shift(1, fill_value=0) == -1) &
            ((zscore < 0.5) | (rsi < 50) | (close_prices < bb_data['bb_middle']))
        )
        
        # 设置信号
        signals.loc[buy_condition, 'signal'] = 1  # 买入信号
        signals.loc[sell_condition, 'signal'] = -1  # 卖出信号
        signals.loc[close_long_condition, 'signal'] = 0  # 平多头
        signals.loc[close_short_condition, 'signal'] = 0  # 平空头
        
        # 计算仓位
        signals['position'] = signals['signal'].replace(0, np.nan).ffill().fillna(0)
        
        # 添加辅助指标到结果中（可选，用于调试和分析）
        signals['zscore'] = zscore
        signals['rsi'] = rsi
        signals['bb_position'] = bb_data['bb_position']
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
            'type': 'mean_reversion',
            'parameters': {
                'short_window': self.short_window,
                'long_window': self.long_window,
                'zscore_threshold': self.zscore_threshold,
                'oversold_threshold': self.oversold_threshold,
                'overbought_threshold': self.overbought_threshold,
                'rsi_period': self.rsi_period,
                'rsi_oversold': self.rsi_oversold,
                'rsi_overbought': self.rsi_overbought
            },
            'description': 'A mean reversion strategy that looks for opportunities when price deviates significantly from its historical average.'
        }