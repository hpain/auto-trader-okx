"""
套利策略实现
"""
import logging
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy


class ArbitrageStrategy(BaseStrategy):
    """
    套利策略
    该策略基于不同市场或交易对之间的价格差异进行套利
    """
    
    def __init__(self, strategy_name: str = "ArbitrageStrategy", config: Dict[str, Any] = None):
        super().__init__(strategy_name, config or {})
        
        # 策略参数
        self.spread_threshold = self.config.get('spread_threshold', 0.005)  # 0.5%的价差阈值
        self.min_volume = self.config.get('min_volume', 1000)  # 最小交易量
        self.max_position_size = self.config.get('max_position_size', 10000)  # 最大头寸规模
        self.spread_lookback = self.config.get('spread_lookback', 20)  # 价差回看窗口
        self.zscore_threshold = self.config.get('zscore_threshold', 2.0)  # Z-score阈值
        self.exit_threshold = self.config.get('exit_threshold', 0.001)  # 0.1%的退出阈值
        self.enable_time_arbitrage = self.config.get('enable_time_arbitrage', False)  # 是否启用时间套利
        self.enable_cross_exchange = self.config.get('enable_cross_exchange', False)  # 是否启用跨交易所套利
        
        self.logger.info(f"ArbitrageStrategy initialized with parameters:")
        self.logger.info(f"  Spread threshold: {self.spread_threshold:.4f}")
        self.logger.info(f"  Min volume: {self.min_volume}")
        self.logger.info(f"  Max position size: {self.max_position_size}")
        self.logger.info(f"  Spread lookback: {self.spread_lookback}")
        self.logger.info(f"  Z-score threshold: {self.zscore_threshold}")
        
        # 用于跟踪当前头寸
        self.current_position = 0
        self.entry_spread = 0
        
    def calculate_zscore(self, data: pd.Series, window: int = 20) -> pd.Series:
        """
        计算Z-score
        
        Args:
            data: 数据序列
            window: 计算窗口
            
        Returns:
            Z-score序列
        """
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        
        zscore = (data - rolling_mean) / rolling_std
        
        return zscore
    
    def calculate_spread(self, price_a: pd.Series, price_b: pd.Series) -> pd.Series:
        """
        计算两个价格序列之间的价差
        
        Args:
            price_a: 价格序列A
            price_b: 价格序列B
            
        Returns:
            价差序列（以百分比形式）
        """
        spread = (price_a - price_b) / ((price_a + price_b) / 2)
        return spread
    
    def detect_cointegration(self, price_a: pd.Series, price_b: pd.Series, 
                           threshold: float = 0.1) -> Tuple[bool, float]:
        """
        检测两个价格序列之间的协整关系（简化版本）
        
        Args:
            price_a: 价格序列A
            price_b: 价格序列B
            threshold: 协整检验阈值
            
        Returns:
            (是否协整, 检验统计量)
        """
        try:
            # 简化的协整检验：计算价差的ADF统计量的代理
            # 实际应用中应该使用更正式的协整检验方法
            spread = self.calculate_spread(price_a, price_b)
            
            # 计算价差的均值回归特性
            mean_reversion_speed = self.estimate_mean_reversion_speed(spread.dropna())
            
            # 如果均值回归速度快于阈值，认为存在协整关系
            is_cointegrated = mean_reversion_speed > threshold
            return is_cointegrated, mean_reversion_speed
        except Exception as e:
            self.logger.error(f"Error in cointegration detection: {e}")
            return False, 0.0
    
    def estimate_mean_reversion_speed(self, spread: pd.Series) -> float:
        """
        估计均值回归速度（使用半衰期作为代理）
        
        Args:
            spread: 价差序列
            
        Returns:
            均值回归速度
        """
        if len(spread) < 2:
            return 0.0
            
        # 计算价差的回归系数
        spread_lagged = spread.shift(1).dropna()
        spread_current = spread[1:]
        
        if len(spread_lagged) < 2:
            return 0.0
            
        # 简单的回归分析
        slope = np.polyfit(spread_lagged, spread_current, 1)[0]
        
        # 计算半衰期作为均值回归速度的代理
        if slope < 1:
            half_life = -np.log(2) / np.log(abs(slope)) if abs(slope) < 1 else float('inf')
            return 1.0 / max(half_life, 1.0)  # 转换为回归速度
        else:
            return 0.0
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号（这是一个简化的版本，实际套利策略会更复杂）
        
        Args:
            data: 包含价格数据的数据框，应该包含至少两个交易对的价格
        
        Returns:
            包含交易信号的数据框
        """
        if not self.validate_data(data):
            signals = pd.DataFrame(index=data.index)
            signals['signal'] = 0  # 默认无操作信号
            signals['position'] = 0
            return signals
        
        # 预处理数据
        processed_data = self.prepare_data(data)
        
        # 初始化信号数据框
        signals = pd.DataFrame(index=processed_data.index)
        signals['signal'] = 0  # 默认无操作信号
        signals['position'] = 0
        signals['spread'] = 0
        signals['zscore'] = 0
        
        # 检查是否包含多个交易对的数据（简化处理）
        # 实际应用中，套利策略通常需要多个数据源
        if 'close' in processed_data.columns:
            # 如果只有一个价格序列，我们模拟套利机会（例如，使用不同时间段的移动平均）
            close_prices = processed_data['close']
            
            # 使用短期和长期移动平均的差值作为"价差"
            short_ma = close_prices.rolling(window=5).mean()
            long_ma = close_prices.rolling(window=20).mean()
            
            # 计算价差（以百分比形式）
            spread = self.calculate_spread(short_ma, long_ma)
            signals['spread'] = spread
            
            # 如果存在原始的两个价格序列，使用它们
            if 'close_a' in processed_data.columns and 'close_b' in processed_data.columns:
                close_a = processed_data['close_a']
                close_b = processed_data['close_b']
                spread = self.calculate_spread(close_a, close_b)
                signals['spread'] = spread
            elif 'close_alt' in processed_data.columns:
                close_alt = processed_data['close_alt']
                spread = self.calculate_spread(close_prices, close_alt)
                signals['spread'] = spread
        
        else:
            # 如果没有close列，尝试从特征中找出价格数据
            price_cols = [col for col in processed_data.columns if 'price' in col.lower() or 'close' in col.lower()]
            if len(price_cols) >= 2:
                close_a = processed_data[price_cols[0]]
                close_b = processed_data[price_cols[1]]
                spread = self.calculate_spread(close_a, close_b)
                signals['spread'] = spread
            else:
                # 如果没有足够的价格数据，返回默认信号
                self.logger.warning("Not enough price data for arbitrage strategy, returning default signals")
                return signals
        
        # 计算价差的Z-score
        zscore = self.calculate_zscore(signals['spread'], window=self.spread_lookback)
        signals['zscore'] = zscore
        
        # 计算交易量相关的列（如果存在）
        volume_available = True
        if 'volume' in processed_data.columns:
            volume = processed_data['volume']
        else:
            volume = pd.Series([self.min_volume] * len(signals), index=signals.index)
        
        # 生成交易信号
        # 做多价差策略：当价差远低于均值时买入价差（买入A卖出B）
        long_spread_condition = (
            (zscore < -self.zscore_threshold) & 
            (volume >= self.min_volume)
        )
        
        # 做空价差策略：当价差远高于均值时卖空价差（卖出A买入B）
        short_spread_condition = (
            (zscore > self.zscore_threshold) & 
            (volume >= self.min_volume)
        )
        
        # 退出条件：价差回归到均值附近或达到目标利润
        exit_long_condition = (
            (self.current_position > 0) &  # 当前持有多头价差
            ((zscore >= -0.5) | (zscore > self.entry_spread - self.exit_threshold))  # 价差回归或达到盈利目标
        )
        
        exit_short_condition = (
            (self.current_position < 0) &  # 当前持有空头价差
            ((zscore <= 0.5) | (zscore < self.entry_spread + self.exit_threshold))  # 价差回归或达到盈利目标
        )
        
        # 设置信号
        signals.loc[long_spread_condition, 'signal'] = 1  # 做多价差
        signals.loc[short_spread_condition, 'signal'] = -1  # 做空价差
        signals.loc[exit_long_condition, 'signal'] = 0  # 平多头价差
        signals.loc[exit_short_condition, 'signal'] = 0  # 平空头价差
        
        # 计算仓位
        signals['position'] = signals['signal'].replace(0, np.nan).ffill().fillna(0)
        
        # 更新当前头寸状态
        current_signal = signals['signal'].iloc[-1]
        current_zscore = signals['zscore'].iloc[-1]
        
        if current_signal != 0 and self.current_position == 0:
            # 开仓
            self.current_position = current_signal
            self.entry_spread = current_zscore
        elif current_signal == 0 and self.current_position != 0:
            # 平仓
            self.current_position = 0
            self.entry_spread = 0
        elif current_signal != 0 and self.current_position != current_signal:
            # 反向开仓
            self.current_position = current_signal
            self.entry_spread = current_zscore
        
        self.logger.debug(f"Arbitrage strategy position: {self.current_position}, entry spread: {self.entry_spread:.4f}")
        
        return signals
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        获取策略信息
        
        Returns:
            策略相关信息的字典
        """
        return {
            'name': self.strategy_name,
            'type': 'arbitrage',
            'parameters': {
                'spread_threshold': self.spread_threshold,
                'min_volume': self.min_volume,
                'max_position_size': self.max_position_size,
                'spread_lookback': self.spread_lookback,
                'zscore_threshold': self.zscore_threshold,
                'exit_threshold': self.exit_threshold,
                'enable_time_arbitrage': self.enable_time_arbitrage,
                'enable_cross_exchange': self.enable_cross_exchange
            },
            'description': 'An arbitrage strategy that looks for price discrepancies between different markets or instruments.'
        }
    
    def update_position(self, new_position: int, entry_price: float):
        """
        更新头寸信息
        
        Args:
            new_position: 新的头寸大小
            entry_price: 建仓价格
        """
        self.current_position = new_position
        if new_position != 0:
            self.entry_spread = entry_price
        else:
            self.entry_spread = 0