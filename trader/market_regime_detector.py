"""
市场状态检测器，用于识别当前市场环境并相应地调整策略
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta


class MarketRegimeDetector:
    """
    市场制度检测器，用于识别市场处于趋势、震荡或高波动状态
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def detect_regime(self, data: pd.DataFrame, lookback_period: int = 20) -> Dict:
        """
        检测当前市场制度
        :param data: 价格数据
        :param lookback_period: 回看周期
        :return: 市场制度信息字典
        """
        if len(data) < lookback_period:
            return {
                'regime': 'insufficient_data',
                'confidence': 0.0,
                'strength': 0.0,
                'description': 'Insufficient data for regime detection'
            }
        
        # 获取最近的数据
        recent_data = data.tail(lookback_period).copy()
        
        # 计算价格变化率
        returns = recent_data['close'].pct_change().dropna()
        
        # 计算波动率
        volatility = returns.std() * np.sqrt(365 * 24)  # 年化波动率 (假设数据是小时级别的)
        
        # 计算趋势强度 - 使用价格与移动平均线的偏离度
        sma_20 = recent_data['close'].rolling(window=20).mean()
        price_ma_ratio = (recent_data['close'] - sma_20) / sma_20
        
        # 计算趋势一致性 - 价格变化方向的一致性
        trend_consistency = abs(returns.sum()) / returns.abs().sum() if returns.abs().sum() != 0 else 0
        
        # 计算价格范围的相对位置 (是否在相对高位/低位)
        recent_high = recent_data['high'].max()
        recent_low = recent_data['low'].min()
        current_price = recent_data['close'].iloc[-1]
        
        if recent_high != recent_low:
            price_position = (current_price - recent_low) / (recent_high - recent_low)
        else:
            price_position = 0.5  # 价格无变化时，认为在中间位置
        
        # 识别市场制度
        regime_info = self._classify_regime(
            volatility=volatility,
            trend_strength=trend_consistency,
            price_position=price_position,
            recent_data=recent_data
        )
        
        return regime_info
    
    def _classify_regime(self, volatility: float, trend_strength: float, 
                        price_position: float, recent_data: pd.DataFrame) -> Dict:
        """
        根据指标分类市场制度
        """
        # 定义阈值
        high_vol_threshold = 0.5  # 高波动率阈值
        low_vol_threshold = 0.15  # 低波动率阈值
        strong_trend_threshold = 0.4  # 强趋势阈值
        weak_trend_threshold = 0.1  # 弱趋势阈值
        
        # 计算价格方向指标
        start_price = recent_data['close'].iloc[0]
        end_price = recent_data['close'].iloc[-1]
        price_direction = (end_price - start_price) / start_price
        
        # 根据波动率和趋势强度判断制度
        if volatility > high_vol_threshold:
            regime = 'high_volatility'
            description = f'High volatility market (vol={volatility:.3f}), uncertain direction'
        elif volatility < low_vol_threshold and trend_strength < weak_trend_threshold:
            regime = 'sideways'
            description = f'Sideways/consolidating market (vol={volatility:.3f}, trend={trend_strength:.3f})'
        elif trend_strength > strong_trend_threshold:
            if price_direction > 0:
                regime = 'uptrend'
                description = f'Strong uptrend (vol={volatility:.3f}, trend={trend_strength:.3f}, dir={price_direction:.3f})'
            else:
                regime = 'downtrend'
                description = f'Strong downtrend (vol={volatility:.3f}, trend={trend_strength:.3f}, dir={price_direction:.3f})'
        else:
            regime = 'moderate'
            description = f'Moderate market conditions (vol={volatility:.3f}, trend={trend_strength:.3f})'
        
        # 计算置信度（简单基于数据的稳定性）
        confidence = min(1.0, volatility * 2)  # 波动率越高，判断的置信度可能越低
        if regime in ['sideways']:
            confidence = min(confidence, 0.8)  # 震荡市场的判断可能较不稳定
        
        return {
            'regime': regime,
            'confidence': confidence,
            'strength': abs(price_direction) if regime in ['uptrend', 'downtrend'] else trend_strength,
            'description': description,
            'volatility': volatility,
            'trend_strength': trend_strength,
            'price_direction': price_direction
        }
    
    def recommend_strategy(self, regime_info: Dict) -> Tuple[str, str]:
        """
        根据市场制度推荐策略
        :param regime_info: 市场制度信息
        :return: (推荐策略, 推荐理由)
        """
        regime = regime_info['regime']
        
        if regime == 'high_volatility':
            return 'ma_fast', 'High volatility - Use fast MA for quick entries/exits'
        elif regime == 'sideways':
            return 'ma_fast', 'Sideways market - Use mean reversion strategy with fast MA'
        elif regime == 'uptrend':
            return 'lgb', 'Uptrend - Use ML strategy for trend following'
        elif regime == 'downtrend':
            return 'lgb', 'Downtrend - Use ML strategy for trend following (short positions if supported)'
        else:  # moderate
            return 'lgb', 'Moderate conditions - Use ML strategy'