"""
市场状态检测器，用于识别当前市场环境并相应地调整策略

Enhanced with:
- Model prediction confidence tracking
- Should-trade signal based on model confidence
- Volatility regime detection
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional, List
from datetime import datetime, timedelta
from collections import deque


class MarketRegimeDetector:
    """
    市场制度检测器，用于识别市场处于趋势、震荡或高波动状态
    
    Enhanced with model confidence tracking to detect when the model
    is uncertain and trading should be reduced or stopped.
    """
    
    def __init__(self, confidence_window: int = 100):
        """
        Args:
            confidence_window: Number of recent predictions to track for confidence analysis
        """
        self.logger = logging.getLogger(__name__)
        self.confidence_window = confidence_window
        
        # Track recent model predictions
        self.prediction_history: deque = deque(maxlen=confidence_window)
        self.last_update = None
        
    def update_prediction(self, probability: float, prediction: int, timestamp: datetime = None):
        """
        Record a new model prediction for confidence tracking.
        
        Args:
            probability: Model's predicted probability (0-1)
            prediction: Binary prediction (0 or 1)
            timestamp: Time of prediction
        """
        self.prediction_history.append({
            'prob': probability,
            'pred': prediction,
            'ts': timestamp or datetime.now()
        })
        self.last_update = timestamp or datetime.now()
    
    def get_model_confidence_signal(self) -> Dict:
        """
        Analyze recent model predictions to determine if trading should continue.
        
        Returns:
            Dict with confidence metrics and trading recommendation
        """
        if len(self.prediction_history) < 10:
            return {
                'should_trade': True,
                'confidence_level': 'unknown',
                'reason': 'Insufficient prediction history',
                'avg_prob': None,
                'prob_std': None,
                'prediction_rate': None
            }
        
        recent = list(self.prediction_history)
        probs = [p['prob'] for p in recent]
        preds = [p['pred'] for p in recent]
        
        avg_prob = np.mean(probs)
        prob_std = np.std(probs)
        prediction_rate = np.mean(preds)  # How often model predicts 1
        
        # Decision logic
        # 1. If avg probability is very low, model is not confident
        # 2. If std is very low and avg is near 0.5, model is uncertain
        # 3. If prediction rate is very low, model sees no opportunities
        
        should_trade = True
        confidence_level = 'high'
        reasons = []
        
        # Check 1: Low average probability
        if avg_prob < 0.25:
            confidence_level = 'low'
            reasons.append(f'Low avg probability ({avg_prob:.2f})')
            should_trade = False
        
        # Check 2: Uncertainty (predictions clustered around 0.5)
        elif 0.4 < avg_prob < 0.6 and prob_std < 0.1:
            confidence_level = 'uncertain'
            reasons.append(f'Predictions clustered near 0.5 (avg={avg_prob:.2f}, std={prob_std:.2f})')
            should_trade = False
        
        # Check 3: No predictions being made
        elif prediction_rate < 0.05:
            confidence_level = 'low'
            reasons.append(f'Very few predictions (rate={prediction_rate:.1%})')
            should_trade = False
        
        # Check 4: Medium confidence
        elif avg_prob < 0.4 or prob_std < 0.05:
            confidence_level = 'medium'
            reasons.append(f'Moderate signals (avg={avg_prob:.2f})')
        
        return {
            'should_trade': should_trade,
            'confidence_level': confidence_level,
            'reason': '; '.join(reasons) if reasons else 'Model showing confident signals',
            'avg_prob': avg_prob,
            'prob_std': prob_std,
            'prediction_rate': prediction_rate,
            'sample_size': len(recent)
        }
    
    def should_trade(self, data: pd.DataFrame = None) -> Tuple[bool, str]:
        """
        Combined decision: should we trade right now?
        
        Combines:
        1. Model confidence signal
        2. Market regime (if data provided)
        
        Returns:
            (should_trade: bool, reason: str)
        """
        # Get model confidence
        model_signal = self.get_model_confidence_signal()
        
        if not model_signal['should_trade']:
            return False, f"Model confidence too low: {model_signal['reason']}"
        
        # If we have market data, also check regime
        if data is not None and len(data) >= 20:
            regime_info = self.detect_regime(data)
            
            # Don't trade in sideways or high volatility uncertain markets
            if regime_info['regime'] == 'sideways':
                return False, f"Market is sideways: {regime_info['description']}"
            
            if regime_info['regime'] == 'high_volatility' and regime_info['trend_strength'] < 0.2:
                return False, f"High volatility without clear trend: {regime_info['description']}"
        
        return True, f"OK to trade. Model confidence: {model_signal['confidence_level']}"
    
    def get_position_multiplier(self) -> float:
        """
        Get a position size multiplier based on current confidence.
        
        Returns:
            0.0 - 1.0 multiplier for position sizing
        """
        signal = self.get_model_confidence_signal()
        
        if not signal['should_trade']:
            return 0.0
        
        if signal['confidence_level'] == 'high':
            return 1.0
        elif signal['confidence_level'] == 'medium':
            return 0.5
        else:
            return 0.25
        
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
        
        # First check model confidence
        model_signal = self.get_model_confidence_signal()
        if not model_signal['should_trade']:
            return 'hold', f"Model confidence low: {model_signal['reason']}"
        
        if regime == 'high_volatility':
            return 'lgb', 'High volatility - Use LGB for robust predictions'
        elif regime == 'sideways':
            return 'hold', 'Sideways market - Reduce trading activity'
        elif regime == 'uptrend':
            return 'lgb', 'Uptrend - Use LGB for trend following'
        elif regime == 'downtrend':
            return 'lgb', 'Downtrend - Use LGB for trend following'
        else:  # moderate
            return 'lgb', 'Moderate conditions - Use LGB'