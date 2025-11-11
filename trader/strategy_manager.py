"""
策略管理器，用于管理多个交易策略并根据市场状态切换
"""
import logging
from typing import Dict, List, Any
import pandas as pd
from .market_regime_detector import MarketRegimeDetector


class StrategyManager:
    """
    策略管理器，负责多个策略的管理、评估和切换
    """
    
    def __init__(self, strategies: List[Any]):
        """
        初始化策略管理器
        
        Args:
            strategies: 策略实例列表
        """
        self.strategies = {strategy.strategy_name: strategy for strategy in strategies} if strategies else {}
        self.active_strategies = list(self.strategies.keys())
        self.performance_tracker = {}
        self.logger = logging.getLogger(__name__)
        
        # 初始化市场状态检测器
        self.regime_detector = MarketRegimeDetector()
    
    def get_strategy(self, strategy_name: str):
        """
        获取指定名称的策略
        
        Args:
            strategy_name: 策略名称
            
        Returns:
            策略实例
        """
        return self.strategies.get(strategy_name)
    
    def get_all_strategies(self) -> List[Any]:
        """
        获取所有策略列表
        
        Returns:
            策略实例列表
        """
        return list(self.strategies.values())
    
    def switch_strategy(self, recommended_strategy: str, cycle_logger=None):
        """
        切换到推荐的策略
        
        Args:
            recommended_strategy: 推荐的策略名称
            cycle_logger: 周期日志记录器
        """
        if recommended_strategy in self.strategies:
            old_active = self.active_strategies.copy()
            self.active_strategies = [recommended_strategy]
            
            if cycle_logger:
                cycle_logger.add_info(f"Switched strategy from {old_active} to {recommended_strategy}")
            
            self.logger.info(f"Strategy switched from {old_active} to {recommended_strategy}")
        else:
            if cycle_logger:
                cycle_logger.add_warning(f"Recommended strategy '{recommended_strategy}' not found, keeping current strategies")
            
            self.logger.warning(f"Recommended strategy '{recommended_strategy}' not found")
    
    def evaluate_strategy_performance(self, symbol: str, data: pd.DataFrame) -> Dict[str, float]:
        """
        评估各策略的性能
        
        Args:
            symbol: 交易对符号
            data: 市场数据
            
        Returns:
            各策略性能字典
        """
        performance = {}
        
        for name, strategy in self.strategies.items():
            try:
                # 这里可以根据历史数据评估策略表现
                # 暂时使用简单的方法，实际应用中可以更复杂
                signal = strategy.generate_signal(data)
                # 这里可以计算策略的回测性能指标
                performance[name] = float(signal)  # 临时实现
            except Exception as e:
                self.logger.error(f"Error evaluating strategy {name}: {e}")
                performance[name] = 0.0
        
        return performance
    
    def get_active_strategies(self) -> List[Any]:
        """
        获取当前激活的策略列表
        
        Returns:
            激活的策略实例列表
        """
        return [self.strategies[name] for name in self.active_strategies if name in self.strategies]
    
    def analyze_market_regime(self, data: pd.DataFrame, cycle_logger=None) -> Dict:
        """
        分析当前市场状态
        
        Args:
            data: 市场数据
            cycle_logger: 周期日志记录器
            
        Returns:
            市场状态信息
        """
        regime_info = self.regime_detector.detect_regime(data)
        
        if cycle_logger:
            cycle_logger.add_info(f"Market regime detected: {regime_info['regime']} - {regime_info['description']}")
        
        self.logger.info(f"Market regime: {regime_info['regime']}, {regime_info['description']}")
        
        return regime_info
    
    def get_recommended_strategy(self, regime_info: Dict, cycle_logger=None) -> str:
        """
        根据市场状态推荐策略
        
        Args:
            regime_info: 市场状态信息
            cycle_logger: 周期日志记录器
            
        Returns:
            推荐的策略名称
        """
        recommended_strategy, reason = self.regime_detector.recommend_strategy(regime_info)
        
        if cycle_logger:
            cycle_logger.add_info(f"Recommended strategy: {recommended_strategy} - {reason}")
        
        self.logger.info(f"Recommended strategy: {recommended_strategy} - {reason}")
        
        # 检查推荐的策略是否存在于我们的策略池中
        if recommended_strategy not in self.strategies:
            # 如果推荐的策略不存在，返回第一个可用的策略
            available_strategies = list(self.strategies.keys())
            if available_strategies:
                recommended_strategy = available_strategies[0]
                if cycle_logger:
                    cycle_logger.add_info(f"Recommended strategy not available, using: {recommended_strategy}")
        
        return recommended_strategy