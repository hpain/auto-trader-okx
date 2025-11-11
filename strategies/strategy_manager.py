"""
策略管理器，用于管理和切换不同的交易策略
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

from strategies.base_strategy import BaseStrategy
from strategies.lgb_strategy import LGBStrategy
from strategies.moving_average import MovingAverageStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.arbitrage_strategy import ArbitrageStrategy


class StrategyManager:
    """
    策略管理器，用于管理和根据市场条件切换不同的交易策略
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化策略管理器
        :param config: 配置字典
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.strategies: Dict[str, BaseStrategy] = {}
        self.active_strategy: Optional[str] = None
        self.strategy_performance: Dict[str, List[Dict]] = {}
        
        self._initialize_strategies()
        
    def _initialize_strategies(self):
        """
        根据配置初始化所有可用的策略
        """
        # 从配置中获取要启用的策略
        enabled_strategies = self.config.get('enabled_strategies', ['lgb'])
        
        for strategy_name in enabled_strategies:
            try:
                if strategy_name == 'lgb':
                    strategy_config = self.config.get('lgb_strategy', {
                        'model_dir': 'models',
                        'model_name': 'best_model.pkl',
                        'metadata_name': 'metadata.json'
                    })
                    self.strategies['lgb'] = LGBStrategy(
                        strategy_name="LGB_Main", 
                        config=strategy_config
                    )
                    
                elif strategy_name == 'ma_fast':
                    strategy_config = self.config.get('ma_fast_strategy', {
                        'short_window': 5,
                        'long_window': 15
                    })
                    self.strategies['ma_fast'] = MovingAverageStrategy(
                        strategy_name="MA_Fast", 
                        config=strategy_config
                    )
                    
                elif strategy_name == 'ma_slow':
                    strategy_config = self.config.get('ma_slow_strategy', {
                        'short_window': 20,
                        'long_window': 50
                    })
                    self.strategies['ma_slow'] = MovingAverageStrategy(
                        strategy_name="MA_Slow", 
                        config=strategy_config
                    )
                    
                elif strategy_name == 'mean_reversion':
                    strategy_config = self.config.get('mean_reversion_strategy', {
                        'short_window': 10,
                        'long_window': 30,
                        'zscore_threshold': 2.0,
                        'rsi_period': 14
                    })
                    self.strategies['mean_reversion'] = MeanReversionStrategy(
                        strategy_name="MeanReversion", 
                        config=strategy_config
                    )
                    
                elif strategy_name == 'arbitrage':
                    strategy_config = self.config.get('arbitrage_strategy', {
                        'spread_threshold': 0.005,
                        'zscore_threshold': 2.0,
                        'min_volume': 1000
                    })
                    self.strategies['arbitrage'] = ArbitrageStrategy(
                        strategy_name="Arbitrage", 
                        config=strategy_config
                    )
                    
                else:
                    self.logger.warning(f"Unknown strategy type: {strategy_name}")
                    continue
                    
                self.strategy_performance[strategy_name] = []
                self.logger.info(f"Initialized strategy: {strategy_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize strategy {strategy_name}: {e}")
        
        # 设置默认策略
        if self.strategies:
            self.active_strategy = list(self.strategies.keys())[0]
            self.logger.info(f"Set default active strategy: {self.active_strategy}")
    
    def generate_signals(self, data: pd.DataFrame, strategy_name: Optional[str] = None) -> pd.DataFrame:
        """
        为指定策略生成信号
        :param data: 市场数据
        :param strategy_name: 策略名称，如果为None则使用当前激活的策略
        :return: 信号DataFrame
        """
        if strategy_name is None:
            strategy_name = self.active_strategy
            
        if strategy_name not in self.strategies:
            self.logger.error(f"Strategy {strategy_name} not found, using active strategy instead")
            strategy_name = self.active_strategy
            
        if strategy_name not in self.strategies:
            raise ValueError("No strategies available")
            
        try:
            strategy = self.strategies[strategy_name]
            signals = strategy.generate_signals(data)
            self.logger.debug(f"Generated signals using strategy: {strategy_name}")
            return signals
        except Exception as e:
            self.logger.error(f"Error generating signals with strategy {strategy_name}: {e}")
            # 返回空信号
            return pd.DataFrame(index=data.index, data={'signal': 0})
    
    def get_active_strategy_name(self) -> Optional[str]:
        """
        获取当前激活的策略名称
        """
        return self.active_strategy
    
    def switch_strategy(self, strategy_name: str) -> bool:
        """
        切换到指定策略
        :param strategy_name: 要切换到的策略名称
        :return: 是否切换成功
        """
        if strategy_name in self.strategies:
            old_strategy = self.active_strategy
            self.active_strategy = strategy_name
            self.logger.info(f"Switched strategy from {old_strategy} to {strategy_name}")
            return True
        else:
            self.logger.error(f"Cannot switch to strategy {strategy_name}, it does not exist")
            return False
    
    def evaluate_strategy_performance(self, strategy_name: str, returns: pd.Series) -> Dict[str, float]:
        """
        评估特定策略的性能
        :param strategy_name: 策略名称
        :param returns: 收益率序列
        :return: 性能指标字典
        """
        if strategy_name not in self.strategies:
            return {}
        
        # 计算基本性能指标
        total_return = (1 + returns).prod() - 1
        avg_return = returns.mean() if len(returns) > 0 else 0
        volatility = returns.std() if len(returns) > 1 else 0
        sharpe_ratio = (avg_return / volatility) * (252 * 24) ** 0.5 if volatility != 0 else 0
        
        # 计算胜率
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        
        # 计算最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() if not drawdown.empty else 0
        
        performance = {
            'total_return': total_return,
            'avg_return': avg_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'trade_count': len(returns)
        }
        
        # 保存性能记录
        self.strategy_performance[strategy_name].append({
            'timestamp': datetime.utcnow().isoformat(),
            'performance': performance
        })
        
        # 只保留最近的性能记录
        if len(self.strategy_performance[strategy_name]) > 20:
            self.strategy_performance[strategy_name] = self.strategy_performance[strategy_name][-20:]
        
        return performance
    
    def get_best_performing_strategy(self, lookback_days: int = 7) -> Optional[str]:
        """
        获取最近表现最好的策略
        :param lookback_days: 回看天数
        :return: 表现最好的策略名称
        """
        best_strategy = None
        best_sharpe = float('-inf')
        
        for strategy_name, perf_records in self.strategy_performance.items():
            if not perf_records:
                continue
                
            # 找到回看期内的性能记录
            recent_records = [
                record for record in perf_records 
                if (datetime.fromisoformat(record['timestamp']) > 
                    datetime.utcnow() - pd.Timedelta(days=lookback_days))
            ]
            
            if not recent_records:
                continue
                
            # 获取最近的性能指标
            latest_perf = recent_records[-1]['performance']
            sharpe_ratio = latest_perf.get('sharpe_ratio', 0)
            
            if sharpe_ratio > best_sharpe:
                best_sharpe = sharpe_ratio
                best_strategy = strategy_name
        
        self.logger.info(f"Best performing strategy (last {lookback_days} days): {best_strategy} with Sharpe: {best_sharpe:.4f}")
        return best_strategy
    
    def should_switch_strategy(self, lookback_days: int = 7) -> bool:
        """
        根据性能判断是否需要切换策略
        :param lookback_days: 回看天数
        :return: 是否需要切换策略
        """
        if not self.active_strategy:
            return False
            
        current_sharpe = 0
        # 获取当前策略最近的性能
        current_perf_records = self.strategy_performance.get(self.active_strategy, [])
        if current_perf_records:
            recent_records = [
                record for record in current_perf_records 
                if (datetime.fromisoformat(record['timestamp']) > 
                    datetime.utcnow() - pd.Timedelta(days=lookback_days))
            ]
            if recent_records:
                current_sharpe = recent_records[-1]['performance'].get('sharpe_ratio', 0)
        
        best_strategy = self.get_best_performing_strategy(lookback_days)
        
        if best_strategy and best_strategy != self.active_strategy:
            best_perf_records = self.strategy_performance.get(best_strategy, [])
            if best_perf_records:
                recent_records = [
                    record for record in best_perf_records 
                    if (datetime.fromisoformat(record['timestamp']) > 
                        datetime.utcnow() - pd.Timedelta(days=lookback_days))
                ]
                if recent_records:
                    best_sharpe = recent_records[-1]['performance'].get('sharpe_ratio', 0)
                    
                    # 如果最佳策略的夏普比率比当前策略高至少0.5，则切换
                    if best_sharpe > current_sharpe + 0.5:
                        self.logger.info(
                            f"Strategy switch recommended: {best_strategy} "
                            f"(Sharpe: {best_sharpe:.4f}) vs {self.active_strategy} "
                            f"(Sharpe: {current_sharpe:.4f})"
                        )
                        return True
        
        return False