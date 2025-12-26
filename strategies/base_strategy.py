"""
基础策略接口，定义所有交易策略应实现的方法
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd


class BaseStrategy(ABC):
    """
    交易策略基类，定义所有策略的接口
    """
    
    def __init__(self, strategy_name: str, config: Dict[str, Any]):
        self.strategy_name = strategy_name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{strategy_name}")
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            data: 包含价格和其他指标的数据框
            
        Returns:
            包含交易信号的数据框
        """
        pass
    
    @abstractmethod
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        获取策略信息
        
        Returns:
            策略相关信息的字典
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        验证输入数据是否有效
        
        Args:
            data: 输入数据
            
        Returns:
            数据是否有效
        """
        if data is None or data.empty:
            self.logger.error("输入数据为空或为 None")
            return False
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            self.logger.error(f"缺少必要列: {missing_columns}")
            return False
        
        return True
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        准备数据，执行基本的数据清理和预处理
        
        Args:
            data: 原始数据
            
        Returns:
            预处理后的数据
        """
        processed_data = data.copy()
        
        # 确保索引是日期时间类型
        if not isinstance(processed_data.index, pd.DatetimeIndex):
            if 'timestamp' in processed_data.columns:
                processed_data['timestamp'] = pd.to_datetime(processed_data['timestamp'])
                processed_data.set_index('timestamp', inplace=True)
        
        # 处理缺失值
        processed_data.ffill(inplace=True)
        processed_data.fillna(0, inplace=True)
        
        return processed_data