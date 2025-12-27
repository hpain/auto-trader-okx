import pandas as pd
import numpy as np
import joblib
import json
from strategies.base_strategy import BaseStrategy

import logging

class LGBStrategy(BaseStrategy):
    """
    一个完全向量化的LightGBM模型预测策略，符合BaseStrategy标准。
    它从元数据动态加载模型、特征列表和预测阈值。
    """

    def __init__(self, strategy_name: str, config: dict):
        """
        初始化LGB策略。

        :param strategy_name: 策略的唯一名称。
        :param config: 包含'model_dir'等路径的配置字典。
        """
        super().__init__(strategy_name, config)
        model_dir = self.config.get('model_dir', 'models')
        model_name = self.config.get('model_name', 'best_model.pkl')
        metadata_name = self.config.get('metadata_name', 'metadata.json')

        self.model_path = f"{model_dir}/{model_name}"
        self.metadata_path = f"{model_dir}/{metadata_name}"
        
        self.logger = logging.getLogger(f"strategy.{strategy_name}")

        try:
            self.model = joblib.load(self.model_path)
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            self.features = metadata['feature_cols']
            self.buy_threshold = metadata.get('best_params', {}).get('confidence_threshold', 0.55)
            # Add sell threshold (default to 0.45 or slightly lower than buy threshold)
            self.sell_threshold = metadata.get('best_params', {}).get('sell_threshold', 0.45)

            self.logger.info(f"MODEL LOADED: {self.model_path}")
            self.logger.info(f"Features Used: {len(self.features)}")
            self.logger.info(f"Thresholds -> Buy: {self.buy_threshold:.2f}, Sell: {self.sell_threshold:.2f}")

        except FileNotFoundError as e:
            self.model = None
            self.logger.error(f"FAILED to load model/metadata: {e}. Strategy DISABLED.")
        except Exception as e:
            self.model = None
            self.logger.error(f"UNKNOWN ERROR during init: {e}. Strategy DISABLED.")

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        使用加载的模型和完整的特征集为整个DataFrame生成信号。
        假定传入的DataFrame `data` 已经包含了所有必要的特征。
        """
        # 创建一个与输入DataFrame索引相同的信号DataFrame，默认无信号
        result_df = pd.DataFrame(index=data.index)
        result_df['signal'] = 0

        if self.model is None:
            print(f"WARN [{self.strategy_name}]: 模型未加载，无法生成信号。")
            return result_df
        
        # 确保所有需要的特征都存在
        missing_features = set(self.features) - set(data.columns)
        if missing_features:
            print(f"WARN [{self.strategy_name}]: 数据中缺少以下必要特征: {missing_features}，无法生成信号。")
            return result_df

        X = data[self.features].copy()

        # Replace infinite values with NaN to avoid crashes
        X.replace([np.inf, -np.inf], np.nan, inplace=True)

        # LightGBM handles NaNs natively. 
        # CRITICAL FIX: Do NOT fill with 0, as 0 can be a valid but extreme feature value (e.g. price=0).
        # We rely on the model's native missing value handling.
        
        # Log if we have significant missing data
        null_counts = X.isnull().sum()
        if null_counts.sum() > 0:
             cols_with_nulls = null_counts[null_counts > 0].index.tolist()
             # Only log if it's not spammy - maybe check logger level or just print once
             # print(f"WARN [{self.strategy_name}]: Input data contains NaNs in columns: {cols_with_nulls}. Relying on model's native handling.")
             pass
        
        try:
            # 对整个DataFrame进行批量预测
            # probabilities将是一个二维数组，第二列是类别为1（上涨）的概率
            probabilities = self.model.predict_proba(X)[:, 1]

            # 使用向量化操作根据阈值生成信号
            # 1: Buy (Prob > Buy Threshold)
            # -1: Sell (Prob < Sell Threshold)
            # 0: Hold (Between thresholds)
            conditions = [
                (probabilities > self.buy_threshold),
                (probabilities < self.sell_threshold)
            ]
            choices = [1, -1]
            result_df['signal'] = np.select(conditions, choices, default=0)
            
        except Exception as e:
            print(f"ERROR [{self.strategy_name}]: 模型批量预测时发生错误: {e}")
            # 出错时返回HOLD信号
            result_df['signal'] = 0

        return result_df

    def get_strategy_info(self) -> dict:
        """
        获取策略信息
        """
        return {
            'strategy_name': self.strategy_name,
            'strategy_type': 'LGBM',
            'model_path': self.model_path,
            'features_count': len(self.features) if hasattr(self, 'features') else 0,
            'buy_threshold': self.buy_threshold if hasattr(self, 'buy_threshold') else 0.5
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
