import pandas as pd
import numpy as np
import joblib
import json
import os
import time
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
        
        # Tracking for MarketRegimeDetector confidence analysis
        self.last_prediction = 0
        self.last_probability = 0.5

        try:
            self.model = joblib.load(self.model_path)
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            self.features = metadata['feature_cols']
            
            # Prioritize config > metadata > default
            meta_params = metadata.get('best_params', {})
            self.buy_threshold = self.config.get('buy_threshold', meta_params.get('confidence_threshold', 0.55))
            self.sell_threshold = self.config.get('sell_threshold', meta_params.get('sell_threshold', 0.45))

            self.logger.info(f"MODEL LOADED: {self.model_path}")
            self.logger.info(f"Features Used: {len(self.features)}")
            self.logger.info(f"Thresholds -> Buy: {self.buy_threshold:.2f}, Sell: {self.sell_threshold:.2f}")

        except FileNotFoundError as e:
            self.model = None
            self.logger.error(f"FAILED to load model/metadata: {e}. Strategy DISABLED.")
        except Exception as e:
            self.model = None
            self.logger.error(f"UNKNOWN ERROR during init: {e}. Strategy DISABLED.")

    def load_market_context(self):
        """
        读取 LLM Supervisor 生成的市场环境文件
        """
        try:
            # TODO: Path should be configurable
            context_path = "data/market_context.json"
            if not os.path.exists(context_path):
                return None
                
            # Check file age (avoid using stale data > 1 hour)
            mtime = os.path.getmtime(context_path)
            if time.time() - mtime > 3600:
                # self.logger.warning(f"Market context file is stale (>1h old). Ignoring.")
                return None
                
            with open(context_path, 'r') as f:
                context = json.load(f)
            return context
        except Exception as e:
            self.logger.error(f"Failed to load market context: {e}")
            return None

    def _generate_missing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """动态生成缺失的技术指标特征"""
        df = df.copy()
        
        # 基础特征
        if 'return' not in df.columns and 'close' in df.columns:
            df['return'] = df['close'].pct_change()
        if 'vol' not in df.columns and 'volume' in df.columns:
            df['vol'] = df['volume']
            
        # 波动率
        if 'return' in df.columns:
            for period in [10, 20, 60]:
                if f'volatility_{period}' not in df.columns:
                    df[f'volatility_{period}'] = df['return'].rolling(period).std()
        
        # Lag 特征
        for lag in [2, 5, 10]:
            if f'return_lag_{lag}' not in df.columns and 'return' in df.columns:
                df[f'return_lag_{lag}'] = df['return'].shift(lag)
            if f'vol_lag_{lag}' not in df.columns and 'vol' in df.columns:
                df[f'vol_lag_{lag}'] = df['vol'].shift(lag)
        
        # SMA/EMA Spread
        if 'close' in df.columns:
            for short, long in [(10, 50), (20, 100), (50, 100)]:
                col = f'sma_spread_{short}_{long}'
                if col not in df.columns:
                    sma_s = df['close'].rolling(short).mean()
                    sma_l = df['close'].rolling(long).mean()
                    df[col] = (sma_s - sma_l) / (sma_l + 1e-10)
            
            if 'ema_spread_20_50' not in df.columns:
                ema20 = df['close'].ewm(span=20).mean()
                ema50 = df['close'].ewm(span=50).mean()
                df['ema_spread_20_50'] = (ema20 - ema50) / (ema50 + 1e-10)
        
        # RSI
        if 'close' in df.columns:
            for period in [14, 30]:
                col = f'rsi_{period}'
                if col not in df.columns:
                    delta = df['close'].diff()
                    gain = delta.where(delta > 0, 0).rolling(period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
                    rs = gain / (loss + 1e-10)
                    df[col] = 100 - (100 / (1 + rs))
        
        # ADX 代理
        if 'return' in df.columns:
            for period in [14, 20]:
                for col in [f'adx_{period}', f'adx_neg_{period}']:
                    if col not in df.columns:
                        df[col] = df['return'].rolling(period).std() * 100
        
        # Bollinger Width
        if 'close' in df.columns:
            for period in [14, 30]:
                col = f'bb_width_{period}'
                if col not in df.columns:
                    ma = df['close'].rolling(period).mean()
                    std = df['close'].rolling(period).std()
                    df[col] = (2 * std) / (ma + 1e-10)
        
        # MACD
        if 'close' in df.columns and 'macd' not in df.columns:
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # OBV
        if 'obv' not in df.columns and 'close' in df.columns and 'volume' in df.columns:
            obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
            max_val = obv.abs().max()
            df['obv'] = obv / max_val if max_val > 0 else 0
        
        # Vol MA Ratio
        if 'vol_ma_ratio' not in df.columns and 'volume' in df.columns:
            vol_ma = df['volume'].rolling(20).mean()
            df['vol_ma_ratio'] = df['volume'] / (vol_ma + 1e-10)
        
        # ROC
        if 'roc_14' not in df.columns and 'close' in df.columns:
            df['roc_14'] = df['close'].pct_change(14) * 100
        
        # Stochastic K
        if 'stoch_k_14' not in df.columns and all(c in df.columns for c in ['high', 'low', 'close']):
            low_14 = df['low'].rolling(14).min()
            high_14 = df['high'].rolling(14).max()
            df['stoch_k_14'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + 1e-10)
        
        # ATR
        if 'atr_14' not in df.columns and all(c in df.columns for c in ['high', 'low', 'close']):
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr_14'] = tr.rolling(14).mean()
        
        # CCI
        if 'cci_14' not in df.columns and all(c in df.columns for c in ['high', 'low', 'close']):
            tp = (df['high'] + df['low'] + df['close']) / 3
            sma_tp = tp.rolling(14).mean()
            mad = tp.rolling(14).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
            df['cci_14'] = (tp - sma_tp) / (0.015 * mad + 1e-10)
        
        # 时间特征
        if hasattr(df.index, 'dayofweek'):
            if 'day_of_week' not in df.columns:
                df['day_of_week'] = df.index.dayofweek
            if 'hour_of_day' not in df.columns:
                df['hour_of_day'] = df.index.hour
        
        # 衍生品默认值
        if 'funding_rate' not in df.columns:
            df['funding_rate'] = 0
        if 'open_interest' not in df.columns:
            df['open_interest'] = 0
        if 'open_interest_value' not in df.columns:
            df['open_interest_value'] = 0
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        使用加载的模型和完整的特征集为整个DataFrame生成信号。
        结合 LLM Supervisor 进行风险控制。
        """
        # 创建一个与输入DataFrame索引相同的信号DataFrame，默认无信号
        result_df = pd.DataFrame(index=data.index)
        result_df['signal'] = 0

        if self.model is None:
            self.logger.warning(f"WARN [{self.strategy_name}]: 模型未加载，无法生成信号。")
            return result_df
        
        # -------------------------------------------------------------------
        # 🛡️ Missing Feature Handling (Graceful Degradation)
        # -------------------------------------------------------------------
        # Categorize features by reliability
        DERIVATIVES_FEATURES = {
            'open_interest', 'open_interest_value', 'funding_rate',
            'count_toptrader_long_short_ratio', 'sum_toptrader_long_short_ratio',
            'count_long_short_ratio', 'sum_taker_long_short_vol_ratio'
        }
        SENTIMENT_FEATURES = {'sent_mean', 'sent_median', 'count', 'prob_bull', 'prob_bear', 'prob_ranging'}
        
        # Features that can be dynamically generated from OHLCV
        GENERATED_FEATURES = {
            'macd', 'macd_signal', 'macd_hist', 'obv', 'vol_ma_ratio',
            'sma_spread_50_100', 'ema_spread_20_50', 'roc_14', 'stoch_k_14',
            'atr_14', 'cci_14', 'day_of_week', 'hour_of_day', 'adx_neg_14',
            'bb_width_14', 'bb_width_30', 'rsi_14', 'rsi_30',
            'volatility_10', 'volatility_20', 'volatility_60',
            'sma_spread_10_50', 'sma_spread_20_100', 'adx_14', 'adx_neg_20',
            'return', 'return_lag_2', 'return_lag_5', 'return_lag_10',
            'vol', 'vol_lag_2', 'vol_lag_5'
        }
        
        # Generate missing features dynamically
        data = self._generate_missing_features(data)
        
        missing_features = set(self.features) - set(data.columns)
        
        if missing_features:
            # Separate critical vs optional missing
            missing_derivatives = missing_features & DERIVATIVES_FEATURES
            missing_sentiment = missing_features & SENTIMENT_FEATURES
            missing_critical = missing_features - DERIVATIVES_FEATURES - SENTIMENT_FEATURES
            
            if missing_critical:
                self.logger.error(f"CRITICAL [{self.strategy_name}]: 缺少核心特征 {missing_critical}，无法生成信号。")
                return result_df
            
            # Fill missing derivatives/sentiment with defaults
            for feat in missing_derivatives | missing_sentiment:
                if feat not in data.columns:
                    data = data.copy()  # Avoid SettingWithCopyWarning
                    data[feat] = 0.0  # Default to 0 (neutral)
                    
            if missing_derivatives:
                self.logger.warning(f"⚠️ [{self.strategy_name}]: 衍生品特征缺失 {missing_derivatives}，使用默认值0")
            if missing_sentiment:
                self.logger.warning(f"⚠️ [{self.strategy_name}]: 情绪特征缺失 {missing_sentiment}，使用默认值0")

        X = data[self.features].copy()
        
        # Replace infinite/NaN values to avoid crashes
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)  # Fill any remaining NaN with 0
        
        # -------------------------------------------------------------------
        # 🧠 LLM Supervisor Integration
        # -------------------------------------------------------------------
        context = self.load_market_context()
        
        # Default multipliers
        risk_multiplier = 1.0
        bias = "neutral"
        
        if context:
            risk_multiplier = context.get('risk_multiplier', 1.0)
            bias = context.get('bias', 'neutral')
            
        # Ensure risk_multiplier is 0.0 to 1.5
        risk_multiplier = max(0.0, min(1.5, float(risk_multiplier)))
        
        if risk_multiplier == 0.0:
            effective_buy_thresh = 1.01 # Impossible
            effective_sell_thresh = -0.01 # Impossible
        else:
            # Dynamic Threshold Scaling
            # Higher risk multiplier (aggressive) -> Lower threshold
            # Lower risk multiplier (defensive) -> Higher threshold
            
            base_dist_buy = self.buy_threshold - 0.5
            effective_buy_thresh = 0.5 + (base_dist_buy / risk_multiplier)
            
            base_dist_sell = 0.5 - self.sell_threshold
            effective_sell_thresh = 0.5 - (base_dist_sell / risk_multiplier)
            
            # Clip safely
            effective_buy_thresh = min(0.99, max(0.51, effective_buy_thresh))
            effective_sell_thresh = min(0.49, max(0.01, effective_sell_thresh))
        # -------------------------------------------------------------------

        try:
            # 对整个DataFrame进行批量预测
            probabilities = self.model.predict_proba(X)[:, 1]

            # 使用向量化操作根据阈值生成信号
            conditions = [
                (probabilities > effective_buy_thresh),
                (probabilities < effective_sell_thresh)
            ]
            choices = [1, -1]
            raw_signals = np.select(conditions, choices, default=0)
            
            # 🧠 Bias Filtering (Directional Control)
            if bias == 'long':
                raw_signals = np.where(raw_signals == -1, 0, raw_signals)
            elif bias == 'short':
                raw_signals = np.where(raw_signals == 1, 0, raw_signals)
            
            result_df['signal'] = raw_signals
            
            # Store last prediction
            if len(probabilities) > 0:
                self.last_probability = float(probabilities[-1])
                self.last_prediction = int(result_df['signal'].iloc[-1] == 1)
            
        except Exception as e:
            self.logger.error(f"ERROR [{self.strategy_name}]: 模型批量预测时发生错误: {e}")
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
        
    def generate_signal(self, df: pd.DataFrame, **kwargs) -> int:
        """
        为单个时间点生成信号（兼容旧接口）
        """
        signals_df = self.generate_signals(df)
        if not signals_df.empty:
            # 返回最后一个信号
            return int(signals_df['signal'].iloc[-1])
        return 0
