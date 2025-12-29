"""
增强版因子挖掘模块，基于改进计划实施优化
"""
import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor, SymbolicClassifier
from gplearn.functions import make_function
from gplearn.fitness import make_fitness
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import os
import json
import warnings
warnings.filterwarnings('ignore')

# --- Monkey Patch for gplearn compatibility with sklearn >= 1.6 ---
import gplearn.genetic
def _validate_data_shim(self, X, y=None, **kwargs):
    # Determine the correct validation method
    if hasattr(self, 'validate_data'):
        return self.validate_data(X, y, **kwargs)
    else:
        from sklearn.utils.validation import check_X_y, check_array
        if y is not None:
            X_out, y_out = check_X_y(X, y, **kwargs)
            if hasattr(X_out, 'shape'):
                self.n_features_in_ = X_out.shape[1]
            return X_out, y_out
        else:
            X_out = check_array(X, **kwargs)
            if hasattr(X_out, 'shape'):
                self.n_features_in_ = X_out.shape[1]
            return X_out

if not hasattr(gplearn.genetic.SymbolicClassifier, '_validate_data'):
    gplearn.genetic.SymbolicClassifier._validate_data = _validate_data_shim
if not hasattr(gplearn.genetic.SymbolicRegressor, '_validate_data'):
    gplearn.genetic.SymbolicRegressor._validate_data = _validate_data_shim
# ------------------------------------------------------------------

# 自定义金融函数
def protected_div(x1, x2):
    """保护除法，避免除以零"""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > 0.001, x1 / x2, 1.0)

def protected_log(x1):
    """保护对数，处理负数和零值"""
    return np.log(np.abs(x1) + 1e-10)

def protected_sqrt(x1):
    """保护平方根，处理负数"""
    return np.sqrt(np.abs(x1))

def momentum(x1, x2):
    """动量函数：当前价格相对过去某个时期的变动"""
    return (x1 - x2) / (np.abs(x2) + 1e-10)

def volatility(close, volume):
    """波动率函数"""
    returns = np.diff(close, prepend=close[0])
    vol = np.abs(returns)
    return np.where(np.isnan(vol), 0, vol)

# 注册自定义函数
protected_div_func = make_function(function=protected_div, name='pdiv', arity=2)
protected_log_func = make_function(function=protected_log, name='plog', arity=1)
protected_sqrt_func = make_function(function=protected_sqrt, name='psqrt', arity=1)
momentum_func = make_function(function=momentum, name='momentum', arity=2)
volatility_func = make_function(function=volatility, name='volatility', arity=2)

# --- 自定义适应度函数: 多头查准率 (Bull Precision) ---
def _bull_precision(y, y_pred, w):
    """
    自定义适应度函数，专注于最大化'做多'信号的准确率。
    我们假设 y_pred > 0 为预测做多 (Class 1)。
    """
    # 将包含NaN的预测视为不做交易 (False)
    # y_pred 是原始公式输出
    preds = np.where(np.isnan(y_pred), 0, y_pred) > 0
    
    # 真实值 (y) 应该是 0 或 1
    # 只需要计算 预测为1 且 真实为1 的比例
    tp = np.sum((y == 1) & preds)
    fp = np.sum((y == 0) & preds)
    
    denom = tp + fp
    
    # 如果没有开单，给一个极低分
    if denom == 0:
        return 0.0
    
    precision = tp / denom
    
    # 惩罚项：如果开单数太少 (例如少于总样本的 0.5%)，则进行惩罚
    # 这是为了防止模型只通过 1 次运气好的交易就拿到 100% 准确率
    coverage = denom / len(y)
    if coverage < 0.005: 
        return 0.0 # 给予极刑，直接归零
        
    return precision

bull_precision_fitness = make_fitness(function=_bull_precision, greater_is_better=True)


class EnhancedFeatureMiner:
    """
    增强版因子挖掘器，基于改进计划实施优化
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = None  # 可以设置为logging实例
        
        # 从改进计划中提取参数
        self.generations = self.config.get('generations', 100)  # 增加代数
        self.population_size = self.config.get('population_size', 5000)  # 增加种群规模
        self.function_set = self.config.get('function_set', [
            'add', 'sub', 'mul', protected_div_func,  # 使用保护除法对象
            protected_sqrt_func, protected_log_func,  # 使用保护函数对象
            'abs', 'neg', 'max', 'min',
            momentum_func  # 自定义动量函数对象
        ])
        self.parsimony_coefficient = self.config.get('parsimony_coefficient', 0.0005)
        self.max_samples = self.config.get('max_samples', 0.9)
        self.test_size = self.config.get('test_size', 0.2)
        
        # 预测目标类型
        self.target_type = self.config.get('target_type', 'classification')  # 'classification' 或 'regression'
        
        # 评估方法
        self.use_time_series_cv = self.config.get('use_time_series_cv', True)
        self.cv_splits = self.config.get('cv_splits', 5)
        
        # 优化目标 (Metric)
        self.metric = self.config.get('metric', 'accuracy') # 'accuracy' or 'bull_precision'
    
    def load_and_prepare_data(self, data_path: str = None, data: pd.DataFrame = None):
        """
        加载和准备数据
        """
        if data is not None:
            self.data = data.copy()
        else:
            # 加载数据
            try:
                if data_path is None:
                    # Default path
                    data_path = 'data/history/binance_BTCUSDT_1h_4y.csv'
                
                print(f"Loading data from: {data_path}")
                
                if data_path.endswith('.parquet'):
                    try:
                        self.data = pd.read_parquet(data_path) # Auto-detect (prefers pyarrow usually)
                    except Exception as e:
                         print(f"Failed to read parquet with default engine, trying fastparquet explicitly: {e}")
                         self.data = pd.read_parquet(data_path, engine='fastparquet')
                    
                    print(f"Successfully loaded Parquet file with shape: {self.data.shape}")
                    
                    # Parquet files from our cache usually have 'y' target column
                    if 'y' in self.data.columns:
                        print("Found existing target column 'y', mapping to 'target'.")
                        self.data['target'] = self.data['y']
                        # Remove future lookahead cols to prevent leakage during mining
                        cols_to_drop = ['future_close', 'future_open', 'future_high', 'future_low', 'future_ret', 'y']
                        self.data.drop(columns=[c for c in cols_to_drop if c in self.data.columns], inplace=True)
                    
                    # Ensure index is datetime
                    if not isinstance(self.data.index, pd.DatetimeIndex):
                         if 'timestamp' in self.data.columns:
                            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
                            self.data.set_index('timestamp', inplace=True)
                         elif 'date' in self.data.columns:
                            self.data['date'] = pd.to_datetime(self.data['date'])
                            self.data.set_index('date', inplace=True)

                else:
                    self.data = pd.read_csv(data_path)
                    self.data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
                    self.data = self.data.sort_values('timestamp').set_index('timestamp')
                    print("CSV Data loaded successfully.")
                    
            except FileNotFoundError:
                print(f"Error: Data file not found at {data_path}")
                # Create mock data
                dates = pd.date_range(start='2020-01-01', periods=2000, freq='H')
                self.data = pd.DataFrame({
                    'timestamp': dates,
                    'open': np.random.rand(2000) * 10000 + 30000,
                    'high': np.random.rand(2000) * 10000 + 30000,
                    'low': np.random.rand(2000) * 10000 + 30000,
                    'close': np.random.rand(2000) * 10000 + 30000,
                    'volume': np.random.rand(2000) * 100000 + 10000
                })
                self.data = self.data.sort_values('timestamp').set_index('timestamp')
        
        # Calculate additional technical indicators only if basic columns present and no extended features
        # If loading from Parquet cache, it likely already has many features.
        # We check a representative column like 'RSI_14' or lower case 'rsi_14'
        if 'rsi_14' not in self.data.columns and 'RSI_14' not in self.data.columns:
             print("Calculating base technical indicators...")
             self._calculate_technical_indicators()
        else:
             print("Skipping base indicator calculation (features already present).")
        
        # Prepare target if not already present
        if 'target' not in self.data.columns:
            self._prepare_target()
        
        # Clean data
        self.data = self.data.replace([np.inf, -np.inf], np.nan)
        self.data = self.data.dropna()
        
        return self.data
    
    def _calculate_technical_indicators(self):
        """
        计算技术指标
        """
        # 基础指标
        self.data['SMA_5'] = self.data['close'].rolling(window=5).mean()
        self.data['SMA_10'] = self.data['close'].rolling(window=10).mean()
        self.data['SMA_20'] = self.data['close'].rolling(window=20).mean()
        self.data['SMA_50'] = self.data['close'].rolling(window=50).mean()
        self.data['SMA_100'] = self.data['close'].rolling(window=100).mean()
        
        # RSI指标
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI_14'] = 100 - (100 / (1 + rs))
        
        # MACD指标
        exp1 = self.data['close'].ewm(span=12).mean()
        exp2 = self.data['close'].ewm(span=26).mean()
        self.data['MACD'] = exp1 - exp2
        self.data['MACD_signal'] = self.data['MACD'].ewm(span=9).mean()
        self.data['MACD_histogram'] = self.data['MACD'] - self.data['MACD_signal']
        
        # 布林带
        self.data['BB_middle'] = self.data['close'].rolling(window=20).mean()
        bb_std = self.data['close'].rolling(window=20).std()
        self.data['BB_upper'] = self.data['BB_middle'] + (bb_std * 2)
        self.data['BB_lower'] = self.data['BB_middle'] - (bb_std * 2)
        self.data['BB_width'] = self.data['BB_upper'] - self.data['BB_lower']
        self.data['BB_position'] = (self.data['close'] - self.data['BB_lower']) / self.data['BB_width']
        
        # ATR (平均真实波幅)
        high_low = self.data['high'] - self.data['low']
        high_close = np.abs(self.data['high'] - self.data['close'].shift())
        low_close = np.abs(self.data['low'] - self.data['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        self.data['ATR'] = true_range.rolling(window=14).mean()
        
        # 成交量指标
        self.data['volume_sma'] = self.data['volume'].rolling(window=20).mean()
        self.data['volume_ratio'] = self.data['volume'] / self.data['volume_sma']
        
        # 价格变动率
        self.data['roc_10'] = self.data['close'].pct_change(periods=10)
        self.data['roc_5'] = self.data['close'].pct_change(periods=5)
        
        # 高低价差
        self.data['hl_ratio'] = self.data['high'] / self.data['low']
        self.data['oc_ratio'] = self.data['open'] / self.data['close']
        
        # 收盘价相对于开盘价的位置
        self.data['candle_body'] = self.data['close'] - self.data['open']
        self.data['candle_range'] = self.data['high'] - self.data['low']
        self.data['body_ratio'] = self.data['candle_body'] / self.data['candle_range']
    
    def _prepare_target(self):
        """
        准备预测目标 (基于改进计划)
        """
        if self.target_type == 'classification':
            # 分类：预测价格是上涨还是下跌
            self.data['target'] = np.sign(self.data['close'].shift(-5) - self.data['close'])  # 5小时后
            # 移除0值（无变化）
            self.data = self.data[self.data['target'] != 0]
            print("分类目标: 预测5小时后价格涨跌方向")
        else:
            # 回归：预测价格回报率
            self.data['target'] = (self.data['close'].shift(-5) - self.data['close']) / self.data['close']
            print("回归目标: 预测5小时后价格回报率")
    
    def mine_features(self):
        """
        执行因子挖掘
        """
        print(f"\n开始进行遗传编程因子挖掘 ({self.target_type} 模式)...")
        print(f"参数设置 - 代数: {self.generations}, 种群规模: {self.population_size}")
        
        # 准备特征和目标
        feature_cols = [col for col in self.data.columns 
                       if col not in ['timestamp', 'target'] and not col.startswith('target')]
        X = self.data[feature_cols].values
        y = self.data['target'].values
        
        # 使用时间序列交叉验证或简单划分
        if self.use_time_series_cv:
            print("使用时间序列交叉验证...")
            tscv = TimeSeriesSplit(n_splits=self.cv_splits)
            results = []
            fold = 0
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # 根据目标类型选择模型
                if self.target_type == 'classification':
                    gp = SymbolicClassifier(
                        generations=self.generations,
                        population_size=self.population_size,
                        function_set=self.function_set,
                        parsimony_coefficient=self.parsimony_coefficient,
                        max_samples=self.max_samples,
                        verbose=1,
                        random_state=42,
                        metric=bull_precision_fitness if self.metric == 'bull_precision' else 'log loss',
                        stopping_criteria=1.1, # 强制跑完所有代数
                        n_jobs=1  # 在交叉验证中避免多进程冲突
                    )
                else:
                    gp = SymbolicRegressor(
                        generations=self.generations,
                        population_size=self.population_size,
                        function_set=self.function_set,
                        parsimony_coefficient=self.parsimony_coefficient,
                        max_samples=self.max_samples,
                        verbose=1,
                        random_state=42,
                        stopping_criteria=1.1, # 强制跑完所有代数
                        n_jobs=1  # 在交叉验证中避免多进程冲突
                    )
                
                print(f"正在处理第 {fold+1}/{self.cv_splits} 折...")
                gp.fit(X_train, y_train)
                
                # 计算性能指标
                y_pred = gp.predict(X_test)
                if self.target_type == 'classification':
                    accuracy = accuracy_score(y_test, y_pred)
                    try:
                        auc_score = roc_auc_score(y_test, y_pred)
                    except:
                        auc_score = 0.5  # 如果计算失败，使用随机猜测的AUC
                    
                    fold_result = {
                        'fold': fold,
                        'accuracy': accuracy,
                        'auc': auc_score,
                        'best_program': str(gp._program)
                    }
                else:
                    r2 = r2_score(y_test, y_pred)
                    fold_result = {
                        'fold': fold,
                        'r2_score': r2,
                        'best_program': str(gp._program)
                    }
                
                results.append(fold_result)
                fold += 1
            
            # 选择最佳结果
            if self.target_type == 'classification':
                best_fold = max(results, key=lambda x: x['accuracy'])
            else:
                best_fold = max(results, key=lambda x: x.get('r2_score', x.get('accuracy', 0)))
        else:
            # 使用简单划分
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=42, 
                stratify=y if self.target_type == 'classification' else None
            )
            
            # 训练遗传编程模型
            if self.target_type == 'classification':
                gp = SymbolicClassifier(
                    generations=self.generations,
                    population_size=self.population_size,
                    function_set=self.function_set,
                    parsimony_coefficient=self.parsimony_coefficient,
                    max_samples=self.max_samples,
                    verbose=1,
                    random_state=42,
                    metric=bull_precision_fitness if self.metric == 'bull_precision' else 'log loss',
                    stopping_criteria=1.1 # 强制跑完所有代数
                )
            else:
                gp = SymbolicRegressor(
                    generations=self.generations,
                    population_size=self.population_size,
                    function_set=self.function_set,
                    parsimony_coefficient=self.parsimony_coefficient,
                    max_samples=self.max_samples,
                    verbose=1,
                    random_state=42,
                    stopping_criteria=1.1 # 强制跑完所有代数
                )
            
            gp.fit(X_train, y_train)
            
            # 计算性能指标
            y_pred = gp.predict(X_test)
            if self.target_type == 'classification':
                accuracy = accuracy_score(y_test, y_pred)
                try:
                    auc_score = roc_auc_score(y_test, y_pred)
                except:
                    auc_score = 0.5  # 如果计算失败，使用随机猜测的AUC
                
                best_fold = {
                    'accuracy': accuracy,
                    'auc': auc_score,
                    'best_program': str(gp._program)
                }
            else:
                r2 = r2_score(y_test, y_pred)
                best_fold = {
                    'r2_score': r2,
                    'best_program': str(gp._program)
                }
        
        print(f"\n最佳结果: {best_fold}")
        
        # 训练在全部数据上的最终模型
        if self.target_type == 'classification':
            final_gp = SymbolicClassifier(
                generations=self.generations,
                population_size=self.population_size,
                function_set=self.function_set,
                parsimony_coefficient=self.parsimony_coefficient,
                max_samples=self.max_samples,
                verbose=1,
                random_state=42,
                metric=bull_precision_fitness if self.metric == 'bull_precision' else 'log loss',
                stopping_criteria=1.1 # 强制跑完所有代数
            )
        else:
            final_gp = SymbolicRegressor(
                generations=self.generations,
                population_size=self.population_size,
                function_set=self.function_set,
                parsimony_coefficient=self.parsimony_coefficient,
                max_samples=self.max_samples,
                verbose=1,
                random_state=42,
                stopping_criteria=1.1 # 强制跑完所有代数
            )
        
        final_gp.fit(X, y)
        
        # 评估最终模型
        y_pred_final = final_gp.predict(X)
        if self.target_type == 'classification':
            final_accuracy = accuracy_score(y, y_pred_final)
            print(f"最终模型准确率: {final_accuracy:.4f}")
        else:
            final_r2 = r2_score(y, y_pred_final)
            print(f"最终模型R²分数: {final_r2:.4f}")
        
        # 保存最佳因子
        self.best_program = final_gp._program
        self.best_program_str = str(final_gp._program)
        
        return final_gp, best_fold
    
    def evaluate_with_benchmark(self, X, y):
        """
        使用基准模型评估性能
        """
        print("\n评估基准模型性能...")
        
        # 根据目标类型使用相应的基准模型
        if self.target_type == 'classification':
            # 分类基准：随机森林
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            # 使用时间序列分割
            tscv = TimeSeriesSplit(n_splits=self.cv_splits)
            rf_scores = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                rf_scores.append(score)
            
            avg_rf_score = np.mean(rf_scores)
            print(f"基准随机森林平均准确率: {avg_rf_score:.4f}")
        else:
            # 回归基准：随机森林
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            tscv = TimeSeriesSplit(n_splits=self.cv_splits)
            rf_scores = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)
                score = r2_score(y_test, y_pred)
                rf_scores.append(score)
            
            avg_rf_score = np.mean(rf_scores)
            print(f"基准随机森林平均R²分数: {avg_rf_score:.4f}")
        
        return avg_rf_score
    
    def save_mined_features(self, output_path: str = None):
        """
        保存挖掘的因子
        """
        if not hasattr(self, 'best_program_str'):
            print("错误: 没有可保存的因子，请先运行 mine_features()")
            return
        
        if output_path is None:
            timestamp = pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')
            output_path = f'models/enhanced_mined_features_{timestamp}.json'
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 获取基础特征列表
        feature_cols = [col for col in self.data.columns 
                       if col not in ['timestamp', 'target'] and not col.startswith('target')]
        
        # Convert function set to serializable list of strings
        serializable_function_set = []
        for f in self.function_set:
            if hasattr(f, 'name'):
                serializable_function_set.append(f.name)
            elif isinstance(f, str):
                serializable_function_set.append(f)
            else:
                serializable_function_set.append(str(f))

        mined_feature_data = {
            "name": f"enhanced_gp_feature_{pd.Timestamp.utcnow().strftime('%Y%m%d%H%M')}",
            "formula": self.best_program_str,
            "base_features": feature_cols,
            "target_type": self.target_type,
            "function_set": serializable_function_set,
            "params": {
                "generations": self.generations,
                "population_size": self.population_size,
                "parsimony_coefficient": self.parsimony_coefficient
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(mined_feature_data, f, indent=4, ensure_ascii=False)
        
        print(f"\n已将发现的最佳因子保存到: {output_path}")
        return output_path


import argparse
import sys

def run_enhanced_feature_mining(data_path: str = None, config: dict = None):
    """
    运行增强版因子挖掘
    """
    if config is None:
        config = {
            'generations': 20,
            'population_size': 1000,
            'target_type': 'classification',  # 使用分类而非回归
            'use_time_series_cv': True,
            'cv_splits': 5
        }
    
    # 创建增强版因子挖掘器
    miner = EnhancedFeatureMiner(config)
    
    # 加载和准备数据
    data = miner.load_and_prepare_data(data_path)
    
    # 执行因子挖掘
    gp_model, best_result = miner.mine_features()
    
    # 比较基准模型
    feature_cols = [col for col in data.columns 
                   if col not in ['timestamp', 'target'] and not col.startswith('target')]
    X = data[feature_cols].values
    y = data['target'].values
    
    benchmark_score = miner.evaluate_with_benchmark(X, y)
    
    # 保存结果
    output_path = miner.save_mined_features()
    
    print("\n--- 最终总结 ---")
    if miner.target_type == 'classification':
        gp_score = best_result.get('accuracy', 0)
        if gp_score > benchmark_score:
            print(f"遗传编程分类器性能 ({gp_score:.4f}) 优于基准 ({benchmark_score:.4f})")
        else:
            print(f"遗传编程分类器性能 ({gp_score:.4f}) 未超越基准 ({benchmark_score:.4f})")
    else:
        gp_score = best_result.get('r2_score', 0)
        if gp_score > benchmark_score:
            print(f"遗传编程回归器性能 (R²={gp_score:.4f}) 优于基准 (R²={benchmark_score:.4f})")
        else:
            print(f"遗传编程回归器性能 (R²={gp_score:.4f}) 未超越基准 (R²={benchmark_score:.4f})")
    
    print(f"最佳因子公式: {miner.best_program_str}")
    print(f"结果已保存至: {output_path}")
    
    return miner, gp_model, best_result


# 如果直接运行此脚本
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Enhanced Feature Mining')
    parser.add_argument('--data_path', type=str, default=None, help='Path to input data (CSV or Parquet)')
    parser.add_argument('--generations', type=int, default=20, help='Number of generations for GP')
    parser.add_argument('--population', type=int, default=1000, help='Population size for GP')
    parser.add_argument('--cv_splits', type=int, default=5, help='Number of CV splits')
    parser.add_argument('--target_type', type=str, default='classification', choices=['classification', 'regression'], help='Target type')
    parser.add_argument('--metric', type=str, default='accuracy', choices=['accuracy', 'bull_precision'], help='Optimization metric')
    
    args = parser.parse_args()
    
    config = {
        'generations': args.generations,
        'population_size': args.population,
        'target_type': args.target_type,
        'metric': args.metric,
        'use_time_series_cv': True,
        'cv_splits': args.cv_splits
    }
    
    try:
        miner, model, result = run_enhanced_feature_mining(args.data_path, config)
    except Exception as e:
        print(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)