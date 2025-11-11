# analysis/auto_adaptation_system.py
import pandas as pd
import numpy as np
import sys
import os
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from data.binance import get_klines_bian
from trader.okx_client import OKXClient
from utils.data_normalization import normalize_binance_df
from features.feature_engineering import generate_features, make_supervised
from models.evolution import train_evolve

class AutoAdaptationSystem:
    def __init__(self):
        """初始化自适应系统"""
        self.config = config
        self.model_dir = config["paths"]["model_dir"]
        self.performance_history_file = os.path.join(self.model_dir, "performance_history.json")
        
    def detect_regime_change(self, df, window_size=60, threshold=0.1):
        """
        检测市场模式变化
        """
        # 计算滚动统计量
        df['returns'] = df['close'].pct_change()
        
        # 使用窗口对比，检查统计特性是否有显著变化
        mid_point = len(df) // 2
        early_period = df.iloc[:mid_point]['returns'].dropna()
        late_period = df.iloc[mid_point:]['returns'].dropna()
        
        if len(early_period) < 10 or len(late_period) < 10:
            return False, "Not enough data"
        
        # 检查多个指标
        mean_change = abs(late_period.mean() - early_period.mean()) / (abs(early_period.mean()) + 1e-8)
        std_change = abs(late_period.std() - early_period.std()) / (early_period.std() + 1e-8)
        skew_change = abs(late_period.skew() - early_period.skew())
        kurt_change = abs(late_period.kurtosis() - early_period.kurtosis())
        
        # 综合判断
        change_score = (mean_change + std_change + skew_change/10 + kurt_change/10) / 4
        has_change = change_score > threshold
        
        return has_change, {
            'change_score': change_score,
            'mean_change': mean_change,
            'std_change': std_change,
            'skew_change': skew_change,
            'kurt_change': kurt_change
        }
    
    def update_parameters_based_on_regime(self, has_change, change_info):
        """
        基于市场模式变化更新参数
        """
        if not has_change:
            # 默认参数
            return {
                'profit_threshold': 0.005,
                'confidence_threshold': 0.95,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.05,
                'n_trials': 100,
                'top_k_features': 20
            }
        
        # 根据变化程度调整参数
        change_level = change_info['change_score']
        
        # 变化越大，越保守
        confidence_threshold = max(0.6, 0.95 - change_level * 0.2)
        profit_threshold = max(0.002, 0.005 - change_level * 0.002)
        
        # 如果波动性增加，调整止损和止盈
        if change_info['std_change'] > 0.1:
            stop_loss_pct = min(0.03, 0.02 + change_info['std_change'] * 0.05)
            take_profit_pct = max(0.03, 0.05 - change_info['std_change'] * 0.02)
        else:
            stop_loss_pct = 0.02
            take_profit_pct = 0.05
        
        # 增加试验次数以找到更优参数
        n_trials = min(200, 100 + int(change_level * 50))
        
        # 减少特征数量以避免过拟合
        top_k_features = max(10, 20 - int(change_level * 5))
        
        return {
            'profit_threshold': profit_threshold,
            'confidence_threshold': confidence_threshold,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'n_trials': n_trials,
            'top_k_features': top_k_features
        }
    
    def should_retrain(self, performance_history, threshold_improvement=0.05, min_retrain_interval=7):
        """
        判断是否需要重新训练
        """
        if not performance_history:
            return True
        
        # 检查最后几次性能是否下降
        recent_performances = [record['performance'] for record in performance_history[-5:] if 'performance' in record]
        
        if len(recent_performances) < 3:
            return True
        
        # 计算性能趋势
        recent_avg = np.mean(recent_performances[-3:])
        earlier_avg = np.mean(recent_performances[:-3][-3:]) if len(recent_performances) > 5 else recent_avg
        
        # 如果性能显著下降，需要重新训练
        if earlier_avg > recent_avg and (earlier_avg - recent_avg) / earlier_avg > threshold_improvement:
            return True
        
        # 检查上次重新训练时间
        if performance_history:
            last_retrain = datetime.fromisoformat(performance_history[-1]['timestamp'])
            if (datetime.now() - last_retrain).days >= min_retrain_interval:
                return True
        
        return False
    
    def save_performance_record(self, record):
        """
        保存性能记录
        """
        history = []
        if os.path.exists(self.performance_history_file):
            with open(self.performance_history_file, 'r', encoding='utf-8') as f:
                try:
                    history = json.load(f)
                except:
                    history = []
        
        history.append(record)
        
        with open(self.performance_history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    
    def load_data_with_cache(self, symbol, interval, years=2):
        """
        加载数据，尝试使用缓存
        """
        # 尝试加载特征缓存
        from features.feature_engineering import generate_features, make_supervised
        
        # Feature Caching Logic (类似research/evolve.py)
        import hashlib
        config_str = f"{symbol}-{interval}-{years}-None"  # news_csv_path
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:10]
        cache_dir = config["paths"]["feature_cache_dir"]
        os.makedirs(cache_dir, exist_ok=True)
        feature_cache_path = os.path.join(cache_dir, f"features_{config_hash}.parquet")

        if os.path.exists(feature_cache_path):
            print(f"Found feature cache, loading from {feature_cache_path}")
            dfm = pd.read_parquet(feature_cache_path)
        else:
            print("No feature cache found, generating features...")
            client = OKXClient(**config["okx"])
            dfp = get_klines_bian(client, symbol, interval, years=years, ignore_local=True)
            if dfp is None or dfp.empty:
                print("Failed to get price data")
                return None
            dfp = normalize_binance_df(dfp)
            dfm = generate_features(dfp, news_csv_path=None)
            dfm.to_parquet(feature_cache_path)
            print(f"Features saved to {feature_cache_path}")
        
        return dfm
    
    def run_auto_adaptation(self):
        """
        执行自适应流程
        """
        print("=== 自适应系统启动 ===")
        
        # 加载数据
        symbol = config.get("trade", {}).get("symbol", "BTC-USDT")
        interval = config.get("trade", {}).get("interval", "1H")
        
        df = self.load_data_with_cache(symbol, interval, years=2)
        if df is None:
            print("无法加载数据，退出自适应流程")
            return
        
        # 检测市场模式变化
        print("检测市场模式变化...")
        has_change, change_info = self.detect_regime_change(df)
        
        print(f"市场模式变化检测结果: {'检测到变化' if has_change else '未检测到显著变化'}")
        if has_change:
            print(f"变化得分: {change_info['change_score']:.3f}")
            print(f"均值变化: {change_info['mean_change']:.3f}")
            print(f"标准差变化: {change_info['std_change']:.3f}")
            print(f"偏度变化: {change_info['skew_change']:.3f}")
            print(f"峰度变化: {change_info['kurt_change']:.3f}")
        
        # 获取调整后的参数
        params = self.update_parameters_based_on_regime(has_change, change_info)
        print(f"使用参数: {params}")
        
        # 加载性能历史
        performance_history = []
        if os.path.exists(self.performance_history_file):
            with open(self.performance_history_file, 'r', encoding='utf-8') as f:
                try:
                    performance_history = json.load(f)
                except:
                    performance_history = []
        
        # 判断是否需要重新训练
        should_retrain = self.should_retrain(performance_history)
        print(f"是否需要重新训练: {'是' if should_retrain else '否'}")
        
        if should_retrain:
            print("开始重新训练模型...")
            
            # 准备训练数据
            data = make_supervised(df, horizon=1, threshold=params['profit_threshold'])
            
            non_feature_cols = ["ts", "dt", "y", "future_high", "future_low", "future_close", "future_ret", "date", "timestamp", "vol_ccy", "vol_ccy_quote", "confirm"]
            feature_cols = [c for c in data.columns if c not in non_feature_cols]
            
            # 数据清洗
            data.replace([np.inf, -np.inf], np.nan, inplace=True)
            data.dropna(subset=feature_cols + ["y"], inplace=True)
            
            if len(data) < 500:
                print("样本太少，无法训练")
                return
            
            # 特征预选择
            if params['top_k_features'] and params['top_k_features'] > 0 and params['top_k_features'] < len(feature_cols):
                from sklearn.feature_selection import SelectKBest, f_classif
                X = data[feature_cols]
                y = data["y"]
                
                selector = SelectKBest(f_classif, k=params['top_k_features'])
                selector.fit(X, y)
                
                selected_mask = selector.get_support()
                new_feature_cols = X.columns[selected_mask]
                
                print(f"选择的特征: {new_feature_cols.tolist()}")
                feature_cols = new_feature_cols.tolist()
            
            # 执行训练
            best_score, best_params = train_evolve(
                data=data,
                feature_cols=feature_cols,
                out_dir=self.model_dir,
                n_trials=params['n_trials'],
                model_list=['lgb'],  # 简化只用lgb
                profit_threshold=params['profit_threshold'],
                confidence_threshold=params['confidence_threshold'],
                stop_loss_pct=params['stop_loss_pct'],
                max_drawdown_limit=0.1,
                success_rate_threshold=0.75,
                take_profit_pct=params['take_profit_pct'],
                interval=interval,
            )
            
            # 记录训练结果
            record = {
                'timestamp': datetime.now().isoformat(),
                'has_regime_change': bool(has_change),  # Convert to Python bool
                'change_info': change_info,
                'used_params': {k: float(v) if isinstance(v, np.number) else v for k, v in params.items()},  # Convert numpy types
                'best_score': float(best_score) if best_score is not None else -1.0,
                'best_params': best_params,
                'performance': float(best_score) if best_score is not None else -1.0
            }
            
            self.save_performance_record(record)
            
            if best_score is not None:
                print(f"训练完成，最佳得分: {best_score:.4f}")
                print(f"最佳参数: {best_params}")
            else:
                print("训练未成功完成")
        else:
            print("跳过重新训练，使用现有模型")
        
        print("=== 自适应系统完成 ===")

if __name__ == "__main__":
    auto_system = AutoAdaptationSystem()
    auto_system.run_auto_adaptation()