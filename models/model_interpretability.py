"""
模型可解释性模块，使用SHAP等工具解释模型预测
"""
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import joblib
try:
    import shap
except ImportError:
    shap = None
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None
from pathlib import Path

from config import config


class ModelInterpretability:
    """
    模型可解释性类，使用SHAP等工具解释机器学习模型的预测
    """
    
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config.get('model_interpretability', {})
        self.enable_shap = self.config.get('enable_shap', True) and (shap is not None)
        if self.config.get('enable_shap', True) and shap is None:
            self.logger.warning("SHAP library not found. Model interpretability will be disabled.")
        self.shap_sample_size = self.config.get('shap_sample_size', 1000)
        self.feature_importance_top_n = self.config.get('feature_importance_top_n', 10)
        
        self.model_path = config.get('paths', {}).get('model_dir', 'models')
        
        # 确保输出目录存在
        self.output_dir = Path("charts/interpretability")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_shap_explanation(self, model_path: str, X_sample: pd.DataFrame, 
                                model_type: str = "lightgbm") -> Optional[Dict]:
        """
        生成SHAP解释
        
        Args:
            model_path: 模型文件路径
            X_sample: 用于解释的样本数据
            model_type: 模型类型
            
        Returns:
            SHAP解释结果字典
        """
        if not self.enable_shap:
            self.logger.info("SHAP explanations are disabled in config.")
            return None
        
        try:
            # 加载模型
            model = joblib.load(model_path)
            
            # 如果样本太大，进行采样
            if len(X_sample) > self.shap_sample_size:
                sample_indices = np.random.choice(len(X_sample), self.shap_sample_size, replace=False)
                X_shap = X_sample.iloc[sample_indices]
            else:
                X_shap = X_sample
            
            # 根据模型类型选择SHAP解释器
            if model_type.lower() == 'lightgbm':
                import lightgbm as lgb
                if isinstance(model, lgb.Booster):
                    explainer = shap.TreeExplainer(model)
                else:
                    explainer = shap.TreeExplainer(model.model)  # 假设是LGBStrategy的包装
            elif model_type.lower() in ['xgboost', 'xgb']:
                explainer = shap.TreeExplainer(model)
            elif model_type.lower() in ['random_forest', 'rf']:
                explainer = shap.TreeExplainer(model)
            elif model_type.lower() in ['logistic_regression', 'lr']:
                explainer = shap.LinearExplainer(model, X_shap)
            else:
                # 使用通用的KernelExplainer（计算更耗时）
                explainer = shap.KernelExplainer(model.predict, X_shap)
            
            # 计算SHAP值
            self.logger.info(f"Computing SHAP values for {len(X_shap)} samples...")
            shap_values = explainer.shap_values(X_shap)
            
            # 如果是二分类问题，shap_values可能是数组的数组
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # 使用正类的SHAP值
            
            # 计算特征重要性
            feature_importance = np.abs(shap_values).mean(0)
            feature_names = X_shap.columns.tolist()
            
            # 创建特征重要性DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            # 获取最重要的N个特征
            top_features = importance_df.head(self.feature_importance_top_n)
            
            self.logger.info(f"SHAP explanation completed. Top {self.feature_importance_top_n} features:")
            for idx, row in top_features.iterrows():
                self.logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            return {
                'shap_values': shap_values,
                'feature_importance': importance_df,
                'top_features': top_features,
                'explainer_type': type(explainer).__name__,
                'sample_size': len(X_shap)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating SHAP explanation: {e}")
            return None
    
    def plot_shap_summary(self, shap_values, X_sample: pd.DataFrame, 
                         title: str = "SHAP Feature Importance", 
                         output_file: Optional[str] = None) -> Optional[str]:
        """
        绘制SHAP摘要图
        
        Args:
            shap_values: SHAP值
            X_sample: 样本数据
            title: 图表标题
            output_file: 输出文件路径
            
        Returns:
            生成的图表文件路径
        """
        if not self.enable_shap or shap_values is None:
            return None
        
        try:
            # 创建SHAP摘要图
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample, show=False, max_display=self.feature_importance_top_n)
            
            # 保存图表
            if output_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = str(self.output_dir / f"shap_summary_{timestamp}.png")
            
            plt.tight_layout()
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"SHAP summary plot saved to: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error plotting SHAP summary: {e}")
            return None
    
    def plot_shap_waterfall(self, shap_values, X_sample: pd.DataFrame, 
                           instance_idx: int = 0, title: str = "SHAP Waterfall",
                           output_file: Optional[str] = None) -> Optional[str]:
        """
        绘制SHAP瀑布图（解释单个预测）
        
        Args:
            shap_values: SHAP值
            X_sample: 样本数据
            instance_idx: 要解释的实例索引
            title: 图表标题
            output_file: 输出文件路径
            
        Returns:
            生成的图表文件路径
        """
        if not self.enable_shap or shap_values is None:
            return None
        
        try:
            # 选择要解释的实例
            if instance_idx >= len(X_sample):
                instance_idx = 0  # 如果索引超出范围，使用第一个实例
            
            # 确保shap_values是正确的形状
            if len(shap_values.shape) > 1:
                instance_shap_values = shap_values[instance_idx, :]
            else:
                instance_shap_values = shap_values[instance_idx]
            
            # 获取该实例的特征值
            instance_features = X_sample.iloc[instance_idx:instance_idx+1]
            
            # 创建瀑布图
            plt.figure(figsize=(10, 8))
            shap.plots.waterfall(
                shap.Explanation(
                    values=instance_shap_values,
                    base_values=np.mean(shap_values) if len(shap_values.shape) > 1 else shap_values.mean(),
                    data=instance_features.iloc[0].values,
                    feature_names=instance_features.columns.tolist()
                ),
                show=False
            )
            
            # 保存图表
            if output_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = str(self.output_dir / f"shap_waterfall_{instance_idx}_{timestamp}.png")
            
            plt.tight_layout()
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"SHAP waterfall plot saved to: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error plotting SHAP waterfall: {e}")
            return None
    
    def generate_feature_importance_report(self, model_path: str, X_sample: pd.DataFrame,
                                         model_type: str = "lightgbm") -> Optional[Dict]:
        """
        生成特征重要性报告
        
        Args:
            model_path: 模型文件路径
            X_sample: 用于解释的样本数据
            model_type: 模型类型
            
        Returns:
            特征重要性报告
        """
        try:
            shap_results = self.generate_shap_explanation(model_path, X_sample, model_type)
            if shap_results is None:
                return None
            
            # 生成摘要图
            summary_plot_path = self.plot_shap_summary(
                shap_results['shap_values'], 
                X_sample,
                output_file=str(self.output_dir / f"feature_importance_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            )
            
            # 生成几个瀑布图来解释特定的预测
            waterfall_plots = []
            for idx in range(min(3, len(X_sample))):  # 解释前3个预测
                waterfall_path = self.plot_shap_waterfall(
                    shap_results['shap_values'],
                    X_sample,
                    instance_idx=idx,
                    output_file=str(self.output_dir / f"prediction_explanation_{idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                )
                if waterfall_path:
                    waterfall_plots.append(waterfall_path)
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'model_path': model_path,
                'feature_importance': shap_results['feature_importance'].to_dict('records'),
                'top_features': shap_results['top_features'].to_dict('records'),
                'summary_plot_path': summary_plot_path,
                'waterfall_plots': waterfall_plots,
                'sample_size': shap_results['sample_size']
            }
            
            # 保存报告
            report_path = self.output_dir / f"feature_importance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            import json
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Feature importance report saved to: {report_path}")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating feature importance report: {e}")
            return None
    
    def integrate_with_trading_system(self, model_path: str, X_current: pd.DataFrame,
                                    model_type: str = "lightgbm") -> Optional[Dict]:
        """
        与交易系统集成，为当前预测提供解释
        
        Args:
            model_path: 模型文件路径
            X_current: 当前特征数据
            model_type: 模型类型
            
        Returns:
            包含解释信息的字典
        """
        if not self.enable_shap:
            return None
        
        try:
            # 简单的SHAP解释，为当前预测提供特征贡献
            model = joblib.load(model_path)
            
            # 使用训练数据的一个小子集来构建explainer（如果可用）
            # 这里使用当前数据作为近似
            if len(X_current) > 100:
                X_background = X_current.sample(n=100)
            else:
                X_background = X_current
            
            # 为当前数据点生成解释
            # 这里简化处理，实际上可能需要从历史数据中获取背景数据
            explainer = shap.Explainer(model.predict, X_background)
            shap_values = explainer(X_current)
            
            # 提取最重要的特征贡献
            if hasattr(shap_values, 'values'):
                current_shap_values = shap_values.values[-1]  # 最后一个数据点（当前）
                feature_names = X_current.columns.tolist()
                
                feature_contributions = list(zip(feature_names, current_shap_values))
                # 按绝对值排序
                feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                
                top_contributions = feature_contributions[:self.feature_importance_top_n]
                
                interpretation = {
                    'timestamp': datetime.now().isoformat(),
                    'model_type': model_type,
                    'top_feature_contributions': top_contributions,
                    'prediction_explanation': {
                        'feature_count': len(feature_contributions),
                        'most_positive_contributor': max(feature_contributions, key=lambda x: x[1]),
                        'most_negative_contributor': min(feature_contributions, key=lambda x: x[1])
                    }
                }
                
                self.logger.info("Model interpretation completed for current prediction")
                return interpretation
            else:
                self.logger.warning("Could not extract SHAP values")
                return None
                
        except Exception as e:
            self.logger.error(f"Error integrating interpretability with trading system: {e}")
            return None


# 全局实例
_model_interpretability = None


def get_model_interpretability(config: Dict) -> ModelInterpretability:
    """
    获取或创建模型可解释性实例
    """
    global _model_interpretability
    if _model_interpretability is None:
        _model_interpretability = ModelInterpretability(config)
    return _model_interpretability