"""
ML Performance Feedback Module

记录 ML 模型的预测表现，供 LLM Supervisor 参考决策。
使用文件存储，与现有 market_context.json 通信方式保持一致。
"""

import os
import json
import time
import logging
from datetime import datetime
from collections import deque
from typing import Optional
import numpy as np

class MLPerformanceFeedback:
    """
    记录 ML 模型的预测表现，供 LLM Supervisor 参考。
    
    数据流向: ML Bot → ml_performance.json → LLM Supervisor
    """
    
    def __init__(self, file_path: str = "data/ml_performance.json", max_history: int = 100):
        self.file_path = file_path
        self.max_history = max_history
        self.logger = logging.getLogger("MLFeedback")
        
        # 内存中的历史记录
        self.predictions = deque(maxlen=max_history)
        
        # 加载已有数据
        self._load()
    
    def _load(self):
        """从文件加载历史数据"""
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, 'r') as f:
                    data = json.load(f)
                    history = data.get('history', [])
                    # 只保留最近的 max_history 条
                    for record in history[-self.max_history:]:
                        self.predictions.append(record)
                self.logger.debug(f"Loaded {len(self.predictions)} prediction records")
        except Exception as e:
            self.logger.warning(f"Failed to load ml_performance.json: {e}")
    
    def record_prediction(
        self, 
        symbol: str,
        prediction: int,  # 1=buy, -1=sell, 0=hold
        probability: float,
        actual_return: Optional[float] = None,
        is_correct: Optional[bool] = None
    ):
        """
        记录单次预测结果
        
        Args:
            symbol: 交易对
            prediction: 预测方向 (1=买, -1=卖, 0=持有)
            probability: 模型置信度 (0-1)
            actual_return: 实际收益 (可后续更新)
            is_correct: 预测是否正确 (可后续更新)
        """
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "prediction": prediction,
            "probability": probability,
            "actual_return": actual_return,
            "is_correct": is_correct
        }
        self.predictions.append(record)
        self.logger.debug(f"Recorded prediction: {symbol} dir={prediction} prob={probability:.3f}")
    
    def update_outcome(self, symbol: str, actual_return: float):
        """
        更新最近一次该 symbol 的预测结果
        
        Args:
            symbol: 交易对
            actual_return: 实际收益率
        """
        # 反向遍历找到最近的未更新记录
        for record in reversed(self.predictions):
            if record['symbol'] == symbol and record['actual_return'] is None:
                record['actual_return'] = actual_return
                # 判断是否正确: 预测方向与实际收益方向一致
                if record['prediction'] != 0:
                    is_correct = (record['prediction'] > 0) == (actual_return > 0)
                    record['is_correct'] = is_correct
                self.logger.debug(f"Updated outcome for {symbol}: return={actual_return:.4f}")
                break
    
    def get_summary(self, window: int = 50) -> dict:
        """
        获取最近 N 次预测的汇总统计
        
        Returns:
            {
                "total_predictions": int,
                "win_rate": float,
                "avg_confidence": float,
                "confidence_calibration": float,  # 置信度校准 (接近1表示校准良好)
                "avg_return": float,
                "trend": "improving" | "stable" | "declining",
                "last_update": str
            }
        """
        # 获取最近 window 条有结果的预测
        recent = [p for p in list(self.predictions)[-window:] if p.get('is_correct') is not None]
        
        if len(recent) < 5:
            return {
                "total_predictions": len(self.predictions),
                "evaluated_predictions": len(recent),
                "win_rate": None,
                "avg_confidence": None,
                "message": "Insufficient data for statistics",
                "last_update": datetime.utcnow().isoformat()
            }
        
        # 计算统计指标
        wins = sum(1 for p in recent if p['is_correct'])
        win_rate = wins / len(recent)
        
        avg_confidence = np.mean([p['probability'] for p in recent])
        
        # 置信度校准: 理想情况下，高置信度预测应该有更高的胜率
        high_conf_preds = [p for p in recent if p['probability'] > 0.6]
        high_conf_win_rate = sum(1 for p in high_conf_preds if p['is_correct']) / len(high_conf_preds) if high_conf_preds else 0
        
        returns = [p['actual_return'] for p in recent if p['actual_return'] is not None]
        avg_return = np.mean(returns) if returns else 0.0
        
        # 趋势判断: 比较最近 10 次与之前 10 次的胜率
        if len(recent) >= 20:
            recent_10 = recent[-10:]
            prev_10 = recent[-20:-10]
            recent_wr = sum(1 for p in recent_10 if p['is_correct']) / 10
            prev_wr = sum(1 for p in prev_10 if p['is_correct']) / 10
            
            if recent_wr > prev_wr + 0.1:
                trend = "improving"
            elif recent_wr < prev_wr - 0.1:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "total_predictions": len(self.predictions),
            "evaluated_predictions": len(recent),
            "win_rate": round(win_rate, 3),
            "avg_confidence": round(avg_confidence, 3),
            "high_conf_win_rate": round(high_conf_win_rate, 3),
            "avg_return": round(avg_return, 5),
            "trend": trend,
            "last_update": datetime.utcnow().isoformat()
        }
    
    def save(self):
        """保存到 JSON 文件 (原子写入)"""
        try:
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            
            data = {
                "summary": self.get_summary(),
                "history": list(self.predictions),
                "last_save": datetime.utcnow().isoformat()
            }
            
            # 原子写入
            temp_file = self.file_path + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(temp_file, self.file_path)
            
            self.logger.debug(f"Saved {len(self.predictions)} records to {self.file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save ml_performance.json: {e}")


# 全局单例 (方便在交易循环中使用)
_feedback_instance: Optional[MLPerformanceFeedback] = None

def get_ml_feedback() -> MLPerformanceFeedback:
    """获取全局 ML 反馈实例"""
    global _feedback_instance
    if _feedback_instance is None:
        _feedback_instance = MLPerformanceFeedback()
    return _feedback_instance
