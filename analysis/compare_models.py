# analysis/compare_models.py
import pandas as pd
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config

def compare_models():
    """比较不同模型的性能"""
    print("=== 模型性能对比分析 ===")
    
    model_dir = config["paths"]["model_dir"]
    
    # 读取历史最佳模型信息
    best_trial_path = os.path.join(model_dir, "best_trial_details.json")
    if os.path.exists(best_trial_path):
        with open(best_trial_path, 'r', encoding='utf-8') as f:
            best_trial_data = json.load(f)
        
        print("历史最佳模型 (2025-09-07):")
        all_time_best = best_trial_data.get("all_time_best", {})
        if all_time_best:
            print(f"  稳定性得分: {all_time_best.get('stability_score', 'N/A'):.4f}")
            print(f"  夏普比率: {all_time_best.get('overall_sharpe', 'N/A'):.4f}")
            print(f"  置信度阈值: {all_time_best['params'].get('confidence_threshold', 'N/A'):.4f}")
            print(f"  树数量: {all_time_best['params'].get('n_estimators', 'N/A')}")
            print(f"  学习率: {all_time_best['params'].get('learning_rate', 'N/A'):.6f}")
        print()
    
    # 读取改进版模型信息
    improved_metadata_path = os.path.join(model_dir, "improved_metadata.json")
    if os.path.exists(improved_metadata_path):
        with open(improved_metadata_path, 'r', encoding='utf-8') as f:
            improved_metadata = json.load(f)
        
        print("最新改进版模型 (2025-09-29):")
        print(f"  稳定性得分: {improved_metadata.get('best_score', 'N/A'):.4f}")
        if 'best_params' in improved_metadata:
            print(f"  置信度阈值: {improved_metadata['best_params'].get('confidence_threshold', 'N/A'):.4f}")
            print(f"  树数量: {improved_metadata['best_params'].get('lgb_n_estimators', 'N/A')}")
            print(f"  学习率: {improved_metadata['best_params'].get('learning_rate', 'N/A'):.6f}")
            print(f"  叶子数量: {improved_metadata['best_params'].get('num_leaves', 'N/A')}")
            print(f"  最大深度: {improved_metadata['best_params'].get('max_depth', 'N/A')}")
        print()
    
    # 性能对比
    print("=== 性能对比 ===")
    if all_time_best and improved_metadata:
        old_score = all_time_best.get('stability_score', 0)
        new_score = improved_metadata.get('best_score', 0)
        score_diff = new_score - old_score
        
        print(f"  稳定性得分变化: {old_score:.4f} → {new_score:.4f} ({score_diff:+.4f})")
        if score_diff < 0:
            print("  [WARNING] 最新模型性能下降")
        else:
            print("  [OK] 最新模型性能提升")
        
        # 参数变化
        old_conf = all_time_best['params'].get('confidence_threshold', 0)
        new_conf = improved_metadata['best_params'].get('confidence_threshold', 0)
        print(f"  置信度阈值变化: {old_conf:.4f} → {new_conf:.4f}")
        
        old_lr = all_time_best['params'].get('learning_rate', 0)
        new_lr = improved_metadata['best_params'].get('learning_rate', 0)
        print(f"  学习率变化: {old_lr:.6f} → {new_lr:.6f}")

if __name__ == "__main__":
    compare_models()