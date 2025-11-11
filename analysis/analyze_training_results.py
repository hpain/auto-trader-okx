# analysis/analyze_training_results.py
import pandas as pd
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config

def analyze_training_results():
    """分析最新的训练结果"""
    print("=== 训练结果分析 ===")
    
    # 读取改进版的元数据
    model_dir = config["paths"]["model_dir"]
    improved_metadata_path = os.path.join(model_dir, "improved_metadata.json")
    
    if os.path.exists(improved_metadata_path):
        try:
            with open(improved_metadata_path, 'r', encoding='utf-8') as f:
                improved_metadata = json.load(f)
            
            print("改进版模型信息:")
            print(f"  最佳得分: {improved_metadata.get('best_score', 'N/A')}")
            print(f"  样本数量: {improved_metadata.get('n_samples', 'N/A')}")
            print(f"  训练时间: {improved_metadata.get('training_timestamp', 'N/A')}")
            if 'best_params' in improved_metadata:
                print(f"  置信度阈值: {improved_metadata['best_params'].get('confidence_threshold', 'N/A')}")
                print(f"  学习率: {improved_metadata['best_params'].get('learning_rate', 'N/A')}")
                print(f"  树数量: {improved_metadata['best_params'].get('lgb_n_estimators', 'N/A')}")
            print()
        except Exception as e:
            print(f"读取改进版元数据时出错: {e}")
    
    # 读取最佳试验详情
    best_trial_path = os.path.join(model_dir, "best_trial_details.json")
    if os.path.exists(best_trial_path):
        try:
            with open(best_trial_path, 'r', encoding='utf-8') as f:
                best_trial_data = json.load(f)
            
            print("历史最佳试验:")
            all_time_best = best_trial_data.get("all_time_best", {})
            if all_time_best:
                print(f"  稳定性得分: {all_time_best.get('stability_score', 'N/A')}")
                print(f"  夏普比率: {all_time_best.get('overall_sharpe', 'N/A')}")
                print(f"  时间戳: {all_time_best.get('timestamp', 'N/A')}")
                if 'params' in all_time_best:
                    print(f"  置信度阈值: {all_time_best['params'].get('confidence_threshold', 'N/A')}")
            
            print("\n最近试验:")
            recent_trials = best_trial_data.get("recent_trials", [])
            if recent_trials:
                latest_trial = recent_trials[0]
                print(f"  稳定性得分: {latest_trial.get('stability_score', 'N/A')}")
                print(f"  夏普比率: {latest_trial.get('overall_sharpe', 'N/A')}")
                print(f"  验证集回报: {latest_trial.get('val_return', 'N/A')}")
                print(f"  时间戳: {latest_trial.get('timestamp', 'N/A')}")
                if 'params' in latest_trial:
                    print(f"  置信度阈值: {latest_trial['params'].get('confidence_threshold', 'N/A')}")
            print()
        except Exception as e:
            print(f"读取最佳试验详情时出错: {e}")
    
    # 分析试验总结
    trials_summary_path = os.path.join(model_dir, "improved_trials_summary.csv")
    if os.path.exists(trials_summary_path):
        try:
            df_trials = pd.read_csv(trials_summary_path)
            
            print("试验统计:")
            print(f"  总试验数: {len(df_trials)}")
            positive_scores = len(df_trials[df_trials['stability_score'] > 0])
            print(f"  正得分试验数: {positive_scores}")
            print(f"  最佳稳定性得分: {df_trials['stability_score'].max():.4f}")
            print(f"  平均稳定性得分: {df_trials['stability_score'].mean():.4f}")
            print(f"  得分标准差: {df_trials['stability_score'].std():.4f}")
            print()
            
            # 找到最佳试验
            if len(df_trials) > 0:
                best_idx = df_trials['stability_score'].idxmax()
                best_trial_row = df_trials.loc[best_idx]
                print("最佳试验参数:")
                print(f"  稳定性得分: {best_trial_row['stability_score']:.4f}")
                print(f"  置信度阈值: {best_trial_row['confidence_threshold']:.4f}")
                
                # 解析参数
                import ast
                params = ast.literal_eval(best_trial_row['params'])
                print(f"  学习率: {params.get('learning_rate', 'N/A')}")
                print(f"  树数量: {params.get('lgb_n_estimators', 'N/A')}")
                print(f"  叶子数量: {params.get('num_leaves', 'N/A')}")
        except Exception as e:
            print(f"读取试验总结时出错: {e}")

if __name__ == "__main__":
    analyze_training_results()