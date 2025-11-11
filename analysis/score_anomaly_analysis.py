# analysis/score_anomaly_analysis.py
import pandas as pd
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config

def analyze_score_anomaly():
    """分析模型得分异常的原因"""
    print("=== 模型得分异常分析 ===")
    
    model_dir = config["paths"]["model_dir"]
    
    # 读取试验总结文件
    improved_summary_path = os.path.join(model_dir, "improved_trials_summary.csv")
    original_summary_path = os.path.join(model_dir, "trials_summary.csv")
    
    print("1. 改进版模型试验得分分布:")
    if os.path.exists(improved_summary_path):
        df_improved = pd.read_csv(improved_summary_path)
        print(f"   总试验数: {len(df_improved)}")
        print(f"   正得分试验数: {len(df_improved[df_improved['stability_score'] > 0])}")
        print(f"   最高得分: {df_improved['stability_score'].max():.4f}")
        print(f"   平均得分: {df_improved['stability_score'].mean():.4f}")
        print(f"   得分标准差: {df_improved['stability_score'].std():.4f}")
        print(f"   得分中位数: {df_improved['stability_score'].median():.4f}")
        print()
        
        # 显示得分分布
        print("   得分分布:")
        score_ranges = [
            (-10, -1),
            (-1, 0),
            (0, 0.5),
            (0.5, 1),
            (1, 2),
            (2, 10)
        ]
        
        for low, high in score_ranges:
            count = len(df_improved[(df_improved['stability_score'] >= low) & (df_improved['stability_score'] < high)])
            if count > 0:
                print(f"     [{low}, {high}): {count} 试验")
        print()
    
    print("2. 原始模型试验得分分布:")
    if os.path.exists(original_summary_path):
        df_original = pd.read_csv(original_summary_path)
        print(f"   总试验数: {len(df_original)}")
        print(f"   正得分试验数: {len(df_original[df_original['stability_score'] > 0])}")
        print(f"   最高得分: {df_original['stability_score'].max():.4f}")
        print(f"   平均得分: {df_original['stability_score'].mean():.4f}")
        print(f"   得分标准差: {df_original['stability_score'].std():.4f}")
        print(f"   得分中位数: {df_original['stability_score'].median():.4f}")
        print()
        
        # 显示得分分布
        print("   得分分布:")
        score_ranges = [
            (-10, -1),
            (-1, 0),
            (0, 0.5),
            (0.5, 1),
            (1, 2),
            (2, 10)
        ]
        
        for low, high in score_ranges:
            count = len(df_original[(df_original['stability_score'] >= low) & (df_original['stability_score'] < high)])
            if count > 0:
                print(f"     [{low}, {high}): {count} 试验")
        print()
    
    print("3. 得分计算公式差异:")
    print("   原始模型:")
    print("     stability_score = mean_sharpe - std_sharpe")
    print("   改进模型:")
    print("     stability_score = mean_sharpe - (0.5 * std_sharpe)")
    print("   关键差异: 改进模型减少了对标准差的惩罚")
    print()
    
    print("4. 可能的问题分析:")
    print("   a) 得分公式修改的影响:")
    print("      - 原始: 如果mean=1.0, std=0.5, score=0.5")
    print("      - 改进: 如果mean=1.0, std=0.5, score=0.75")
    print("      - 影响: 得分可能被放大")
    print()
    
    print("   b) 数据分布变化:")
    print("      - 改进模型使用2年而非3年数据")
    print("      - 可能选择了更容易预测的时间段")
    print()
    
    print("   c) 验证方法差异:")
    print("      - 原始模型: 交叉验证得分")
    print("      - 改进模型: 交叉验证得分 + holdout验证得分")
    print("      - 可能报告的是不同类型的得分")
    print()
    
    print("5. 验证方法检查:")
    best_trial_path = os.path.join(model_dir, "best_trial_details.json")
    if os.path.exists(best_trial_path):
        with open(best_trial_path, 'r', encoding='utf-8') as f:
            best_trial_data = json.load(f)
        
        all_time_best = best_trial_data.get("all_time_best", {})
        if all_time_best:
            print("   历史最佳记录:")
            print(f"     稳定性得分: {all_time_best.get('stability_score', 'N/A')}")
            print(f"     夏普比率: {all_time_best.get('overall_sharpe', 'N/A')}")
            print(f"     时间戳: {all_time_best.get('timestamp', 'N/A')}")
            print()
    
    print("6. 可能的问题:")
    print("   [WARNING] 得分从<0.1跃升到0.97确实异常")
    print("   [WARNING] 需要检查是否是:")
    print("     1. 不同的得分计算方法")
    print("     2. 不同的数据集")
    print("     3. 得分归一化问题")
    print("     4. 代码bug导致的数值错误")
    print()
    
    print("7. 建议验证:")
    print("   1. 对比相同数据集上的得分")
    print("   2. 检查得分计算公式的实现")
    print("   3. 验证是否真的有更好的实盘表现")
    print("   4. 进行回测验证")

if __name__ == "__main__":
    analyze_score_anomaly()