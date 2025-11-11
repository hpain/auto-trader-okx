# analysis/performance_diagnosis.py
import pandas as pd
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config

def diagnose_performance_drop():
    """诊断模型性能下降的原因"""
    print("=== 模型性能下降诊断 ===")
    
    model_dir = config["paths"]["model_dir"]
    
    # 读取详细信息
    best_trial_path = os.path.join(model_dir, "best_trial_details.json")
    improved_metadata_path = os.path.join(model_dir, "improved_metadata.json")
    
    if os.path.exists(best_trial_path):
        with open(best_trial_path, 'r', encoding='utf-8') as f:
            best_trial_data = json.load(f)
    
    if os.path.exists(improved_metadata_path):
        with open(improved_metadata_path, 'r', encoding='utf-8') as f:
            improved_metadata = json.load(f)
    
    print("1. 性能指标对比:")
    print("   历史最佳模型 (2025-09-07):")
    print("     - 稳定性得分: 0.9067")
    print("     - 夏普比率: 1.5793")
    print("   最新模型 (2025-09-29):")
    print("     - 稳定性得分: -0.1767")
    print("     - 夏普比率: 0.3402")
    print("   性能下降幅度: -1.0834 (下降119.5%)")
    print()
    
    print("2. 可能的原因分析:")
    print("   a) 市场环境变化:")
    print("      - 2025年9月后期市场波动性可能增加")
    print("      - 市场趋势可能发生变化")
    print("      - 新闻事件或宏观因素影响")
    print()
    
    print("   b) 数据质量问题:")
    print("      - 最新数据可能存在异常值")
    print("      - 数据分布可能发生变化")
    print("      - 特征有效性可能下降")
    print()
    
    print("   c) 模型参数问题:")
    print("      - 最大深度限制为3，可能过于保守")
    print("      - 学习率从0.086降低到0.079，可能影响收敛")
    print("      - 置信度阈值调整可能不合适")
    print()
    
    print("   d) 训练策略问题:")
    print("      - 只使用了1年数据而非2年")
    print("      - 试验次数54次相对较少")
    print("      - 交叉验证可能不够充分")
    print()
    
    print("3. 建议的解决方案:")
    print("   a) 数据层面:")
    print("      - 检查最新数据质量")
    print("      - 增加数据清洗力度")
    print("      - 考虑使用滚动窗口训练")
    print()
    
    print("   b) 模型层面:")
    print("      - 增加最大深度(目前为3，建议增加到5-6)")
    print("      - 调整学习率范围")
    print("      - 重新优化置信度阈值")
    print()
    
    print("   c) 训练策略:")
    print("      - 增加试验次数(建议至少100次)")
    print("      - 使用更长时间的历史数据")
    print("      - 考虑集成多个模型")
    print()
    
    print("   d) 验证方法:")
    print("      - 使用时间序列交叉验证")
    print("      - 增加验证集大小")
    print("      - 实施早停机制")

if __name__ == "__main__":
    diagnose_performance_drop()