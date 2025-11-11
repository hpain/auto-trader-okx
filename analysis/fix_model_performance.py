# analysis/fix_model_performance.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
import subprocess

def fix_model_performance():
    """修复模型性能问题"""
    print("=== 修复模型性能问题 ===")
    
    # 建议的参数调整
    print("1. 调整训练参数:")
    print("   - 增加最大深度: 从3增加到6")
    print("   - 增加试验次数: 从54增加到100")
    print("   - 使用完整历史数据: 2年而非1年")
    print("   - 调整置信度范围: 0.55-0.65")
    print()
    
    # 建议的训练命令
    cmd = [
        "python", "research/improved_evolution.py",
        "--years", "2",          # 使用完整2年数据
        "--trials", "100",        # 增加试验次数
        "--min-confidence", "0.55",  # 调整置信度范围下限
        "--max-confidence", "0.65",  # 调整置信度范围上限
        "--profit-threshold", "0.005",  # 略微提高盈利阈值
        "--stop-loss-pct", "0.02",      # 保持止损
        "--take-profit-pct", "0.05",    # 保持止盈
        "--models", "lgb",              # 使用LightGBM
        "--ignore-local"               # 获取最新数据
    ]
    
    print("2. 推荐训练命令:")
    print("   " + " ".join(cmd))
    print()
    
    # 执行训练
    print("3. 开始训练...")
    try:
        # 使用 subprocess 运行命令
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 实时输出训练过程
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # 等待进程完成
        process.wait()
        
        if process.returncode == 0:
            print("\n✅ 训练完成!")
        else:
            print(f"\n❌ 训练失败，返回码: {process.returncode}")
            stderr_output = process.stderr.read()
            if stderr_output:
                print(f"错误信息:\n{stderr_output}")
                
    except Exception as e:
        print(f"❌ 训练过程中出现异常: {e}")

if __name__ == "__main__":
    fix_model_performance()