import subprocess
import os

# 设置环境变量禁用 pager
env = os.environ.copy()
env['GIT_PAGER'] = 'cat'

try:
    # 检查未提交的改变
    print("=" * 60)
    print("检查未提交的改变...")
    print("=" * 60)
    
    result = subprocess.run(
        ['git', 'status', '--short'],
        capture_output=True,
        text=True,
        env=env,
        timeout=10
    )
    
    if result.stdout.strip():
        print("\n未提交的文件:")
        print(result.stdout)
        
        # 统计文件类型
        lines = result.stdout.strip().split('\n')
        new_files = [l for l in lines if l.startswith('??')]
        modified_files = [l for l in lines if l.startswith(' M') or l.startswith('M ')]
        deleted_files = [l for l in lines if l.startswith(' D') or l.startswith('D ')]
        
        print(f"\n统计:")
        print(f"  新增文件: {len(new_files)}")
        print(f"  修改文件: {len(modified_files)}")
        print(f"  删除文件: {len(deleted_files)}")
        
        # 显示修改的文件详情
        if modified_files:
            print("\n" + "=" * 60)
            print("修改的文件详情:")
            print("=" * 60)
            for line in modified_files:
                filename = line.split()[-1]
                print(f"\n文件: {filename}")
                
                # 显示差异
                diff_result = subprocess.run(
                    ['git', 'diff', filename],
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=10
                )
                
                if diff_result.stdout:
                    # 只显示前50行
                    diff_lines = diff_result.stdout.split('\n')[:50]
                    print('\n'.join(diff_lines))
                    if len(diff_result.stdout.split('\n')) > 50:
                        print("\n... (差异太长,已截断)")
    else:
        print("\n✅ 工作区干净,没有未提交的改变")
    
    # 检查未跟踪的文件
    print("\n" + "=" * 60)
    print("未跟踪的文件 (新文件):")
    print("=" * 60)
    
    result = subprocess.run(
        ['git', 'ls-files', '--others', '--exclude-standard'],
        capture_output=True,
        text=True,
        env=env,
        timeout=10
    )
    
    if result.stdout.strip():
        untracked = result.stdout.strip().split('\n')
        for f in untracked[:20]:  # 只显示前20个
            print(f"  {f}")
        if len(untracked) > 20:
            print(f"  ... 还有 {len(untracked) - 20} 个文件")
    else:
        print("  (无)")
    
except subprocess.TimeoutExpired:
    print("命令超时")
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
