import subprocess
import os

env = os.environ.copy()
env['GIT_PAGER'] = 'cat'

try:
    print("=" * 60)
    print("检查远程 run_autonomous.py 是否有更新...")
    print("=" * 60)
    
    # 1. Fetch 远程
    print("\n1. 获取远程最新代码...")
    result = subprocess.run(
        ['git', 'fetch', 'origin'],
        capture_output=True,
        text=True,
        env=env,
        timeout=30
    )
    print("✅ Fetch 完成")
    
    # 2. 检查 run_autonomous.py 是否有更新
    print("\n2. 检查 run_autonomous.py 更新...")
    result = subprocess.run(
        ['git', 'log', 'HEAD..origin/main_lt', '--oneline', '--', 'run_autonomous.py'],
        capture_output=True,
        text=True,
        env=env,
        timeout=10
    )
    
    if result.stdout.strip():
        print("⚠️ 远程有 run_autonomous.py 的更新:")
        print(result.stdout)
        
        # 显示差异统计
        print("\n3. 差异统计:")
        result = subprocess.run(
            ['git', 'diff', 'HEAD', 'origin/main_lt', '--stat', '--', 'run_autonomous.py'],
            capture_output=True,
            text=True,
            env=env,
            timeout=10
        )
        print(result.stdout)
        
        # 检查是否有 Bug Fix 相关的改动
        print("\n4. 检查远程是否包含 Bug Fix:")
        result = subprocess.run(
            ['git', 'show', 'origin/main_lt:run_autonomous.py'],
            capture_output=True,
            text=True,
            env=env,
            timeout=10
        )
        
        content = result.stdout
        bug_fixes = [
            "BUG FIX 1",
            "BUG FIX 2", 
            "BUG FIX 3",
            "BUG FIX 4",
            "BUG FIX 5"
        ]
        
        for bug_fix in bug_fixes:
            if bug_fix in content:
                print(f"  ✅ 远程包含: {bug_fix}")
            else:
                print(f"  ❌ 远程不包含: {bug_fix}")
        
        print("\n" + "=" * 60)
        print("结论: 远程有更新,需要合并")
        print("=" * 60)
        
    else:
        print("✅ 远程没有 run_autonomous.py 的更新")
        print("\n结论: 可以直接提交本地改动")
    
    # 3. 检查其他文件
    print("\n" + "=" * 60)
    print("检查其他修改文件...")
    print("=" * 60)
    
    result = subprocess.run(
        ['git', 'log', 'HEAD..origin/main_lt', '--oneline', '--', 'config/settings.yaml.example'],
        capture_output=True,
        text=True,
        env=env,
        timeout=10
    )
    
    if result.stdout.strip():
        print("⚠️ 远程有 config/settings.yaml.example 的更新")
    else:
        print("✅ 远程没有 config/settings.yaml.example 的更新")
    
except subprocess.TimeoutExpired:
    print("命令超时")
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
