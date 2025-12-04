import subprocess
import os

# 设置环境变量禁用 pager
env = os.environ.copy()
env['GIT_PAGER'] = 'cat'
env['PAGER'] = 'cat'

try:
    # Fetch 远程更新
    print("Fetching from origin...")
    result = subprocess.run(
        ['git', 'fetch', 'origin'],
        capture_output=True,
        text=True,
        env=env,
        timeout=30
    )
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    # 检查本地和远程的差异
    print("\n" + "="*50)
    print("Checking differences...")
    result = subprocess.run(
        ['git', 'log', 'HEAD..origin/main_lt', '--oneline'],
        capture_output=True,
        text=True,
        env=env,
        timeout=10
    )
    
    if result.stdout.strip():
        print("New commits on origin/main_lt:")
        print(result.stdout)
    else:
        print("Local branch is up to date with origin/main_lt")
    
    # 显示当前状态
    print("\n" + "="*50)
    print("Current status:")
    result = subprocess.run(
        ['git', 'status', '-sb'],
        capture_output=True,
        text=True,
        env=env,
        timeout=10
    )
    print(result.stdout)
    
except subprocess.TimeoutExpired:
    print("Command timed out")
except Exception as e:
    print(f"Error: {e}")
