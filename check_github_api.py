import requests
import json

# 从 .git/config 读取仓库信息
try:
    with open('.git/config', 'r') as f:
        config = f.read()
        # 提取仓库 URL
        for line in config.split('\n'):
            if 'url = ' in line and 'github.com' in line:
                url = line.split('url = ')[1].strip()
                print(f"Repository URL: {url}")
                
                # 解析仓库信息
                if 'github.com/' in url:
                    parts = url.split('github.com/')[1].replace('.git', '').split('/')
                    owner = parts[0]
                    repo = parts[1]
                    print(f"Owner: {owner}, Repo: {repo}")
                    
                    # 获取最新提交
                    api_url = f"https://api.github.com/repos/{owner}/{repo}/commits/main_lt"
                    print(f"\nFetching from: {api_url}")
                    
                    response = requests.get(api_url, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        remote_sha = data['sha']
                        commit_message = data['commit']['message']
                        commit_date = data['commit']['author']['date']
                        
                        print(f"\nRemote latest commit:")
                        print(f"  SHA: {remote_sha}")
                        print(f"  Message: {commit_message}")
                        print(f"  Date: {commit_date}")
                        
                        # 读取本地 HEAD
                        with open('.git/refs/heads/main_lt', 'r') as f:
                            local_sha = f.read().strip()
                        
                        print(f"\nLocal HEAD:")
                        print(f"  SHA: {local_sha}")
                        
                        if local_sha == remote_sha:
                            print("\n✅ Local branch is UP TO DATE with remote")
                        else:
                            print("\n⚠️ Local branch is BEHIND remote")
                            print(f"   Need to pull {len(remote_sha)} commits")
                    else:
                        print(f"API request failed: {response.status_code}")
                        print(response.text)
                break
                
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
