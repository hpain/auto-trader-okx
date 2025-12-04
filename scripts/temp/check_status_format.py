"""
快速验证executor更新是否正确集成
"""
import json
import os
from datetime import datetime


def check_status_file():
    """检查status.json文件格式是否正确更新"""
    
    # 检查是否有status.json文件
    if os.path.exists("status.json"):
        try:
            with open("status.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            
            print("Current status.json contents:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            
            # 检查新增的字段
            expected_fields = [
                "risk_events", 
                "latest_risk_event", 
                "active_risk_alerts", 
                "latest_risk_alert",
                "performance_summary"
            ]
            
            print("\nChecking for new fields in status.json:")
            for field in expected_fields:
                if field in data:
                    print(f"  ✓ {field}: {data[field]}")
                else:
                    print(f"  ✗ {field}: MISSING")
                    
        except Exception as e:
            print(f"Error reading status.json: {e}")
    else:
        print("No status.json file found. This is expected if the executor hasn't run yet.")


if __name__ == "__main__":
    check_status_file()