
"""
测试新的风险监控功能
"""
import tempfile
import os
from datetime import datetime, timedelta
from trader.trade_tracker import TradeTracker
from trader.risk_monitor import RiskMonitor


def test_risk_monitoring():
    """测试风险监控功能"""
    
    # 创建临时数据库用于测试
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db.close()
    
    try:
        # 初始化跟踪器和监控器
        tracker = TradeTracker(db_path=temp_db.name)
        monitor = RiskMonitor(
            db_path=temp_db.name,
            config={
                'daily_loss_threshold': -100.0,
                'max_consecutive_losses': 3,
                'max_single_loss_pct': -0.02,
                'min_account_balance': 500.0
            }
        )
        
        print("开始测试风险监控功能...")
        
        # 模拟交易记录
        print("\n1. 记录几笔正常盈利交易...")
        tracker.record_trade_entry("trade_1", "BTC-USDT", "buy", 50000.0, 0.1)
        tracker.record_trade_exit("trade_1", 50010.0)  # 小幅盈利
        
        tracker.record_trade_entry("trade_2", "BTC-USDT", "buy", 50010.0, 0.1)
        tracker.record_trade_exit("trade_2", 50020.0)  # 小幅盈利
        
        tracker.record_trade_entry("trade_3", "BTC-USDT", "buy", 50020.0, 0.1)
        tracker.record_trade_exit("trade_3", 50015.0)  # 小幅亏损
        
        # 检查监控结果
        performance = monitor.get_performance_summary()
        print(f"当前表现: {performance}")
        
        # 应该没有风险事件
        alerts = monitor.check_risk_conditions(current_balance=10000.0)
        print(f"风险告警数量 (正常交易后): {len(alerts)}")
        
        # 模拟连续亏损
        print("\n2. 模拟连续亏损...")
        for i in range(4):
            tracker.record_trade_entry(f"loss_{i}", "BTC-USDT", "buy", 50000.0, 0.1)
            tracker.record_trade_exit(f"loss_{i}", 49000.0)  # 2% 亏损
        
        # 检查是否检测到连续亏损风险
        performance = monitor.get_performance_summary()
        print(f"连续亏损后表现: {performance}")
        
        alerts = monitor.check_risk_conditions(current_balance=10000.0)
        print(f"风险告警数量 (连续亏损后): {len(alerts)}")
        for alert in alerts:
            print(f"  - {alert['type']}: {alert['message']}")
        
        # 模拟大额亏损
        print("\n3. 模拟单笔大额亏损...")
        tracker.record_trade_entry("big_loss", "BTC-USDT", "buy", 50000.0, 0.5)
        tracker.record_trade_exit("big_loss", 40000.0)  # 20% 亏损
        
        alerts = monitor.check_risk_conditions(current_balance=10000.0)
        print(f"风险告警数量 (大额亏损后): {len(alerts)}")
        for alert in alerts:
            print(f"  - {alert['type']}: {alert['message']}")
            
        print("\n风险监控功能测试完成！")
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_db.name):
            os.unlink(temp_db.name)


if __name__ == "__main__":
    test_risk_monitoring()
