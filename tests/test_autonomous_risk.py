
"""
测试自主交易系统的新风险监控功能
"""
import os
import sys
import tempfile
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from trader.trade_tracker import TradeTracker
from trader.risk_monitor import RiskMonitor


def test_autonomous_trading_risk_features():
    """测试自主交易系统的新风险功能"""
    
    # 创建临时数据库用于测试
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db.close()
    
    try:
        print("开始测试自主交易系统的风险功能...")
        
        # 1. 测试交易跟踪器
        print("\n1. 测试TradeTracker...")
        tracker = TradeTracker(db_path=temp_db.name)
        
        # 记录一些交易
        tracker.record_trade_entry("test_buy_1", "BTC-USDT", "buy", 50000.0, 0.1)
        tracker.record_trade_exit("test_buy_1", 50100.0)  # 盈利交易
        
        tracker.record_trade_entry("test_buy_2", "BTC-USDT", "buy", 50100.0, 0.1)
        tracker.record_trade_exit("test_buy_2", 50050.0)  # 小幅亏损
        
        # 检查性能数据
        perf = tracker.get_recent_performance()
        print(f"   最近表现: {perf}")
        
        daily_pnl = tracker.get_daily_pnl()
        print(f"   今日盈亏: {daily_pnl}")
        
        tracker.conn.close()
        print("   OK: TradeTracker功能正常")
        
        # 2. 测试风险监控器
        print("\n2. 测试RiskMonitor...")
        monitor = RiskMonitor(
            db_path=temp_db.name,
            config={
                'daily_loss_threshold': -100.0,
                'max_consecutive_losses': 3,
                'max_single_loss_pct': -0.02,
                'min_account_balance': 500.0
            }
        )
        
        # 添加几笔亏损交易来测试风险检测
        tracker2 = TradeTracker(db_path=temp_db.name)
        for i in range(4):  # 4笔连续亏损，超过阈值3
            trade_id = f"loss_{i}"
            tracker2.record_trade_entry(trade_id, "BTC-USDT", "buy", 50000.0, 0.1)
            tracker2.record_trade_exit(trade_id, 49500.0)  # 1% 亏损
        tracker2.conn.close()
        
        # 检查风险监控
        alerts = monitor.check_risk_conditions(current_balance=8000.0)
        print(f"   风险告警数量: {len(alerts)}")
        for alert in alerts:
            print(f"     - {alert['type']}: {alert['message']}")
        
        # 性能摘要
        summary = monitor.get_performance_summary()
        print(f"   性能摘要: {summary}")
        
        print("   OK: RiskMonitor功能正常")
        
        print("\n自主交易系统风险功能测试完成！")
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理临时文件
        if os.path.exists(temp_db.name):
            try:
                os.unlink(temp_db.name)
            except:
                # 在Windows上，如果文件被锁定，可以跳过
                pass


if __name__ == "__main__":
    success = test_autonomous_trading_risk_features()
    if success:
        print("\nOK: 所有测试通过！")
    else:
        print("\nERROR: 测试失败！")
        sys.exit(1)
