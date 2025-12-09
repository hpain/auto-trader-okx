import os
from datetime import datetime

def fix_enhanced_monitor():
    file_path = 'trader/enhanced_monitor.py'
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Keep lines 1-310 (indices 0-310)
    # Line 310 is "        report.append("Risk Indicators:")\n"
    kept_lines = lines[:310]
    
    new_content = """        report.append(f"  Daily PnL: {daily_pnl: .2f}")
        report.append(f"  Consecutive Losses: {consecutive_losses}")
        
        return "\\n".join(report)

    def get_dashboard_data(self) -> Dict[str, Any]:
        \"\"\"
        获取仪表板数据
        \"\"\"
        system_status = self.notification_manager.get_system_status()
        recent_performance = self.risk_monitor.trade_tracker.get_recent_performance(days=7)
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'system_status': system_status,
            'trading_performance': {
                'total_trades': recent_performance.get('total_trades', 0),
                'win_rate': recent_performance.get('win_rate', 0),
                'avg_return': recent_performance.get('avg_return', 0),
                'sharpe_ratio': recent_performance.get('sharpe_ratio', 0),
                'total_pnl': recent_performance.get('total_pnl', 0),
                'max_drawdown': self.risk_monitor.trade_tracker.get_max_drawdown()
            },
            'risk_indicators': {
                'daily_pnl': self.risk_monitor.trade_tracker.get_daily_pnl(),
                'consecutive_losses': self.risk_monitor.trade_tracker.get_consecutive_losses_count(),
                'unresolved_risk_events': len(self.risk_monitor.trade_tracker.get_unresolved_risk_events())
            }
        }

    def get_daily_pnl(self) -> float:
        \"\"\"
        获取当日盈亏
        \"\"\"
        return self.risk_monitor.trade_tracker.get_daily_pnl()


def get_enhanced_monitor(db_path: str = "trader_state.db") -> EnhancedMonitor:
    \"\"\"
    获取或创建增强监控器实例
    \"\"\"
    global _enhanced_monitor
    if _enhanced_monitor is None:
        _enhanced_monitor = EnhancedMonitor(db_path)
    return _enhanced_monitor
"""

    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(kept_lines)
        f.write(new_content)
    
    print("Successfully fixed trader/enhanced_monitor.py")

if __name__ == "__main__":
    fix_enhanced_monitor()
