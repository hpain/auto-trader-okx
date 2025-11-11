"""
增强版监控系统，整合了性能监控、风险监控和系统资源监控
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd

from trader.risk_monitor import RiskMonitor
from utils.notification_manager import get_notification_manager, AlertSeverity, Alert
from utils.config_loader import load_config


class EnhancedMonitor:
    """
    增强版监控系统，提供全面的监控和告警功能
    """
    
    def __init__(self, db_path: str = "trader_state.db", config_path: str = "config/settings.yaml"):
        self.logger = logging.getLogger(__name__)
        
        # 加载配置
        try:
            self.config = load_config(config_path)
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found, using empty config")
            self.config = {}
        
        # 初始化原有风险监控器
        self.risk_monitor = RiskMonitor(db_path, self.config.get('risk_monitoring', {}))
        
        # 初始化通知管理器
        self.notification_manager = get_notification_manager(self.config)
        
        # 性能指标阈值
        self.performance_thresholds = self.config.get('performance_thresholds', {
            'min_sharpe_ratio': 0.5,
            'min_win_rate': 0.4,
            'max_drawdown': 0.15,
            'max_daily_loss': -500.0,
            'min_avg_return': 0.001
        })
        
        # 系统资源阈值
        self.system_thresholds = self.config.get('system_thresholds', {
            'max_cpu_percent': 80,
            'max_memory_percent': 80,
            'max_disk_percent': 85
        })
        
        # 用于跟踪模型性能的指标
        self.model_performance = {}
        
    def check_performance_metrics(self) -> List[Dict]:
        """
        检查性能指标
        """
        performance_alerts = []
        
        # 从风险监控器获取近期表现
        recent_performance = self.risk_monitor.trade_tracker.get_recent_performance(days=7)
        
        if recent_performance['total_trades'] >= 10:  # 确保有足够的样本量
            # 检查夏普比率
            sharpe_ratio = recent_performance.get('sharpe_ratio', 0)
            min_sharpe = self.performance_thresholds.get('min_sharpe_ratio', 0.5)
            if sharpe_ratio < min_sharpe:
                performance_alerts.append({
                    'type': 'LOW_SHARPE_RATIO',
                    'message': f'夏普比率过低: {sharpe_ratio:.3f} < {min_sharpe:.3f}',
                    'severity': 'MEDIUM',
                    'current_value': sharpe_ratio,
                    'threshold': min_sharpe
                })
            
            # 检查胜率
            win_rate = recent_performance.get('win_rate', 0)
            min_win_rate = self.performance_thresholds.get('min_win_rate', 0.4)
            if win_rate < min_win_rate:
                performance_alerts.append({
                    'type': 'LOW_WIN_RATE',
                    'message': f'胜率过低: {win_rate:.2%} < {min_win_rate:.2%}',
                    'severity': 'MEDIUM',
                    'current_value': win_rate,
                    'threshold': min_win_rate
                })
            
            # 检查平均收益率
            avg_return = recent_performance.get('avg_return', 0)
            min_avg_return = self.performance_thresholds.get('min_avg_return', 0.001)
            if avg_return < min_avg_return:
                performance_alerts.append({
                    'type': 'LOW_AVG_RETURN',
                    'message': f'平均收益率过低: {avg_return:.4f} < {min_avg_return:.4f}',
                    'severity': 'MEDIUM',
                    'current_value': avg_return,
                    'threshold': min_avg_return
                })
        
        # 检查最大回撤
        max_drawdown = self.risk_monitor.trade_tracker.get_max_drawdown()
        max_allowed_drawdown = self.performance_thresholds.get('max_drawdown', 0.15)
        if abs(max_drawdown) > max_allowed_drawdown:
            performance_alerts.append({
                'type': 'HIGH_MAX_DRAWDOWN',
                'message': f'最大回撤过高: {max_drawdown:.3f} > {max_allowed_drawdown:.3f}',
                'severity': 'HIGH',
                'current_value': max_drawdown,
                'threshold': max_allowed_drawdown
            })
        
        # 检查每日亏损
        daily_pnl = self.risk_monitor.trade_tracker.get_daily_pnl()
        max_daily_loss = self.performance_thresholds.get('max_daily_loss', -500.0)
        if daily_pnl < max_daily_loss:
            performance_alerts.append({
                'type': 'HIGH_DAILY_LOSS',
                'message': f'当日亏损过多: {daily_pnl:.2f} < {max_daily_loss:.2f}',
                'severity': 'HIGH',
                'current_value': daily_pnl,
                'threshold': max_daily_loss
            })
        
        return performance_alerts
    
    def check_system_resources(self) -> List[Dict]:
        """
        检查系统资源使用情况
        """
        system_alerts = []
        
        # 获取系统状态
        system_status = self.notification_manager.get_system_status()
        
        # 检查CPU使用率
        cpu_percent = system_status.get('cpu_percent', 0)
        max_cpu = self.system_thresholds.get('max_cpu_percent', 80)
        if cpu_percent > max_cpu:
            system_alerts.append({
                'type': 'HIGH_CPU_USAGE',
                'message': f'CPU使用率过高: {cpu_percent}% > {max_cpu}%',
                'severity': 'MEDIUM',
                'current_value': cpu_percent,
                'threshold': max_cpu
            })
        
        # 检查内存使用率
        memory_percent = system_status.get('memory_percent', 0)
        max_memory = self.system_thresholds.get('max_memory_percent', 80)
        if memory_percent > max_memory:
            system_alerts.append({
                'type': 'HIGH_MEMORY_USAGE',
                'message': f'内存使用率过高: {memory_percent}% > {max_memory}%',
                'severity': 'MEDIUM',
                'current_value': memory_percent,
                'threshold': max_memory
            })
        
        # 检查磁盘使用率
        disk_percent = system_status.get('disk_percent', 0)
        max_disk = self.system_thresholds.get('max_disk_percent', 85)
        if disk_percent > max_disk:
            system_alerts.append({
                'type': 'HIGH_DISK_USAGE',
                'message': f'磁盘使用率过高: {disk_percent}% > {max_disk}%',
                'severity': 'MEDIUM',
                'current_value': disk_percent,
                'threshold': max_disk
            })
        
        return system_alerts
    
    def check_model_performance(self, model_metrics: Optional[Dict] = None) -> List[Dict]:
        """
        检查模型性能
        """
        model_alerts = []
        
        if model_metrics:
            # 检查模型准确率（如果提供）
            if 'accuracy' in model_metrics:
                accuracy = model_metrics['accuracy']
                if accuracy < 0.55:  # 假设最低可接受准确率为55%
                    model_alerts.append({
                        'type': 'LOW_MODEL_ACCURACY',
                        'message': f'模型准确率过低: {accuracy:.3f} < 0.55',
                        'severity': 'HIGH',
                        'current_value': accuracy,
                        'threshold': 0.55
                    })
            
            # 检查模型夏普比率（如果提供）
            if 'sharpe_ratio' in model_metrics:
                model_sharpe = model_metrics['sharpe_ratio']
                min_model_sharpe = self.performance_thresholds.get('min_sharpe_ratio', 0.5)
                if model_sharpe < min_model_sharpe:
                    model_alerts.append({
                        'type': 'LOW_MODEL_SHARPE',
                        'message': f'模型夏普比率过低: {model_sharpe:.3f} < {min_model_sharpe:.3f}',
                        'severity': 'HIGH',
                        'current_value': model_sharpe,
                        'threshold': min_model_sharpe
                    })
        
        return model_alerts
    
    def monitor_all(self, current_balance: Optional[float] = None, 
                   model_metrics: Optional[Dict] = None) -> Dict[str, List[Dict]]:
        """
        执行所有类型的监控
        """
        all_alerts = {}
        
        # 执行原有的风险监控
        all_alerts['risk'] = self.risk_monitor.check_risk_conditions(current_balance)
        
        # 执行性能监控
        all_alerts['performance'] = self.check_performance_metrics()
        
        # 执行系统资源监控
        all_alerts['system'] = self.check_system_resources()
        
        # 执行模型性能监控
        all_alerts['model'] = self.check_model_performance(model_metrics)
        
        return all_alerts
    
    def send_alerts(self, alerts_by_category: Dict[str, List[Dict]]) -> List[bool]:
        """
        发送所有类型的告警
        """
        sent_results = []
        
        for category, alerts in alerts_by_category.items():
            for alert in alerts:
                if alert['severity'] == 'CRITICAL':
                    severity_enum = AlertSeverity.CRITICAL
                elif alert['severity'] == 'HIGH':
                    severity_enum = AlertSeverity.HIGH
                elif alert['severity'] == 'MEDIUM':
                    severity_enum = AlertSeverity.MEDIUM
                else:
                    severity_enum = AlertSeverity.LOW
                
                # 根据告警类型发送到相应渠道
                if category == 'risk':
                    sent_results.append(self.notification_manager.send_risk_alert(
                        alert['type'], alert['message']
                    ))
                elif category == 'performance':
                    sent_results.append(self.notification_manager.send_performance_alert(
                        alert['type'], 
                        alert.get('current_value', 0), 
                        alert.get('threshold', 0)
                    ))
                elif category == 'system':
                    alert_obj = Alert(
                        title=f"System Alert: {alert['type']}",
                        message=alert['message'],
                        severity=severity_enum,
                        category=category,
                        additional_data=alert
                    )
                    sent_results.append(self.notification_manager.send_alert(alert_obj))
                elif category == 'model':
                    alert_obj = Alert(
                        title=f"Model Alert: {alert['type']}",
                        message=alert['message'],
                        severity=severity_enum,
                        category=category,
                        additional_data=alert
                    )
                    sent_results.append(self.notification_manager.send_alert(alert_obj))
        
        return sent_results
    
    def generate_status_report(self) -> str:
        """
        生成状态报告
        """
        report = []
        report.append("=== Trading System Status Report ===")
        report.append(f"Report Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 系统资源信息
        system_status = self.notification_manager.get_system_status()
        report.append("System Resources:")
        report.append(f"  CPU Usage: {system_status.get('cpu_percent', 'N/A')}%")
        report.append(f"  Memory Usage: {system_status.get('memory_percent', 'N/A')}%")
        report.append(f"  Disk Usage: {system_status.get('disk_percent', 'N/A')}%")
        report.append(f"  Uptime: {system_status.get('uptime', 'N/A')} hours")
        report.append(f"  Active Processes: {system_status.get('process_count', 'N/A')}")
        report.append("")
        
        # 交易表现信息
        recent_performance = self.risk_monitor.trade_tracker.get_recent_performance(days=7)
        report.append("Trading Performance (Last 7 Days):")
        report.append(f"  Total Trades: {recent_performance.get('total_trades', 'N/A')}")
        report.append(f"  Win Rate: {recent_performance.get('win_rate', 'N/A'): .2%}")
        report.append(f"  Avg Return: {recent_performance.get('avg_return', 'N/A'): .4f}")
        report.append(f"  Sharpe Ratio: {recent_performance.get('sharpe_ratio', 'N/A'): .3f}")
        report.append(f"  Total PnL: {recent_performance.get('total_pnl', 'N/A'): .2f}")
        report.append(f"  Max Drawdown: {self.risk_monitor.trade_tracker.get_max_drawdown(): .3f}")
        report.append("")
        
        # 风险指标
        daily_pnl = self.risk_monitor.trade_tracker.get_daily_pnl()
        consecutive_losses = self.risk_monitor.trade_tracker.get_consecutive_losses_count()
        report.append("Risk Indicators:")
        report.append(f"  Daily PnL: {daily_pnl: .2f}")
        report.append(f"  Consecutive Losses: {consecutive_losses}")
        
        return "\n".join(report)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        获取仪表板数据
        """
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


# 全局实例
_enhanced_monitor = None


def get_enhanced_monitor(db_path: str = "trader_state.db") -> EnhancedMonitor:
    """
    获取或创建增强监控器实例
    """
    global _enhanced_monitor
    if _enhanced_monitor is None:
        _enhanced_monitor = EnhancedMonitor(db_path)
    return _enhanced_monitor