"""
实时监控和告警模块，用于监控交易性能和风险指标
"""
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from trader.trade_tracker import TradeTracker


class RiskMonitor:
    """
    风险监控器，用于实时监控交易风险并发出告警
    """
    
    def __init__(self, db_path: str = "trader_state.db", config: Optional[Dict] = None):
        self.trade_tracker = TradeTracker(db_path)
        self.config = config or {}
        self.last_alert_time = {}
        
        # 默认风险阈值
        self.daily_loss_threshold = self.config.get('daily_loss_threshold', -500.0)  # 每日最大亏损
        self.max_drawdown_threshold = self.config.get('max_drawdown_threshold', 0.10)  # 最大回撤10%
        self.max_consecutive_losses = self.config.get('max_consecutive_losses', 5)  # 最大连续亏损次数
        self.max_single_loss_pct = self.config.get('max_single_loss_pct', -0.03)  # 单笔最大亏损比例
        self.min_account_balance = self.config.get('min_account_balance', 1000.0)  # 最小账户余额
        
        # 邮件配置
        self.email_config = self.config.get('email', {})
        self.smtp_server = self.email_config.get('smtp_server', 'localhost')
        self.smtp_port = self.email_config.get('smtp_port', 587)
        self.email_user = self.email_config.get('user', '')
        self.email_password = self.email_config.get('password', '')
        self.alert_recipients = self.email_config.get('recipients', [])
        
    def check_risk_conditions(self, current_balance: Optional[float] = None) -> List[Dict]:
        """
        检查各种风险条件
        """
        risk_alerts = []
        
        # 检查每日亏损
        daily_pnl = self.trade_tracker.get_daily_pnl()
        if daily_pnl < self.daily_loss_threshold:
            risk_alerts.append({
                'type': 'DAILY_LOSS_EXCEEDED',
                'message': f'今日亏损超出阈值: {daily_pnl:.2f} < {self.daily_loss_threshold}',
                'severity': 'HIGH'
            })
        
        # 检查连续亏损
        consecutive_losses = self.trade_tracker.get_consecutive_losses_count()
        if consecutive_losses >= self.max_consecutive_losses:
            risk_alerts.append({
                'type': 'CONSECUTIVE_LOSSES',
                'message': f'连续亏损次数过多: {consecutive_losses} >= {self.max_consecutive_losses}',
                'severity': 'MEDIUM'
            })
        
        # 如果提供了当前余额，检查最小余额
        if current_balance is not None and current_balance < self.min_account_balance:
            risk_alerts.append({
                'type': 'LOW_ACCOUNT_BALANCE',
                'message': f'账户余额过低: {current_balance:.2f} < {self.min_account_balance:.2f}',
                'severity': 'HIGH'
            })
        
        # 检查近期表现
        recent_performance = self.trade_tracker.get_recent_performance(days=7)
        if recent_performance['total_trades'] >= 10:  # 只有在有足够的交易样本时才检查
            win_rate = recent_performance['win_rate']
            if win_rate < 0.35:  # 胜率低于35%
                risk_alerts.append({
                    'type': 'LOW_WIN_RATE',
                    'message': f'近期胜率过低: {win_rate:.2%} < 35%',
                    'severity': 'MEDIUM'
                })
        
        return risk_alerts
    
    def send_alert(self, alert_type: str, message: str, severity: str = 'MEDIUM'):
        """
        发送告警
        """
        current_time = datetime.now()
        
        # 防止重复告警（基于告警类型）
        if alert_type in self.last_alert_time:
            time_since_last = current_time - self.last_alert_time[alert_type]
            if time_since_last < timedelta(minutes=30):  # 30分钟内不重复告警同一类型
                return
        
        self.last_alert_time[alert_type] = current_time
        
        # 记录到日志
        log_func = logging.warning if severity in ['HIGH', 'CRITICAL'] else logging.info
        log_func(f'RISK ALERT [{severity}]: {message}')
        
        # 发送邮件告警（如果有配置）
        if self.alert_recipients and self.email_user:
            try:
                self._send_email_alert(alert_type, message, severity)
            except Exception as e:
                logging.error(f"Failed to send email alert: {e}")
    
    def _send_email_alert(self, alert_type: str, message: str, severity: str):
        """
        发送邮件告警
        """
        if not self.alert_recipients:
            return
            
        msg = MIMEMultipart()
        msg['From'] = self.email_user
        msg['To'] = ', '.join(self.alert_recipients)
        msg['Subject'] = f"[RISK ALERT] {severity}: {alert_type}"
        
        body = f"""
        Risk Alert Notification
        
        Type: {alert_type}
        Severity: {severity}
        Message: {message}
        Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Please check your trading system immediately.
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.starttls()
            server.login(self.email_user, self.email_password)
            server.send_message(msg)
    
    def monitor_and_alert(self, current_balance: Optional[float] = None):
        """
        执行监控并发送告警
        """
        risk_alerts = self.check_risk_conditions(current_balance)
        
        for alert in risk_alerts:
            self.send_alert(
                alert_type=alert['type'],
                message=alert['message'],
                severity=alert['severity']
            )
        
        return risk_alerts

    def get_performance_summary(self) -> Dict:
        """
        获取性能摘要
        """
        daily_pnl = self.trade_tracker.get_daily_pnl()
        recent_performance = self.trade_tracker.get_recent_performance(days=7)
        consecutive_losses = self.trade_tracker.get_consecutive_losses_count()
        
        return {
            'daily_pnl': daily_pnl,
            'recent_performance': recent_performance,
            'consecutive_losses': consecutive_losses,
            'unresolved_risk_events': len(self.trade_tracker.get_unresolved_risk_events())
        }