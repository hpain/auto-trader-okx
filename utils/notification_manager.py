"""
增强的通知和告警管理模块
"""
import smtplib
import ssl
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import List, Dict, Optional, Any
import json
import requests
from dataclasses import dataclass
from enum import Enum
import time


class AlertSeverity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    title: str
    message: str
    severity: AlertSeverity
    category: str = "general"
    timestamp: datetime = None
    additional_data: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class NotificationManager:
    """
    增强的通知管理器，支持多种通知方式
    """
    
    def __init__(self, config: Dict):
        self.config = config.get('risk_monitoring', {}).get('email', {})
        self.smtp_server = self.config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = self.config.get('smtp_port', 587)
        self.email_user = self.config.get('user', '')
        self.email_password = self.config.get('password', '')
        self.recipients = self.config.get('recipients', [])
        self.enabled = bool(self.email_user and self.email_password and self.recipients)
        
        self.logger = logging.getLogger(__name__)
        
        if not self.enabled:
            self.logger.warning("Email notifications are disabled - missing SMTP configuration")
        
        # 记录上次告警时间，避免重复告警
        self.last_alert_time = {}
    
    def _should_send_alert(self, alert_type: str, min_interval_minutes: int = 30) -> bool:
        """
        检查是否应该发送告警（基于最小间隔时间）
        """
        current_time = datetime.utcnow()
        if alert_type in self.last_alert_time:
            time_since_last = (current_time - self.last_alert_time[alert_type]).total_seconds() / 60
            if time_since_last < min_interval_minutes:
                return False
        
        self.last_alert_time[alert_type] = current_time
        return True
    
    def send_email(self, subject: str, body: str, recipients: Optional[List[str]] = None) -> bool:
        """
        发送邮件通知
        """
        if not self.enabled:
            return False
            
        if recipients is None:
            recipients = self.recipients
            
        try:
            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            
            # 添加邮件正文
            msg.attach(MIMEText(body, 'html'))  # 使用HTML格式提高可读性
            
            # 创建SSL上下文
            context = ssl.create_default_context()
            
            # 发送邮件
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.email_user, self.email_password)
                server.sendmail(self.email_user, recipients, msg.as_string())
            
            self.logger.info(f"Email alert sent successfully: {subject}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
            return False
    
    def send_slack_notification(self, message: str, webhook_url: Optional[str] = None) -> bool:
        """
        发送Slack通知
        """
        if not webhook_url:
            webhook_url = self.config.get('slack_webhook_url')
        
        if not webhook_url:
            self.logger.warning("Slack notifications are disabled - missing webhook URL")
            return False
        
        try:
            payload = {
                "text": message,
                "username": "Trading Bot Alert",
                "icon_emoji": ":warning:"
            }
            
            response = requests.post(webhook_url, json=payload)
            if response.status_code == 200:
                self.logger.info("Slack notification sent successfully")
                return True
            else:
                self.logger.error(f"Failed to send Slack notification: {response.status_code}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to send Slack notification: {e}")
            return False
    
    def send_alert(self, alert: Alert, min_interval_minutes: int = 30) -> bool:
        """
        发送格式化的告警
        """
        alert_type = f"{alert.category}_{alert.severity.value}"
        
        if not self._should_send_alert(alert_type, min_interval_minutes):
            self.logger.debug(f"Skipping duplicate alert: {alert.title}")
            return False
        
        # HTML格式的邮件内容
        html_body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; }}
                .severity {{ color: {'red' if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL] else 'orange' if alert.severity == AlertSeverity.MEDIUM else 'green'}; font-weight: bold; }}
                .category {{ color: #666; font-style: italic; }}
                .content {{ margin: 15px 0; }}
                .additional-data {{ background-color: #f9f9f9; padding: 10px; border-left: 4px solid #ccc; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>Trading System Alert</h2>
            </div>
            
            <div class="category">Category: {alert.category}</div>
            <div class="severity">Severity: {alert.severity.value}</div>
            <div>Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</div>
            
            <h3>{alert.title}</h3>
            
            <div class="content">
                <p>{alert.message}</p>
            </div>
            
            {f'<div class="additional-data"><h4>Additional Data:</h4><pre>{json.dumps(alert.additional_data, indent=2)}</pre></div>' if alert.additional_data else ''}
            
            <hr>
            <p><em>This is an automated alert from the trading system.</em></p>
        </body>
        </html>
        """
        
        # 发送邮件
        email_sent = self.send_email(
            subject=f"[{alert.severity.value}] {alert.title}",
            body=html_body
        )
        
        # 发送Slack通知（如果配置了）
        slack_sent = False
        if self.config.get('slack_webhook_url'):
            slack_message = f"*[{alert.severity.value}] {alert.title}*\n{alert.message}"
            slack_sent = self.send_slack_notification(slack_message)
        
        return email_sent or slack_sent
    
    def send_performance_alert(self, metric_name: str, current_value: float, threshold: float, 
                             symbol: str = "BTC-USDT", min_interval_minutes: int = 60) -> bool:
        """
        发送性能相关告警
        """
        title = f"Performance Alert: {metric_name}"
        message = f"""
Performance Alert for {symbol}:

- Metric: {metric_name}
- Current Value: {current_value:.4f}
- Threshold: {threshold:.4f}
- Status: {'BELOW' if current_value < threshold else 'ABOVE'} threshold

This indicates potential issues with trading performance.
        """.strip()
        
        severity = AlertSeverity.HIGH if abs(current_value) < threshold * 0.5 else AlertSeverity.MEDIUM
        alert = Alert(
            title=title, 
            message=message, 
            severity=severity, 
            category="performance"
        )
        
        return self.send_alert(alert, min_interval_minutes)
    
    def send_risk_alert(self, risk_type: str, details: str, min_interval_minutes: int = 30) -> bool:
        """
        发送风险相关告警
        """
        title = f"Risk Alert: {risk_type}"
        message = f"""
Risk Alert:

- Type: {risk_type}
- Details: {details}

Immediate attention required.
        """.strip()
        
        severity = AlertSeverity.CRITICAL
        alert = Alert(
            title=title, 
            message=message, 
            severity=severity, 
            category="risk"
        )
        
        return self.send_alert(alert, min_interval_minutes)
    
    def send_system_status_alert(self, status_data: Dict, min_interval_minutes: int = 120) -> bool:
        """
        发送系统状态告警
        """
        title = "System Status Report"
        message = f"""
System Status Report:

- CPU Usage: {status_data.get('cpu_percent', 'N/A')}%
- Memory Usage: {status_data.get('memory_percent', 'N/A')}%
- Disk Usage: {status_data.get('disk_percent', 'N/A')}%
- Uptime: {status_data.get('uptime', 'N/A')} hours
- Active Processes: {status_data.get('process_count', 'N/A')}

Monitor system health to ensure optimal trading performance.
        """.strip()
        
        # 根据系统资源使用情况确定严重程度
        cpu_usage = status_data.get('cpu_percent', 0)
        memory_usage = status_data.get('memory_percent', 0)
        
        if cpu_usage > 90 or memory_usage > 90:
            severity = AlertSeverity.HIGH
        elif cpu_usage > 80 or memory_usage > 80:
            severity = AlertSeverity.MEDIUM
        else:
            severity = AlertSeverity.LOW
            
        alert = Alert(
            title=title, 
            message=message, 
            severity=severity, 
            category="system"
        )
        
        return self.send_alert(alert, min_interval_minutes)
    
    def send_error_alert(self, error_context: str, error_details: str, min_interval_minutes: int = 15) -> bool:
        """
        发送错误告警
        """
        title = f"System Error: {error_context}"
        message = f"""
System Error Alert:

- Context: {error_context}
- Details: {error_details}

Please investigate immediately.
        """.strip()
        
        alert = Alert(
            title=title, 
            message=message, 
            severity=AlertSeverity.CRITICAL, 
            category="error"
        )
        
        return self.send_alert(alert, min_interval_minutes)
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        获取系统状态信息
        """
        # 计算系统正常运行时间（小时）
        boot_time = psutil.boot_time()
        uptime_seconds = time.time() - boot_time
        uptime_hours = round(uptime_seconds / 3600, 2)
        
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent if hasattr(psutil, 'disk_usage') else psutil.disk_usage('.').percent,
            'uptime': uptime_hours,
            'process_count': len(psutil.pids()),
            'timestamp': datetime.utcnow().isoformat()
        }


# 全局实例
_notification_manager = None


def get_notification_manager(config: Dict) -> NotificationManager:
    """
    获取或创建通知管理器实例
    """
    global _notification_manager
    if _notification_manager is None:
        _notification_manager = NotificationManager(config)
    return _notification_manager