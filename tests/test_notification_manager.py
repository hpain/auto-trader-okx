"""
测试通知管理模块
"""
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from utils.notification_manager import NotificationManager, Alert, AlertSeverity, get_notification_manager


class TestNotificationManager(unittest.TestCase):
    
    def setUp(self):
        """设置测试环境"""
        self.config = {
            'risk_monitoring': {
                'email': {
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'user': 'test@example.com',
                    'password': 'test_password',
                    'recipients': ['admin@example.com']
                }
            }
        }
        self.notification_manager = NotificationManager(self.config)
    
    def test_initialization(self):
        """测试初始化"""
        # 测试配置正确加载
        self.assertEqual(self.notification_manager.smtp_server, 'smtp.gmail.com')
        self.assertEqual(self.notification_manager.smtp_port, 587)
        self.assertEqual(self.notification_manager.email_user, 'test@example.com')
        self.assertEqual(self.notification_manager.recipients, ['admin@example.com'])
        self.assertTrue(self.notification_manager.enabled)  # 配置完整，应启用
    
    @patch('smtplib.SMTP')
    def test_send_email(self, mock_smtp):
        """测试发送邮件"""
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        result = self.notification_manager.send_email(
            subject="Test Subject",
            body="Test Body",
            recipients=["test@example.com"]
        )
        
        self.assertTrue(result)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with('test@example.com', 'test_password')
        mock_server.sendmail.assert_called_once()
    
    def test_alert_creation(self):
        """测试告警创建"""
        alert = Alert(
            title="Test Alert",
            message="Test message",
            severity=AlertSeverity.HIGH
        )
        
        self.assertEqual(alert.title, "Test Alert")
        self.assertEqual(alert.message, "Test message")
        self.assertEqual(alert.severity, AlertSeverity.HIGH)
        self.assertIsNotNone(alert.timestamp)
    
    @patch('utils.notification_manager.NotificationManager.send_email')
    def test_send_alert(self, mock_send_email):
        """测试发送告警"""
        mock_send_email.return_value = True
        
        alert = Alert(
            title="Test Alert",
            message="Test message",
            severity=AlertSeverity.HIGH
        )
        
        result = self.notification_manager.send_alert(alert)
        
        self.assertTrue(result)
        mock_send_email.assert_called_once()


if __name__ == '__main__':
    unittest.main()