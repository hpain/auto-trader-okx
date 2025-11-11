"""
测试系统监控模块
"""
import unittest
from unittest.mock import patch, MagicMock
from utils.system_monitor import SystemMonitor, get_system_monitor

# 尝试导入psutil，如果失败则在测试中处理
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class TestSystemMonitor(unittest.TestCase):
    
    def setUp(self):
        """设置测试环境"""
        self.config = {
            'cpu_threshold': 80,
            'memory_threshold': 80,
            'disk_threshold': 85,
            'monitor_interval': 1
        }
        self.system_monitor = SystemMonitor(self.config)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.system_monitor.cpu_threshold, 80)
        self.assertEqual(self.system_monitor.memory_threshold, 80)
        self.assertEqual(self.system_monitor.disk_threshold, 85)
        self.assertEqual(self.system_monitor.config['monitor_interval'], 1)
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.swap_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    @patch('psutil.pids')
    @patch('psutil.boot_time')
    @patch('psutil.cpu_freq')
    @unittest.skipIf(not PSUTIL_AVAILABLE, "psutil not available")
    def test_get_system_status(self, mock_cpu_freq, mock_boot_time, mock_pids, 
                               mock_net_io, mock_disk_usage, mock_swap, 
                               mock_memory, mock_cpu_percent):
        """测试获取系统状态"""
        # 模拟系统信息
        mock_cpu_percent.return_value = 50
        mock_memory.return_value = MagicMock(total=10000000000, available=2000000000, 
                                             percent=80, used=8000000000, free=2000000000)
        mock_swap.return_value = MagicMock(total=2000000000, used=100000000, percent=5)
        mock_disk_usage.return_value = MagicMock(total=500000000000, used=200000000000, 
                                                 free=300000000000, percent=40)
        mock_net_io.return_value = MagicMock(bytes_sent=1000, bytes_recv=2000, 
                                             packets_sent=10, packets_recv=20)
        mock_pids.return_value = [1, 2, 3, 4, 5]
        mock_boot_time.return_value = 1600000000.0
        mock_cpu_freq.return_value = MagicMock(current=2500, min=800, max=4000)
        
        status = self.system_monitor.get_system_status()
        
        # 验证返回的数据结构
        self.assertIn('timestamp', status)
        self.assertIn('cpu', status)
        self.assertIn('memory', status)
        self.assertIn('disk', status)
        self.assertIn('network', status)
        self.assertIn('processes', status)
        self.assertIn('system', status)
        
        # 验证CPU信息
        self.assertEqual(status['cpu']['percent'], 50)
        
        # 验证内存信息
        self.assertEqual(status['memory']['percent'], 80)
        
        # 验证磁盘信息
        self.assertEqual(status['disk']['percent'], 40)
        
        # 验证进程信息
        self.assertEqual(status['processes']['count'], 5)
    
    def test_check_for_alerts_no_alerts(self):
        """测试无警报情况"""
        status = {
            'cpu': {'percent': 50},
            'memory': {'percent': 60},
            'disk': {'percent': 50},
            'processes': {'count': 100}
        }
        
        alerts = self.system_monitor._check_for_alerts(status)
        
        self.assertEqual(len(alerts), 0)
    
    def test_check_for_alerts_with_alerts(self):
        """测试有警报情况"""
        status = {
            'cpu': {'percent': 90},  # 超过阈值80
            'memory': {'percent': 90},  # 超过阈值80
            'disk': {'percent': 95},  # 超过阈值85
            'processes': {'count': 2000}  # 超过阈值1000（默认）
        }
        
        alerts = self.system_monitor._check_for_alerts(status)
        
        self.assertGreater(len(alerts), 0)  # 应该有警报
        # 检查是否包含预期的警报信息
        alert_str = ' '.join(alerts)
        self.assertIn('CPU', alert_str)
        self.assertIn('memory', alert_str)
        self.assertIn('disk', alert_str)
    
    def test_callback_management(self):
        """测试回调管理"""
        callback = lambda x: None
        
        # 添加回调
        self.system_monitor.add_alert_callback(callback)
        self.assertIn(callback, self.system_monitor.alert_callbacks)
        
        # 移除回调
        self.system_monitor.remove_alert_callback(callback)
        self.assertNotIn(callback, self.system_monitor.alert_callbacks)


if __name__ == '__main__':
    unittest.main()