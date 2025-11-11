"""
系统资源监控模块
"""
import logging
import time
import os
from datetime import datetime
from typing import Dict, Optional, List
import json
import threading
import pandas as pd

# 尝试导入psutil，如果失败则在需要时抛出错误
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available, system monitoring will be limited")


class SystemMonitor:
    """
    系统资源监控器，监控CPU、内存、磁盘等系统资源使用情况
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 监控阈值
        self.cpu_threshold = self.config.get('cpu_threshold', 80)  # CPU使用率阈值
        self.memory_threshold = self.config.get('memory_threshold', 80)  # 内存使用率阈值
        self.disk_threshold = self.config.get('disk_threshold', 85)  # 磁盘使用率阈值
        self.process_threshold = self.config.get('process_threshold', 1000)  # 进程数量阈值
        
        # 监控历史记录
        self.monitor_history = []
        self.max_history_size = self.config.get('max_history_size', 1000)
        
        # 监控状态
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # 回调函数列表
        self.alert_callbacks = []
        
        self.logger.info("System Monitor initialized")
    
    def get_system_status(self) -> Dict:
        """
        获取当前系统状态
        
        Returns:
            包含系统状态信息的字典
        """
        if not PSUTIL_AVAILABLE:
            # 如果psutil不可用，返回基本的模拟数据
            timestamp = datetime.now()
            return {
                'timestamp': timestamp.isoformat(),
                'cpu': {
                    'percent': 0,
                    'count': 0,
                    'frequency': None
                },
                'memory': {
                    'total': 0,
                    'available': 0,
                    'percent': 0,
                    'used': 0,
                    'free': 0,
                    'swap_total': 0,
                    'swap_used': 0,
                    'swap_percent': 0
                },
                'disk': {
                    'total': 0,
                    'used': 0,
                    'free': 0,
                    'percent': 0
                },
                'network': {
                    'bytes_sent': 0,
                    'bytes_recv': 0,
                    'packets_sent': 0,
                    'packets_recv': 0
                },
                'processes': {
                    'count': 0
                },
                'system': {
                    'boot_time': timestamp.isoformat(),
                    'uptime_seconds': 0,
                    'uptime_formatted': "0:00:00"
                },
                'alerts': []
            }
        
        timestamp = datetime.now()
        
        # CPU信息
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        # 内存信息
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # 磁盘信息
        disk_usage = psutil.disk_usage('/')
        
        # 网络信息
        net_io = psutil.net_io_counters()
        
        # 进程信息
        process_count = len(psutil.pids())
        
        # 系统启动时间
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime = timestamp - boot_time
        
        system_status = {
            'timestamp': timestamp.isoformat(),
            'cpu': {
                'percent': cpu_percent,
                'count': cpu_count,
                'frequency': {
                    'current': cpu_freq.current if cpu_freq else None,
                    'min': cpu_freq.min if cpu_freq else None,
                    'max': cpu_freq.max if cpu_freq else None
                } if cpu_freq else None
            },
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'used': memory.used,
                'free': memory.free,
                'swap_total': swap.total,
                'swap_used': swap.used,
                'swap_percent': swap.percent
            },
            'disk': {
                'total': disk_usage.total,
                'used': disk_usage.used,
                'free': disk_usage.free,
                'percent': disk_usage.percent
            },
            'network': {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            },
            'processes': {
                'count': process_count
            },
            'system': {
                'boot_time': boot_time.isoformat(),
                'uptime_seconds': uptime.total_seconds(),
                'uptime_formatted': str(uptime)
            }
        }
        
        # 添加警报标志
        system_status['alerts'] = self._check_for_alerts(system_status)
        
        return system_status
    
    def _check_for_alerts(self, status: Dict) -> List[str]:
        """
        检查是否需要发出警报
        
        Args:
            status: 系统状态字典
            
        Returns:
            警报列表
        """
        alerts = []
        
        # 检查CPU使用率
        if status['cpu']['percent'] > self.cpu_threshold:
            alerts.append(f"High CPU usage: {status['cpu']['percent']:.1f}% > {self.cpu_threshold}%")
        
        # 检查内存使用率
        if status['memory']['percent'] > self.memory_threshold:
            alerts.append(f"High memory usage: {status['memory']['percent']:.1f}% > {self.memory_threshold}%")
        
        # 检查磁盘使用率
        if status['disk']['percent'] > self.disk_threshold:
            alerts.append(f"High disk usage: {status['disk']['percent']:.1f}% > {self.disk_threshold}%")
        
        # 检查进程数量
        if status['processes']['count'] > self.process_threshold:
            alerts.append(f"High process count: {status['processes']['count']} > {self.process_threshold}")
        
        return alerts
    
    def add_alert_callback(self, callback_func):
        """
        添加警报回调函数
        
        Args:
            callback_func: 回调函数
        """
        if callback_func not in self.alert_callbacks:
            self.alert_callbacks.append(callback_func)
    
    def remove_alert_callback(self, callback_func):
        """
        移除警报回调函数
        
        Args:
            callback_func: 回调函数
        """
        if callback_func in self.alert_callbacks:
            self.alert_callbacks.remove(callback_func)
    
    def _monitor_loop(self):
        """
        监控循环，运行在独立线程中
        """
        while self.is_monitoring:
            try:
                status = self.get_system_status()
                
                # 保存监控历史
                self.monitor_history.append(status)
                if len(self.monitor_history) > self.max_history_size:
                    self.monitor_history.pop(0)
                
                # 如果有警报，触发回调
                if status['alerts']:
                    self.logger.warning(f"System alerts: {', '.join(status['alerts'])}")
                    for callback in self.alert_callbacks:
                        try:
                            callback(status)
                        except Exception as e:
                            self.logger.error(f"Error in alert callback: {e}")
                
                # 等待下次监控
                time.sleep(self.config.get('monitor_interval', 30))  # 默认30秒间隔
                
            except Exception as e:
                self.logger.error(f"Error in system monitoring loop: {e}")
                time.sleep(5)  # 出错后等待5秒再继续
    
    def start_monitoring(self):
        """
        开始系统监控
        """
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitoring_thread.start()
            self.logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """
        停止系统监控
        """
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)  # 最多等待5秒
        self.logger.info("System monitoring stopped")
    
    def get_monitoring_history(self, minutes: int = 60) -> List[Dict]:
        """
        获取指定时间范围内的监控历史
        
        Args:
            minutes: 时间范围（分钟）
            
        Returns:
            监控历史记录列表
        """
        if not self.monitor_history:
            return []
        
        cutoff_time = datetime.now() - pd.Timedelta(minutes=minutes)
        history = [
            record for record in self.monitor_history
            if datetime.fromisoformat(record['timestamp']) > cutoff_time
        ]
        
        return history
    
    def get_resource_trends(self, minutes: int = 60) -> Dict:
        """
        获取资源使用趋势
        
        Args:
            minutes: 时间范围（分钟）
            
        Returns:
            资源使用趋势字典
        """
        history = self.get_monitoring_history(minutes)
        
        if not history:
            return {}
        
        # 提取资源使用数据
        timestamps = [datetime.fromisoformat(record['timestamp']) for record in history]
        cpu_usage = [record['cpu']['percent'] for record in history]
        memory_usage = [record['memory']['percent'] for record in history]
        disk_usage = [record['disk']['percent'] for record in history]
        
        trends = {
            'timestamps': timestamps,
            'cpu': {
                'usage': cpu_usage,
                'avg': sum(cpu_usage) / len(cpu_usage),
                'min': min(cpu_usage),
                'max': max(cpu_usage)
            },
            'memory': {
                'usage': memory_usage,
                'avg': sum(memory_usage) / len(memory_usage),
                'min': min(memory_usage),
                'max': max(memory_usage)
            },
            'disk': {
                'usage': disk_usage,
                'avg': sum(disk_usage) / len(disk_usage),
                'min': min(disk_usage),
                'max': max(disk_usage)
            },
            'total_records': len(history)
        }
        
        return trends
    
    def save_history_to_file(self, filename: str = None):
        """
        将监控历史保存到文件
        
        Args:
            filename: 文件名
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"system_monitor_history_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.monitor_history, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"System monitor history saved to {filename}")


# 全局实例
_system_monitor = None


def get_system_monitor(config: Dict = None) -> SystemMonitor:
    """
    获取或创建系统监控器实例
    """
    global _system_monitor
    if _system_monitor is None:
        _system_monitor = SystemMonitor(config)
    return _system_monitor