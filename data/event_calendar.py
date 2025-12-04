"""
宏观事件日历模块
用于跟踪和预警重要的宏观经济事件
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class MacroEventCalendar:
    """
    宏观事件日历
    
    功能:
    1. 跟踪定期事件 (FOMC, CPI, NFP等)
    2. 跟踪一次性事件 (重大政策变化等)
    3. 提前预警即将到来的事件
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        初始化事件日历
        
        Args:
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # 定期事件配置
        self.recurring_events = {
            'FOMC': {
                'description': 'Federal Reserve FOMC Meeting',
                'impact': 'CRITICAL',
                'action': 'pause_trading',
                'dates': [  # 2025 年 FOMC 会议日期
                    '2025-01-29', '2025-03-19', '2025-05-07',
                    '2025-06-18', '2025-07-30', '2025-09-17',
                    '2025-11-05', '2025-12-17'
                ]
            },
            'CPI': {
                'description': 'US Consumer Price Index Release',
                'impact': 'HIGH',
                'action': 'reduce_position',
                'dates': [  # 2025 年 CPI 发布日期 (示例,需要更新)
                    '2025-01-15', '2025-02-13', '2025-03-12',
                    '2025-04-10', '2025-05-14', '2025-06-11',
                    '2025-07-11', '2025-08-13', '2025-09-11',
                    '2025-10-10', '2025-11-13', '2025-12-10'
                ]
            },
            'NFP': {
                'description': 'US Non-Farm Payrolls',
                'impact': 'MEDIUM',
                'action': 'reduce_position',
                'dates': [  # 2025 年 NFP 发布日期 (每月第一个周五,示例)
                    '2025-01-10', '2025-02-07', '2025-03-07',
                    '2025-04-04', '2025-05-02', '2025-06-06',
                    '2025-07-03', '2025-08-01', '2025-09-05',
                    '2025-10-03', '2025-11-07', '2025-12-05'
                ]
            }
        }
        
        # 一次性事件 (需要手动添加)
        self.one_time_events = [
            {
                'date': '2025-01-20',
                'description': 'US Presidential Inauguration',
                'impact': 'HIGH',
                'action': 'reduce_position'
            },
            # 可以手动添加其他重要事件
        ]
        
        self.logger.info(f"MacroEventCalendar initialized with {len(self.recurring_events)} recurring event types")
    
    def check_upcoming_events(self, days_ahead: int = 3) -> Dict:
        """
        检查未来 N 天是否有重要事件
        
        Args:
            days_ahead: 提前多少天检查
            
        Returns:
            {
                'has_event': True/False,
                'event': {...} or None,
                'days_until': N or None
            }
        """
        today = datetime.utcnow().date()
        
        # 检查定期事件
        for event_type, event_info in self.recurring_events.items():
            if 'dates' in event_info:
                for date_str in event_info['dates']:
                    try:
                        event_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                        days_until = (event_date - today).days
                        
                        if 0 <= days_until <= days_ahead:
                            self.logger.info(f"Upcoming event: {event_type} in {days_until} days")
                            return {
                                'has_event': True,
                                'event': {
                                    'type': event_type,
                                    'date': date_str,
                                    **event_info
                                },
                                'days_until': days_until
                            }
                    except ValueError as e:
                        self.logger.error(f"Invalid date format {date_str}: {e}")
                        continue
        
        # 检查一次性事件
        for event in self.one_time_events:
            try:
                event_date = datetime.strptime(event['date'], '%Y-%m-%d').date()
                days_until = (event_date - today).days
                
                if 0 <= days_until <= days_ahead:
                    self.logger.info(f"Upcoming one-time event: {event['description']} in {days_until} days")
                    return {
                        'has_event': True,
                        'event': event,
                        'days_until': days_until
                    }
            except ValueError as e:
                self.logger.error(f"Invalid date format {event['date']}: {e}")
                continue
        
        return {'has_event': False, 'event': None, 'days_until': None}
    
    def get_action_for_event(self, event: Dict) -> str:
        """
        根据事件影响级别决定行动
        
        Args:
            event: 事件字典
            
        Returns:
            'pause_trading' | 'reduce_position' | 'normal'
        """
        action = event.get('action', 'normal')
        return action
    
    def get_position_multiplier(self, event: Dict) -> float:
        """
        根据事件影响级别返回仓位调整系数
        
        Args:
            event: 事件字典
            
        Returns:
            仓位系数 (0.0 - 1.0)
        """
        impact = event.get('impact', 'LOW')
        
        if impact == 'CRITICAL':
            return 0.0  # 暂停交易
        elif impact == 'HIGH':
            return 0.3  # 降低到 30%
        elif impact == 'MEDIUM':
            return 0.5  # 降低到 50%
        else:
            return 1.0  # 正常
    
    def add_one_time_event(self, date: str, description: str, 
                          impact: str = 'MEDIUM', action: str = 'reduce_position'):
        """
        添加一次性事件
        
        Args:
            date: 日期 (YYYY-MM-DD)
            description: 事件描述
            impact: 影响级别 ('CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW')
            action: 建议行动 ('pause_trading' | 'reduce_position' | 'normal')
        """
        self.one_time_events.append({
            'date': date,
            'description': description,
            'impact': impact,
            'action': action
        })
        self.logger.info(f"Added one-time event: {description} on {date}")
    
    def get_summary(self) -> str:
        """获取日历状态摘要"""
        total_recurring = sum(len(e.get('dates', [])) for e in self.recurring_events.values())
        return (f"MacroEventCalendar: "
                f"{total_recurring} recurring events, "
                f"{len(self.one_time_events)} one-time events")


def get_event_calendar(config: Dict = None, logger: logging.Logger = None) -> MacroEventCalendar:
    """
    工厂函数: 创建事件日历实例
    
    Args:
        config: 配置字典 (预留,暂未使用)
        logger: 日志记录器
        
    Returns:
        MacroEventCalendar 实例
    """
    calendar = MacroEventCalendar(logger=logger)
    
    # 如果配置中有自定义事件,可以在这里添加
    if config and 'custom_events' in config:
        for event in config['custom_events']:
            calendar.add_one_time_event(
                date=event['date'],
                description=event['description'],
                impact=event.get('impact', 'MEDIUM'),
                action=event.get('action', 'reduce_position')
            )
    
    return calendar
