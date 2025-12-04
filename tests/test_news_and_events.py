import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.news_monitor import CryptoPanicMonitor
from data.event_calendar import MacroEventCalendar

class TestNewsAndEvents(unittest.TestCase):
    def setUp(self):
        self.news_monitor = CryptoPanicMonitor()
        self.calendar = MacroEventCalendar()

    def test_fatal_news_trigger_emergency_stop(self):
        """Test that fatal news triggers emergency stop"""
        # Mock news items
        now = datetime.utcnow()
        news_items = [
            {
                'title': 'SEC files lawsuit against Binance',
                'published_at': now.isoformat(),
                'url': 'http://example.com',
                'votes': {'positive': 100}
            }
        ]
        
        impact = self.news_monitor.analyze_news_impact(news_items)
        
        self.assertEqual(impact['action'], 'emergency_stop')
        self.assertEqual(impact['severity'], 'CRITICAL')
        self.assertTrue('sec' in impact['reason'] or 'lawsuit' in impact['reason'])

    def test_warning_news_trigger_reduce_position(self):
        """Test that multiple warning news trigger reduce position"""
        now = datetime.utcnow()
        # Create 3 warning news items
        news_items = [
            {
                'title': f'Market risk increases {i}',
                'published_at': now.isoformat(),
                'url': 'http://example.com'
            } for i in range(3)
        ]
        
        impact = self.news_monitor.analyze_news_impact(news_items)
        
        self.assertEqual(impact['action'], 'reduce_position')
        self.assertEqual(impact['severity'], 'WARNING')

    def test_benign_news_normal_action(self):
        """Test that benign news results in normal action"""
        now = datetime.utcnow()
        news_items = [
            {
                'title': 'Bitcoin hits new high',
                'published_at': now.isoformat(),
                'url': 'http://example.com'
            }
        ]
        
        impact = self.news_monitor.analyze_news_impact(news_items)
        
        self.assertEqual(impact['action'], 'normal')
        self.assertEqual(impact['severity'], 'INFO')

    def test_old_news_ignored(self):
        """Test that old news is ignored"""
        old_time = datetime.utcnow() - timedelta(minutes=10)
        news_items = [
            {
                'title': 'SEC lawsuit',
                'published_at': old_time.isoformat(),
                'url': 'http://example.com'
            }
        ]
        
        impact = self.news_monitor.analyze_news_impact(news_items)
        
        self.assertEqual(impact['action'], 'normal')
        self.assertEqual(impact['severity'], 'INFO')

    @patch('data.event_calendar.datetime')
    def test_macro_event_detection(self, mock_datetime):
        """Test detection of upcoming macro events"""
        # Mock today as 2025-01-26 (3 days before FOMC on 2025-01-29)
        mock_now = datetime(2025, 1, 26)
        mock_datetime.utcnow.return_value = mock_now
        mock_datetime.strptime = datetime.strptime # Keep original strptime
        
        # We need to mock .date() on the return value of utcnow()
        # But datetime.utcnow() returns a datetime object, so we can just use the mock_now object
        # However, in the code: today = datetime.utcnow().date()
        # So mock_datetime.utcnow.return_value must have a .date() method.
        # datetime objects already have .date(), so returning a real datetime object is fine.
        
        upcoming = self.calendar.check_upcoming_events(days_ahead=3)
        
        self.assertTrue(upcoming['has_event'])
        self.assertEqual(upcoming['event']['type'], 'FOMC')
        self.assertEqual(upcoming['days_until'], 3)
        self.assertEqual(upcoming['event']['action'], 'pause_trading')

    @patch('data.event_calendar.datetime')
    def test_no_upcoming_events(self, mock_datetime):
        """Test when there are no upcoming events"""
        # Mock a date with no events (e.g., 2025-01-01)
        mock_now = datetime(2025, 1, 1)
        mock_datetime.utcnow.return_value = mock_now
        mock_datetime.strptime = datetime.strptime
        
        upcoming = self.calendar.check_upcoming_events(days_ahead=3)
        
        self.assertFalse(upcoming['has_event'])

if __name__ == '__main__':
    unittest.main()
