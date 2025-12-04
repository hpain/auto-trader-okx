import unittest
from unittest.mock import MagicMock, patch, mock_open
import sys
import os
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock yaml before importing modules that depend on it
sys.modules['yaml'] = MagicMock()
sys.modules['ta'] = MagicMock()
sys.modules['ta.momentum'] = MagicMock()
sys.modules['ta.trend'] = MagicMock()
sys.modules['ta.volatility'] = MagicMock()
sys.modules['lightgbm'] = MagicMock()
sys.modules['ccxt'] = MagicMock()
sys.modules['okx'] = MagicMock()
sys.modules['dune_client'] = MagicMock()
sys.modules['dune_client.client'] = MagicMock()
sys.modules['dotenv'] = MagicMock()
sys.modules['vaderSentiment'] = MagicMock()
sys.modules['vaderSentiment.vaderSentiment'] = MagicMock()
sys.modules['joblib'] = MagicMock()
sys.modules['pyarrow'] = MagicMock()
sys.modules['sklearn'] = MagicMock()
sys.modules['sklearn.model_selection'] = MagicMock()
sys.modules['sklearn.metrics'] = MagicMock()
sys.modules['sklearn.preprocessing'] = MagicMock()
sys.modules['sklearn.ensemble'] = MagicMock()
sys.modules['optuna'] = MagicMock()
sys.modules['shap'] = MagicMock()
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['seaborn'] = MagicMock()
sys.modules['psutil'] = MagicMock()
sys.modules['GPUtil'] = MagicMock()

from run_autonomous import AutonomousTrader

class TestAutonomousNewsIntegration(unittest.TestCase):
    def setUp(self):
        # Mock config
        self.config_patcher = patch('run_autonomous.config')
        self.mock_config = self.config_patcher.start()
        self.mock_config.get.return_value = {}
        
        # Mock other dependencies to avoid initialization errors
        self.patches = [
            patch('run_autonomous.OKXExchange'),
            patch('run_autonomous.StrategyManager'),
            patch('run_autonomous.PerformanceMonitor'),
            patch('run_autonomous.get_enhanced_monitor'),
            patch('run_autonomous.get_slippage_monitor'),
            patch('run_autonomous.get_portfolio_manager'),
            patch('run_autonomous.get_model_interpretability'),
            patch('run_autonomous.get_system_monitor'),
            patch('run_autonomous.StateManager'),
            patch('run_autonomous.MarketRegimeDetector'),
            patch('run_autonomous.get_news_monitor'),
            patch('run_autonomous.get_event_calendar'),
        ]
        
        for p in self.patches:
            p.start()
            
        # Initialize trader
        self.trader = AutonomousTrader()
        
        # Setup mock news monitor and event calendar
        self.trader.news_monitor = MagicMock()
        self.trader.event_calendar = MagicMock()
        self.trader.logger = MagicMock()

    def tearDown(self):
        self.config_patcher.stop()
        for p in self.patches:
            p.stop()

    def test_sense_triggers_event_pause(self):
        """Test that sense() pauses trading on critical macro events"""
        # Mock upcoming critical event
        self.trader.event_calendar.check_upcoming_events.return_value = {
            'has_event': True,
            'event': {
                'description': 'FOMC Meeting',
                'impact': 'CRITICAL',
                'action': 'pause_trading',
                'date': '2025-01-29'
            },
            'days_until': 0
        }
        self.trader.event_calendar.get_position_multiplier.return_value = 0.0
        
        # Mock file opening for flag creation
        m = mock_open()
        with patch('builtins.open', m):
            self.trader.sense()
            
        # Verify flag file was written
        m.assert_called_with('event_pause.flag', 'w')
        handle = m()
        handle.write.assert_any_call('Event: FOMC Meeting\n')
        
        # Verify logger was called
        self.trader.logger.critical.assert_any_call("⛔ Pausing trading due to FOMC Meeting")

    def test_sense_reduces_position_on_event(self):
        """Test that sense() reduces position on high impact events"""
        # Mock upcoming high impact event
        self.trader.event_calendar.check_upcoming_events.return_value = {
            'has_event': True,
            'event': {
                'description': 'CPI Release',
                'impact': 'HIGH',
                'action': 'reduce_position'
            },
            'days_until': 1
        }
        self.trader.event_calendar.get_position_multiplier.return_value = 0.5
        
        self.trader.sense()
        
        # Verify multiplier was set
        self.assertEqual(self.trader.event_position_multiplier, 0.5)
        self.trader.logger.warning.assert_any_call("🔻 Reducing position to 50.0% due to upcoming event")

    def test_decide_and_act_triggers_emergency_stop(self):
        """Test that decide_and_act() stops trading on critical news"""
        # Mock critical news
        self.trader.news_monitor.fetch_latest_news.return_value = [{'title': 'SEC Lawsuit'}]
        self.trader.news_monitor.analyze_news_impact.return_value = {
            'action': 'emergency_stop',
            'severity': 'CRITICAL',
            'reason': 'Regulatory ban',
            'news': [{'title': 'SEC Lawsuit', 'keyword': 'lawsuit', 'url': 'http://test'}]
        }
        
        # Mock file opening
        m = mock_open()
        with patch('builtins.open', m):
            self.trader.decide_and_act({}, {})
            
        # Verify flag file was written
        m.assert_called_with('emergency_stop.flag', 'w')
        handle = m()
        handle.write.assert_any_call('Reason: Regulatory ban\n')
        
        # Verify logger
        self.trader.logger.critical.assert_any_call("⛔ STOPPING TRADING")

    def test_decide_and_act_reduces_position_on_warning(self):
        """Test that decide_and_act() reduces position on warning news"""
        # Mock warning news
        self.trader.news_monitor.fetch_latest_news.return_value = [{'title': 'Risk Warning'}]
        self.trader.news_monitor.analyze_news_impact.return_value = {
            'action': 'reduce_position',
            'severity': 'WARNING',
            'reason': 'Multiple warnings',
            'news': [{'title': 'Risk Warning'}]
        }
        
        # Mock config for warning multiplier
        self.mock_config.get.return_value = {'warning_position_multiplier': 0.5}
        
        self.trader.decide_and_act({}, {})
        
        # Verify multiplier was set
        self.assertEqual(self.trader.news_position_multiplier, 0.5)
        self.trader.logger.warning.assert_any_call("🔻 Reducing position to 50.0% due to news")

if __name__ == '__main__':
    unittest.main()
