
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import pandas as pd
import asyncio

# Mock dependencies before import
with patch.dict('sys.modules', {
    'utils.config_loader': MagicMock(),
    'utils.logger': MagicMock(),
    'exchange.ccxt_exchange': MagicMock(),
}):
    from exchange.aggregated_exchange import AggregatedExchange
    from exchange.factory import ExchangeFactory

class TestAggregatedExchange(unittest.TestCase):

    def setUp(self):
        """Set up mock exchanges before each test."""
        # Mock the factory to return our mock exchanges
        self.mock_primary_exchange = MagicMock()
        self.mock_primary_exchange.__class__.__name__ = 'MockPrimary'
        self.mock_secondary_exchange = MagicMock()
        self.mock_secondary_exchange.__class__.__name__ = 'MockSecondary'

        # Configure the factory mock
        def factory_side_effect(exchange_id, **kwargs):
            if exchange_id == 'binance':
                return self.mock_primary_exchange
            if exchange_id == 'okx':
                return self.mock_secondary_exchange
            return MagicMock()

        ExchangeFactory.create_exchange = MagicMock(side_effect=factory_side_effect)

        # Mock config loader
        from utils.config_loader import load_config
        load_config.return_value = {
            'exchange': {
                'primary_exchange': 'binance',
                'secondary_exchanges': ['okx']
            }
        }

        # Instantiate the class we are testing
        self.aggregated_exchange = AggregatedExchange(
            primary_exchange_id='binance',
            secondary_exchange_ids=['okx']
        )

    def test_initialization(self):
        """Test that the aggregated exchange initializes correctly."""
        self.assertEqual(self.aggregated_exchange.primary_exchange, self.mock_primary_exchange)
        self.assertIn(self.mock_primary_exchange, self.aggregated_exchange.all_exchanges)
        self.assertIn(self.mock_secondary_exchange, self.aggregated_exchange.all_exchanges)
        self.assertEqual(len(self.aggregated_exchange.all_exchanges), 2)

    def test_fetch_historical_data_uses_primary_only(self):
        """Test that fetching historical data only calls the primary exchange."""
        self.aggregated_exchange.fetch_historical_data('BTC/USDT', '1h', 1)

        # Assert that only the primary exchange was called
        self.mock_primary_exchange.fetch_historical_data.assert_called_once_with('BTC/USDT', '1h', 1)
        # Assert that the secondary exchange was NOT called
        self.mock_secondary_exchange.fetch_historical_data.assert_not_called()

    def test_get_current_price_aggregates(self):
        """Test that get_current_price averages the prices from all exchanges."""
        self.mock_primary_exchange.get_current_price.return_value = 100.0
        self.mock_secondary_exchange.get_current_price.return_value = 102.0

        price = self.aggregated_exchange.get_current_price('BTC/USDT')

        self.assertEqual(price, 101.0)
        self.mock_primary_exchange.get_current_price.assert_called_once_with('BTC/USDT')
        self.mock_secondary_exchange.get_current_price.assert_called_once_with('BTC/USDT')

    def test_get_current_price_with_one_failure(self):
        """Test that get_current_price works even if one exchange fails."""
        self.mock_primary_exchange.get_current_price.return_value = 100.0
        self.mock_secondary_exchange.get_current_price.side_effect = Exception("API Error")

        price = self.aggregated_exchange.get_current_price('BTC/USDT')

        self.assertEqual(price, 100.0)

    def test_fetch_candles_aggregation(self):
        """Test the aggregation logic of fetch_candles."""
        # Create sample dataframes
        df1 = pd.DataFrame({
            'open': [100], 'high': [110], 'low': [99], 'close': [105], 'volume': [1000]
        }, index=pd.to_datetime(['2023-01-01']))
        
        df2 = pd.DataFrame({
            'open': [102], 'high': [108], 'low': [98], 'close': [103], 'volume': [1500]
        }, index=pd.to_datetime(['2023-01-01']))

        # Mock the async fetch to return these dataframes
        async def mock_fetch_candles_async(exchange, *args, **kwargs):
            if exchange == self.mock_primary_exchange:
                return df1
            if exchange == self.mock_secondary_exchange:
                return df2
            return None
        
        self.aggregated_exchange.fetch_candles_async = AsyncMock(side_effect=mock_fetch_candles_async)

        # Run fetch_candles
        result_df = self.aggregated_exchange.fetch_candles('BTC/USDT', '1h', limit=1)

        # Check aggregated values
        self.assertEqual(result_df.iloc[0]['open'], 101)      # (100+102)/2
        self.assertEqual(result_df.iloc[0]['high'], 110)      # max(110, 108)
        self.assertEqual(result_df.iloc[0]['low'], 98)       # min(99, 98)
        self.assertEqual(result_df.iloc[0]['close'], 104)     # (105+103)/2
        self.assertEqual(result_df.iloc[0]['volume'], 2500)   # 1000+1500

if __name__ == '__main__':
    unittest.main()
