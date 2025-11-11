
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import pandas as pd

# Mock ccxt before it's imported by other modules
with patch.dict('sys.modules', {'ccxt': MagicMock()}) as mock_sys:
    mock_ccxt = mock_sys['ccxt']
    mock_exchange_class = MagicMock()
    mock_exchange_instance = MagicMock()
    mock_exchange_instance.id = 'mockexchange'
    mock_exchange_instance.urls = {'test': 'https://test.com'}
    type(mock_exchange_instance).has = PropertyMock(return_value={'createOco': False})
    mock_ccxt.mockexchange = mock_exchange_class
    mock_exchange_class.return_value = mock_exchange_instance

    from exchange.ccxt_exchange import CcxtExchange

class TestCcxtExchange(unittest.TestCase):

    def setUp(self):
        """Set up a mock for the ccxt library before each test."""
        # The patch is now managed globally for the module import
        self.exchange = CcxtExchange(exchange_id='mockexchange', api_key='test_key', api_secret='test_secret', sandbox=True)
        self.exchange.exchange = mock_exchange_instance
        self.mock_exchange_instance = mock_exchange_instance
        self.mock_ccxt = mock_ccxt
        self.mock_exchange_class = mock_exchange_class

    def tearDown(self):
        """Stop the patcher after each test."""
        # No need to stop a patcher here as it's handled by the with statement
        pass

    def test_initialization(self):
        """Test that the exchange is initialized correctly."""
        self.mock_ccxt.mockexchange.assert_called_with({
            'apiKey': 'test_key',
            'secret': 'test_secret',
        })
        self.mock_exchange_instance.set_sandbox_mode.assert_called_with(True)
        self.assertEqual(self.exchange.exchange_id, 'mockexchange')

    def test_fetch_candles(self):
        """Test fetching candle data."""
        # Sample OHLCV data from ccxt
        ohlcv_data = [
            [1672531200000, 100, 105, 99, 101, 1000],
            [1672531260000, 101, 106, 100, 102, 1200],
        ]
        self.mock_exchange_instance.fetch_ohlcv.return_value = ohlcv_data

        df = self.exchange.fetch_candles(symbol='BTC/USDT', timeframe='1m', limit=2)

        self.mock_exchange_instance.fetch_ohlcv.assert_called_with('BTC/USDT', '1m', None, 2)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]['open'], 100)
        self.assertEqual(df.index[0], pd.to_datetime('2023-01-01 00:00:00'))

    def test_get_balance(self):
        """Test fetching balance."""
        balance_data = {
            'free': {'USDT': 1000.0, 'BTC': 1.5},
            'total': {'USDT': 1000.0, 'BTC': 1.5},
        }
        self.mock_exchange_instance.fetch_balance.return_value = balance_data

        balance = self.exchange.get_balance('USDT')
        self.assertEqual(balance, 1000.0)

        balance_btc = self.exchange.get_balance('BTC')
        self.assertEqual(balance_btc, 1.5)

        balance_eth = self.exchange.get_balance('ETH')
        self.assertEqual(balance_eth, 0.0)

    def test_create_order(self):
        """Test creating an order."""
        order_details = {'id': '12345', 'symbol': 'BTC/USDT'}
        self.mock_exchange_instance.create_order.return_value = order_details

        order = self.exchange.create_order(symbol='BTC/USDT', order_type='limit', side='buy', amount=1, price=50000)
        self.mock_exchange_instance.create_order.assert_called_with('BTC/USDT', 'limit', 'buy', 1, 50000)
        self.assertEqual(order['id'], '12345')

    def test_get_order(self):
        """Test fetching an order."""
        order_details = {'id': '12345', 'status': 'closed'}
        self.mock_exchange_instance.fetch_order.return_value = order_details

        order = self.exchange.get_order(order_id='12345', symbol='BTC/USDT')
        self.mock_exchange_instance.fetch_order.assert_called_with('12345', 'BTC/USDT')
        self.assertEqual(order['status'], 'closed')

    def test_cancel_order(self):
        """Test canceling an order."""
        cancel_details = {'id': '12345', 'status': 'canceled'}
        self.mock_exchange_instance.cancel_order.return_value = cancel_details

        result = self.exchange.cancel_order(order_id='12345', symbol='BTC/USDT')
        self.mock_exchange_instance.cancel_order.assert_called_with('12345', 'BTC/USDT')
        self.assertEqual(result['status'], 'canceled')

    def test_get_current_price(self):
        """Test fetching the current price."""
        ticker_data = {'last': 52000.0}
        self.mock_exchange_instance.fetch_ticker.return_value = ticker_data

        price = self.exchange.get_current_price(symbol='BTC/USDT')
        self.mock_exchange_instance.fetch_ticker.assert_called_with('BTC/USDT')
        self.assertEqual(price, 52000.0)

    def test_place_oco_order_not_supported(self):
        """Test OCO order when not supported by the exchange."""
        with self.assertRaises(NotImplementedError):
            self.exchange.place_oco_order(symbol='BTC/USDT', side='sell', amount=1, take_profit_price=55000, stop_loss_price=49000)

    def test_place_oco_order_supported(self):
        """Test OCO order when supported by the exchange."""
        # Re-configure the mock to support OCO
        type(self.mock_exchange_instance).has = PropertyMock(return_value={'createOco': True})
        
        oco_order_details = {'id': 'oco1', 'info': {}}
        self.mock_exchange_instance.create_order.return_value = oco_order_details

        order = self.exchange.place_oco_order(symbol='BTC/USDT', side='sell', amount=1, take_profit_price=55000, stop_loss_price=49000)

        expected_params = {
            'stopPrice': 49000,
            'stopLimitPrice': 49000,
        }
        self.mock_exchange_instance.create_order.assert_called_with(
            symbol='BTC/USDT',
            type='oco',
            side='sell',
            amount=1,
            price=55000,
            params=expected_params
        )
        self.assertEqual(order['id'], 'oco1')

if __name__ == '__main__':
    unittest.main()
