import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

from exchange.binance_exchange import BinanceExchange

class TestBinanceExchange(unittest.TestCase):

    def setUp(self):
        # Initialize without credentials for testing public endpoints
        self.exchange = BinanceExchange(sandbox=True)

    @patch('requests.get')
    def test_get_current_price_success(self, mock_get):
        """Test fetching the current price successfully."""
        # Arrange: Configure the mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'symbol': 'BTCUSDT', 'price': '65000.50'}
        mock_get.return_value = mock_response

        # Act: Call the method under test
        price = self.exchange.get_current_price('BTC-USDT')

        # Assert: Check the outcome
        self.assertEqual(price, 65000.50)
        # Verify that the correct URL was called
        expected_url = 'https://testnet.binance.vision/api/v3/ticker/price'
        mock_get.assert_called_once_with(
            expected_url,
            headers={},
            params={'symbol': 'BTCUSDT'},
            timeout=10
        )

    @patch('requests.get')
    def test_fetch_candles_success(self, mock_get):
        """Test fetching candles successfully."""
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 200
        # Binance API returns a list of lists for klines
        mock_response.json.return_value = [
            [
                1499040000000,      # Kline open time
                "4261.48000000",    # Open price
                "4313.62000000",    # High price
                "4261.32000000",    # Low price
                "4308.83000000",    # Close price
                "47.18100900",      # Volume
                1499644799999,      # Kline close time
                "202346.48040262",  # Quote asset volume
                337,                # Number of trades
                "28.35963500",      # Taker buy base asset volume
                "122209.95833461",  # Taker buy quote asset volume
                "0"                 # Unused field
            ]
        ]
        mock_get.return_value = mock_response

        # Act
        df = self.exchange.fetch_candles('BTC-USDT', '1h', limit=1)

        # Assert
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['close'], 4308.83)
        # Verify the call
        expected_url = 'https://testnet.binance.vision/api/v3/klines'
        mock_get.assert_called_once_with(
            expected_url,
            headers={},
            params={'symbol': 'BTCUSDT', 'interval': '1h', 'limit': 1},
            timeout=10
        )

    @patch('requests.post')
    def test_place_oco_order_success(self, mock_post):
        """Test placing an OCO order successfully."""
        # Arrange: Initialize with credentials for authenticated endpoint
        self.exchange = BinanceExchange(
            api_key='test_api_key',
            api_secret='test_secret_key',
            sandbox=True
        )

        # Mock the response from the server
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "orderListId": 0,
            "contingencyType": "OCO",
            "listStatusType": "EXEC_STARTED",
            "listOrderStatus": "EXECUTING",
            "listClientOrderId": "A123456789",
            "transactionTime": 1616544000000,
            "symbol": "BTCUSDT",
            "orders": [],
            "orderReports": []
        }
        mock_post.return_value = mock_response

        # Act: Place an OCO order
        response = self.exchange.place_oco_order(
            symbol='BTC-USDT',
            side='SELL',
            amount=0.1,
            take_profit_price=66000.0,
            stop_loss_price=64000.0
        )

        # Assert: Check the response and the call made
        self.assertIsNotNone(response)
        self.assertEqual(response['listStatusType'], 'EXEC_STARTED')

        # Get the actual call arguments
        called_args, called_kwargs = mock_post.call_args
        expected_url = 'https://testnet.binance.vision/api/v3/order/oco'
        self.assertEqual(called_args[0], expected_url)

        # Check headers
        self.assertEqual(called_kwargs['headers']['X-MBX-APIKEY'], 'test_api_key')

        # Check params - signature and timestamp will be dynamic
        sent_params = called_kwargs['params']
        self.assertIn('timestamp', sent_params)
        self.assertIn('signature', sent_params)
        self.assertEqual(sent_params['symbol'], 'BTCUSDT')
        self.assertEqual(sent_params['side'], 'SELL')
        self.assertEqual(sent_params['quantity'], '0.1')
        self.assertEqual(sent_params['price'], '66000.0')
        self.assertEqual(sent_params['stopPrice'], '64000.0')
        self.assertEqual(sent_params['stopLimitPrice'], '64000.0')
        self.assertEqual(sent_params['stopLimitTimeInForce'], 'GTC')

if __name__ == '__main__':
    unittest.main()
