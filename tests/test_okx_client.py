import unittest
from unittest.mock import patch, MagicMock
import requests
import os

os.environ['OKX_API_KEY'] = 'test_key'
os.environ['OKX_SECRET_KEY'] = 'test_secret'
os.environ['OKX_PASSPHRASE'] = 'test_passphrase'

from trader.okx_client import OKXClient

class TestOKXClientRetry(unittest.TestCase):

    def setUp(self):
        self.client = OKXClient(flag='0')

    @patch('requests.get')
    def test_request_success_on_first_try(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'code': '0', 'data': [{'instId': 'BTC-USDT'}]}
        mock_get.return_value = mock_response
        result = self.client._request('GET', '/api/v5/market/tickers')
        mock_get.assert_called_once()
        self.assertEqual(result['data'][0]['instId'], 'BTC-USDT')

    @patch('requests.get')
    def test_request_retries_and_succeeds(self, mock_get):
        mock_error = MagicMock()
        mock_error.raise_for_status.side_effect = requests.exceptions.HTTPError()
        mock_error.status_code = 502
        mock_success = MagicMock()
        mock_success.status_code = 200
        mock_success.json.return_value = {'msg': 'Success'}
        mock_get.side_effect = [mock_error, mock_error, mock_success]
        result = self.client._request('GET', '/api/v5/path', max_retries=3, delay=0.1)
        self.assertEqual(mock_get.call_count, 3)
        self.assertEqual(result['msg'], 'Success')

    @patch('requests.get')
    def test_request_fails_after_max_retries(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout()
        result = self.client._request('GET', '/api/v5/path', max_retries=3, delay=0.1)
        self.assertEqual(mock_get.call_count, 3)
        self.assertIsNone(result)