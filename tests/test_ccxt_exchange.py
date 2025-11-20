import pytest
import pandas as pd
from unittest.mock import MagicMock, PropertyMock, patch, AsyncMock

@pytest.fixture
def exchange_and_mock(mocker):
    """
    A pytest fixture that provides a freshly initialized CcxtExchange instance,
    its internal mock exchange, mock exceptions, and mock loggers for each test.
    This version uses a simplified, more robust mocking strategy.
    """
    # 1. Create a single, comprehensive mock for the entire 'ccxt' module
    mock_ccxt = MagicMock(name='ccxt')

    # 2. Define the necessary mock exception classes on the auto-created sub-mocks
    errors = mock_ccxt.base.errors
    errors.BaseError = type('BaseError', (Exception,), {})
    errors.ExchangeError = type('ExchangeError', (errors.BaseError,), {})
    errors.NetworkError = type('NetworkError', (errors.BaseError,), {})
    errors.RequestTimeout = type('RequestTimeout', (errors.NetworkError,), {})
    errors.InvalidOrder = type('InvalidOrder', (errors.ExchangeError,), {})
    errors.OrderNotFound = type('OrderNotFound', (errors.InvalidOrder,), {})
    errors.InsufficientFunds = type('InsufficientFunds', (errors.BaseError,), {})
    errors.RateLimitExceeded = type('RateLimitExceeded', (errors.ExchangeError,), {})

    # 3. Create the mock exchange instance and the class that returns it
    mock_instance = MagicMock(
        spec=[
            'id', 'urls', 'has', 'set_sandbox_mode', 'load_markets', 'close', 
            'fetch_ohlcv', 'fetch_balance', 'create_order', 'fetch_order', 
            'cancel_order', 'fetch_ticker', 'rateLimit', 'parse_timeframe'
        ],
        load_markets=AsyncMock(),
        close=AsyncMock(),
        fetch_ohlcv=AsyncMock(),
        fetch_balance=AsyncMock(),
        create_order=AsyncMock(),
        fetch_order=AsyncMock(),
        cancel_order=AsyncMock(),
        fetch_ticker=AsyncMock(),
        rateLimit=100,
        parse_timeframe=MagicMock(return_value=60000)
    )
    mock_instance.id = 'mockexchange'
    mock_instance.urls = {'test': 'https://test.com'}
    type(mock_instance).has = PropertyMock(return_value={'createOco': False})
    
    mock_exchange_class = MagicMock(return_value=mock_instance)

    # 4. Attach the mock exchange class to the correct sub-mock
    # This is the key part that fixes the AttributeError.
    # The code under test does `import ccxt.async_support as ccxt`, which means
    # the `ccxt` variable in the code is the `async_support` module.
    # Then it calls `getattr(ccxt, 'mockexchange')`.
    mock_ccxt.async_support.mockexchange = mock_exchange_class

    # 5. Precisely patch the logger used within the ccxt_exchange module
    mock_loggers = {
        'info': mocker.patch('exchange.ccxt_exchange.logger.info'),
        'warning': mocker.patch('exchange.ccxt_exchange.logger.warning'),
        'critical': mocker.patch('exchange.ccxt_exchange.logger.critical'),
        'error': mocker.patch('exchange.ccxt_exchange.logger.error'),
    }

    # 6. Patch *only* the top-level 'ccxt' module into sys.modules
    with patch.dict('sys.modules', {'ccxt': mock_ccxt}):
        from exchange.ccxt_exchange import CcxtExchange
        
        exchange = CcxtExchange(exchange_id='mockexchange', api_key='test_key', api_secret='test_secret', sandbox=True)
        
        # 7. Yield all necessary components to the tests
        yield exchange, mock_instance, errors, mock_loggers


class TestCcxtExchange:

    def test_initialization(self, exchange_and_mock):
        exchange, mock_instance, _, _ = exchange_and_mock
        mock_instance.set_sandbox_mode.assert_called_with(True)
        assert exchange.exchange_id == 'mockexchange'

    async def test_fetch_candles(self, exchange_and_mock):
        exchange, mock_instance, _, _ = exchange_and_mock
        ohlcv_data = [[1672531200000, 100, 105, 99, 101, 1000]]
        mock_instance.fetch_ohlcv.return_value = ohlcv_data

        df = await exchange.fetch_candles(symbol='BTC-USDT', timeframe='1m', limit=1)

        mock_instance.fetch_ohlcv.assert_awaited_with('BTC/USDT', '1m', None, 1)
        assert not df.empty

    async def test_create_order_success(self, exchange_and_mock):
        """Test successful order creation."""
        exchange, mock_instance, _, _ = exchange_and_mock
        order_details = {'id': '12345', 'symbol': 'BTC/USDT'}
        mock_instance.create_order.return_value = order_details

        order = await exchange.create_order(symbol='BTC-USDT', order_type='limit', side='buy', amount=1, price=50000)
        
        mock_instance.create_order.assert_awaited_with('BTC/USDT', 'limit', 'buy', 1, 50000)
        assert order is not None
        assert order['id'] == '12345'

    async def test_create_order_insufficient_funds(self, exchange_and_mock):
        """Test create_order handles InsufficientFunds correctly."""
        exchange, mock_instance, errors, mock_loggers = exchange_and_mock
        mock_instance.create_order.side_effect = errors.InsufficientFunds('Not enough balance')

        order = await exchange.create_order(symbol='BTC-USDT', order_type='limit', side='buy', amount=1, price=50000)
        
        assert order is None
        mock_loggers['critical'].assert_called_once()

    async def test_create_order_network_error(self, exchange_and_mock):
        """Test create_order handles NetworkError correctly."""
        exchange, mock_instance, errors, mock_loggers = exchange_and_mock
        mock_instance.create_order.side_effect = errors.NetworkError('Connection failed')

        order = await exchange.create_order(symbol='BTC-USDT', order_type='limit', side='buy', amount=1, price=50000)
        
        assert order is None
        mock_loggers['warning'].assert_called_once()

    async def test_get_order_success(self, exchange_and_mock):
        """Test successful fetching of an order."""
        exchange, mock_instance, _, _ = exchange_and_mock
        order_details = {'id': '12345', 'status': 'closed'}
        mock_instance.fetch_order.return_value = order_details

        order = await exchange.get_order(order_id='12345', symbol='BTC-USDT')
        
        mock_instance.fetch_order.assert_awaited_with('12345', 'BTC/USDT')
        assert order is not None
        assert order['status'] == 'closed'

    async def test_get_order_not_found(self, exchange_and_mock):
        """Test get_order handles OrderNotFound correctly."""
        exchange, mock_instance, errors, mock_loggers = exchange_and_mock
        mock_instance.fetch_order.side_effect = errors.OrderNotFound('Order not found')

        order = await exchange.get_order(order_id='12345', symbol='BTC-USDT')
        
        assert order is None
        mock_loggers['warning'].assert_called_once()

    async def test_get_order_network_error(self, exchange_and_mock):
        """Test get_order handles NetworkError correctly."""
        exchange, mock_instance, errors, mock_loggers = exchange_and_mock
        mock_instance.fetch_order.side_effect = errors.NetworkError('Connection failed')

        order = await exchange.get_order(order_id='12345', symbol='BTC-USDT')
        
        assert order is None
        mock_loggers['warning'].assert_called_once()

    async def test_cancel_order_success(self, exchange_and_mock):
        """Test successful order cancellation."""
        exchange, mock_instance, _, mock_loggers = exchange_and_mock
        cancel_details = {'id': '12345', 'status': 'canceled'}
        mock_instance.cancel_order.return_value = cancel_details

        result = await exchange.cancel_order(order_id='12345', symbol='BTC-USDT')
        
        mock_instance.cancel_order.assert_awaited_with('12345', 'BTC/USDT')
        assert result is not None
        assert result['status'] == 'canceled'
        mock_loggers['info'].assert_called_once()


    async def test_cancel_order_not_found(self, exchange_and_mock):
        """Test cancel_order handles OrderNotFound correctly."""
        exchange, mock_instance, errors, mock_loggers = exchange_and_mock
        mock_instance.cancel_order.side_effect = errors.OrderNotFound('Order already filled or cancelled')

        result = await exchange.cancel_order(order_id='12345', symbol='BTC-USDT')
        
        assert result is None
        mock_loggers['info'].assert_called_with("Attempted to cancel order '12345' on mockexchange, but it was not found (already filled/cancelled?). Error: Order already filled or cancelled")


    async def test_cancel_order_network_error(self, exchange_and_mock):
        """Test cancel_order handles NetworkError correctly."""
        exchange, mock_instance, errors, mock_loggers = exchange_and_mock
        mock_instance.cancel_order.side_effect = errors.NetworkError('Connection failed')

        result = await exchange.cancel_order(order_id='12345', symbol='BTC-USDT')
        
        assert result is None
        mock_loggers['warning'].assert_called_once()

    async def test_fetch_candles_network_error(self, exchange_and_mock):
        """Test that fetch_candles handles network errors gracefully."""
        exchange, mock_instance, errors, mock_loggers = exchange_and_mock
        mock_instance.fetch_ohlcv.side_effect = errors.NetworkError('API connection failed')
        df = await exchange.fetch_candles(symbol='BTC-USDT', timeframe='1m')
        assert isinstance(df, pd.DataFrame)
        assert df.empty
        mock_loggers['error'].assert_called_once()


    async def test_get_balance_network_error(self, exchange_and_mock):
        """Test that get_balance handles network errors."""
        exchange, mock_instance, errors, mock_loggers = exchange_and_mock
        mock_instance.fetch_balance.side_effect = errors.NetworkError('API connection failed')
        balance = await exchange.get_balance('USDT')
        assert balance == 0.0
        mock_loggers['error'].assert_called_once()
