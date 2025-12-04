import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from ccxt.base.errors import InsufficientFunds, NetworkError
from exchange.ccxt_exchange import CcxtExchange

pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_exchange_instance():
    """
    A more robust fixture that mocks the ccxt library at the module level,
    allowing CcxtExchange to perform its initialization logic.
    """
    # This is the mock for the exchange *instance* (e.g., what binance() returns)
    mock_exchange_obj = MagicMock()
    mock_exchange_obj.create_order = AsyncMock()
    mock_exchange_obj.urls = {'test': 'https://test.binance.com'} # Needed for sandbox mode logic
    mock_exchange_obj.set_sandbox_mode = MagicMock()

    # This is the mock for the exchange *class* (e.g., ccxt.async_support.binance)
    mock_exchange_class = MagicMock(return_value=mock_exchange_obj)

    # We patch the 'async_support' module that is imported in our ccxt_exchange.py
    with patch('exchange.ccxt_exchange.async_support') as mock_async_support:
        # Configure the mock module to return our mock class
        # when getattr(async_support, 'binance') is called
        setattr(mock_async_support, 'binance', mock_exchange_class)
        
        # Yield the configured mock object so we can inspect it in tests
        yield mock_exchange_obj

async def test_create_order_sufficient_funds(mock_exchange_instance):
    """
    Test that create_order returns the order dictionary when the API call is successful.
    """
    # The fixture has already mocked the backend, so we can just instantiate our class
    exchange = CcxtExchange(exchange_id='binance', sandbox=True)
    
    expected_order = {'id': '123', 'symbol': 'BTC/USDT', 'status': 'open'}
    mock_exchange_instance.create_order.return_value = expected_order
    
    result = await exchange.create_order(
        symbol='BTC-USDT',
        order_type='limit',
        side='buy',
        amount=1,
        price=50000
    )
    
    assert result == expected_order
    mock_exchange_instance.create_order.assert_called_once_with('BTC/USDT', 'limit', 'buy', 1, 50000)

async def test_create_order_insufficient_funds(mock_exchange_instance):
    """
    Test that create_order returns None when ccxt raises InsufficientFunds.
    """
    exchange = CcxtExchange(exchange_id='binance', sandbox=True)
    
    mock_exchange_instance.create_order.side_effect = InsufficientFunds("Not enough balance")
    
    result = await exchange.create_order(
        symbol='BTC-USDT',
        order_type='limit',
        side='buy',
        amount=1,
        price=50000
    )
    
    assert result is None
    mock_exchange_instance.create_order.assert_called_once()

async def test_create_order_network_error(mock_exchange_instance):
    """
    Test that create_order returns None when ccxt raises a NetworkError.
    """
    exchange = CcxtExchange(exchange_id='binance', sandbox=True)
    
    mock_exchange_instance.create_order.side_effect = NetworkError("Connection timed out")
    
    result = await exchange.create_order(
        symbol='BTC-USDT',
        order_type='limit',
        side='buy',
        amount=1,
        price=50000
    )
    
    assert result is None
    mock_exchange_instance.create_order.assert_called_once()