
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from ccxt.base.errors import InsufficientFunds, NetworkError

from exchange.ccxt_exchange import CcxtExchange

# Since we are dealing with async methods, we need to mark our tests with pytest-asyncio
pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_ccxt_exchange_instance():
    """Fixture to create a mocked CCXT exchange instance."""
    # Mock the underlying ccxt exchange object
    mock_exchange = MagicMock()
    mock_exchange.create_order = AsyncMock()
    
    # Create an instance of our CcxtExchange wrapper
    exchange = CcxtExchange(exchange_id='binance', sandbox=True)
    # Replace the actual exchange object with our mock
    exchange.exchange = mock_exchange
    
    return exchange

async def test_create_order_sufficient_funds(mock_ccxt_exchange_instance):
    """
    Test that create_order returns the order dictionary when the API call is successful.
    """
    exchange = mock_ccxt_exchange_instance
    
    # Configure the mock to return a successful order
    expected_order = {'id': '123', 'symbol': 'BTC/USDT', 'status': 'open'}
    exchange.exchange.create_order.return_value = expected_order
    
    # Call the method
    result = await exchange.create_order(
        symbol='BTC-USDT',
        order_type='limit',
        side='buy',
        amount=1,
        price=50000
    )
    
    # Assert the result is what we expect
    assert result == expected_order
    exchange.exchange.create_order.assert_called_once()

async def test_create_order_insufficient_funds(mock_ccxt_exchange_instance):
    """
    Test that create_order returns None when ccxt raises InsufficientFunds.
    """
    exchange = mock_ccxt_exchange_instance
    
    # Configure the mock to raise an exception
    exchange.exchange.create_order.side_effect = InsufficientFunds("Not enough balance")
    
    # Call the method
    result = await exchange.create_order(
        symbol='BTC-USDT',
        order_type='limit',
        side='buy',
        amount=1,
        price=50000
    )
    
    # Assert the result is None
    assert result is None
    exchange.exchange.create_order.assert_called_once()

async def test_create_order_network_error(mock_ccxt_exchange_instance):
    """
    Test that create_order returns None when ccxt raises a NetworkError.
    """
    exchange = mock_ccxt_exchange_instance
    
    # Configure the mock to raise an exception
    exchange.exchange.create_order.side_effect = NetworkError("Connection timed out")
    
    # Call the method
    result = await exchange.create_order(
        symbol='BTC-USDT',
        order_type='limit',
        side='buy',
        amount=1,
        price=50000
    )
    
    # Assert the result is None
    assert result is None
    exchange.exchange.create_order.assert_called_once()
