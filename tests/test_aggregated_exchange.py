import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd

# Since AggregatedExchange uses ExchangeFactory, we mock the factory's output.
# This fixture provides fresh mocks for each test function.
@pytest.fixture
def mock_exchange_factory():
    """
    Mocks the ExchangeFactory. It now creates MagicMocks and attaches AsyncMocks
    to each async method, ensuring they are awaitable.
    """
    
    def create_mock_exchange(exchange_id, class_name):
        """Helper to create a properly configured mock exchange."""
        exchange = MagicMock()
        exchange.exchange_id = exchange_id
        exchange.__class__.__name__ = class_name
        
        # Explicitly make async methods awaitable
        exchange.load = AsyncMock()
        exchange.close = AsyncMock()
        exchange.fetch_historical_data = MagicMock()
        exchange.get_current_price = AsyncMock()
        exchange.fetch_candles = AsyncMock()
        return exchange

    mock_primary = create_mock_exchange('binance', 'MockPrimary')
    mock_secondary = create_mock_exchange('okx', 'MockSecondary')

    def factory_side_effect(exchange_id, market_type, **kwargs):
        if exchange_id == 'binance':
            return mock_primary
        if exchange_id == 'okx':
            return mock_secondary
        return create_mock_exchange('other', 'MockOther')

    with patch('exchange.factory.ExchangeFactory.create_exchange', side_effect=factory_side_effect) as factory_mock:
        yield factory_mock, mock_primary, mock_secondary

@pytest.fixture
async def aggregated_exchange(mock_exchange_factory):
    """
    Provides a fully initialized AggregatedExchange instance for tests.
    This correctly uses the async constructor.
    """
    # Import the class under test here, as it might be affected by other patches.
    from exchange.aggregated_exchange import AggregatedExchange
    
    # Use the correct async factory method to create the instance
    agg_ex = await AggregatedExchange.create_async(
        primary_exchange_id='binance',
        secondary_exchange_ids=['okx'],
        sandbox=True
    )
    return agg_ex

class TestAggregatedExchange:

    async def test_initialization(self, aggregated_exchange, mock_exchange_factory):
        """Test that the aggregated exchange initializes correctly."""
        _, mock_primary, mock_secondary = mock_exchange_factory
        
        assert aggregated_exchange.primary_spot_exchange == mock_primary
        assert mock_primary in aggregated_exchange.all_spot_exchanges
        assert mock_secondary in aggregated_exchange.all_spot_exchanges
        assert len(aggregated_exchange.all_spot_exchanges) == 2
        
        # Check that load() was called on each underlying exchange (once for spot, once for swap)
        assert mock_primary.load.await_count == 2
        assert mock_secondary.load.await_count == 2

    async def test_fetch_historical_data_uses_primary_only(self, aggregated_exchange, mock_exchange_factory):
        """Test that fetching historical data only calls the primary exchange."""
        _, mock_primary, mock_secondary = mock_exchange_factory
        
        # This method is synchronous in the current implementation, so no await
        aggregated_exchange.fetch_historical_data('BTC/USDT', '1h', 1)

        # Assert that only the primary exchange was called
        mock_primary.fetch_historical_data.assert_called_once_with('BTC/USDT', '1h', 1)
        # Assert that the secondary exchange was NOT called
        mock_secondary.fetch_historical_data.assert_not_called()

    async def test_get_current_price_aggregates(self, aggregated_exchange, mock_exchange_factory):
        """Test that get_current_price averages the prices from all exchanges."""
        _, mock_primary, mock_secondary = mock_exchange_factory
        
        mock_primary.get_current_price.return_value = 100.0
        mock_secondary.get_current_price.return_value = 102.0

        price = await aggregated_exchange.get_current_price('BTC/USDT')

        assert price == 101.0
        mock_primary.get_current_price.assert_awaited_once_with('BTC/USDT')
        mock_secondary.get_current_price.assert_awaited_once_with('BTC/USDT')

    async def test_get_current_price_with_one_failure(self, aggregated_exchange, mock_exchange_factory):
        """Test that get_current_price works even if one exchange fails."""
        _, mock_primary, mock_secondary = mock_exchange_factory

        mock_primary.get_current_price.return_value = 100.0
        mock_secondary.get_current_price.side_effect = Exception("API Error")

        price = await aggregated_exchange.get_current_price('BTC/USDT')

        assert price == 100.0

    async def test_fetch_candles_aggregation(self, aggregated_exchange, mock_exchange_factory):
        """Test the aggregation logic of fetch_candles."""
        _, mock_primary, mock_secondary = mock_exchange_factory

        df1 = pd.DataFrame({
            'open': [100], 'high': [110], 'low': [99], 'close': [105], 'vol': [1000]
        }, index=pd.to_datetime(['2023-01-01']))
        
        df2 = pd.DataFrame({
            'open': [102], 'high': [108], 'low': [98], 'close': [103], 'vol': [1500]
        }, index=pd.to_datetime(['2023-01-01']))

        mock_primary.fetch_candles.return_value = df1
        mock_secondary.fetch_candles.return_value = df2

        result_df = await aggregated_exchange.fetch_candles('BTC/USDT', '1h', limit=1)

        assert result_df.iloc[0]['open'] == 101      # (100+102)/2
        assert result_df.iloc[0]['high'] == 110      # max(110, 108)
        assert result_df.iloc[0]['low'] == 98       # min(99, 98)
        assert result_df.iloc[0]['close'] == 104     # (105+103)/2
        assert result_df.iloc[0]['vol'] == 2500   # 1000+1500