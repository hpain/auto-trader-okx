
import asyncio
from typing import List, Dict, Any, Optional
import pandas as pd

from data.dune_client import DuneAnalyticsClient
from exchange.base import Exchange
from exchange.factory import ExchangeFactory
import logging

logger = logging.getLogger(__name__)

class AggregatedExchange(Exchange):
    """
    Aggregates data from multiple exchanges and on-chain sources.
    """

    def __init__(self, primary_spot_exchange: Exchange, all_spot_exchanges: List[Exchange], all_swap_exchanges: List[Exchange], dune_client: Optional[DuneAnalyticsClient] = None):
        self.primary_spot_exchange = primary_spot_exchange
        self.all_spot_exchanges = all_spot_exchanges
        self.all_swap_exchanges = all_swap_exchanges
        self.dune_client = dune_client
        logger.info(f"AggregatedExchange initialized with {len(all_spot_exchanges)} spot and {len(all_swap_exchanges)} swap exchanges.")
        if self.dune_client:
            logger.info("Dune Analytics client also initialized.")

    @classmethod
    async def create_async(cls, primary_exchange_id: str, secondary_exchange_ids: List[str], api_key: str = None, api_secret: str = None, passphrase: str = None, sandbox: bool = True, dune_api_key: str = None):
        logger.info(f"Asynchronously creating AggregatedExchange for primary: {primary_exchange_id} and secondaries: {secondary_exchange_ids}")
        
        all_exchange_ids = list(set([primary_exchange_id] + secondary_exchange_ids))
        
        spot_tasks = []
        swap_tasks = []
        for exchange_id in all_exchange_ids:
            # Create spot instance task
            spot_tasks.append(ExchangeFactory.create_exchange(
                exchange_id, market_type='spot', api_key=api_key, api_secret=api_secret, passphrase=passphrase, sandbox=sandbox
            ))
            # Create swap instance task
            swap_tasks.append(ExchangeFactory.create_exchange(
                exchange_id, market_type='swap', api_key=api_key, api_secret=api_secret, passphrase=passphrase, sandbox=sandbox
            ))

        # Run all creation tasks concurrently
        created_spot_exchanges = await asyncio.gather(*spot_tasks, return_exceptions=True)
        created_swap_exchanges = await asyncio.gather(*swap_tasks, return_exceptions=True)

        # Initialize Dune Client
        dune_client = None
        try:
            dune_client = DuneAnalyticsClient(api_key=dune_api_key)
            logger.info("Successfully initialized Dune Analytics client.")
        except Exception as e:
            logger.warning(f"Failed to initialize Dune Analytics client: {e}. On-chain data will be unavailable.")

        # Process results
        primary_spot_exchange = None
        all_spot_exchanges = []
        all_swap_exchanges = []

        for i, exchange_id in enumerate(all_exchange_ids):
            # Process spot results
            spot_result = created_spot_exchanges[i]
            if isinstance(spot_result, Exception):
                logger.error(f"Failed to initialize SPOT exchange {exchange_id}: {spot_result}")
            else:
                all_spot_exchanges.append(spot_result)
                if exchange_id == primary_exchange_id:
                    primary_spot_exchange = spot_result
            
            # Process swap results
            swap_result = created_swap_exchanges[i]
            if isinstance(swap_result, Exception):
                logger.error(f"Failed to initialize SWAP exchange {exchange_id}: {swap_result}")
            else:
                all_swap_exchanges.append(swap_result)

        if not primary_spot_exchange:
            raise ConnectionError(f"Failed to initialize primary SPOT exchange {primary_exchange_id}")

        return cls(primary_spot_exchange, all_spot_exchanges, all_swap_exchanges, dune_client)

    async def fetch_candles_async(self, exchange: Exchange, symbol: str, timeframe: str, since: Optional[int] = None, limit: Optional[int] = None) -> Optional[tuple[str, pd.DataFrame]]:
        """Asynchronously fetch candles from a single exchange."""
        exchange_name = exchange.__class__.__name__
        try:
            df = await asyncio.to_thread(exchange.fetch_candles, symbol, timeframe, since, limit)
            logger.info(f"OK: Successfully fetched {len(df)} candles from {exchange_name} for {symbol}")
            return exchange_name, df
        except Exception as e:
            logger.warning(f"FAIL: Failed to fetch candles from {exchange_name}: {e}")
            return exchange_name, None

    def fetch_candles(self, symbol: str, timeframe: str, since: Optional[int] = None, limit: Optional[int] = 100) -> pd.DataFrame:
        """
        Fetch recent candle data by aggregating results from all available SPOT exchanges.
        """
        exchange_names = [exc.__class__.__name__ for exc in self.all_spot_exchanges]
        logger.info(f"Querying {len(self.all_spot_exchanges)} SPOT sources for recent candles: {exchange_names}")
        
        async def _fetch_all():
            tasks = [self.fetch_candles_async(exc, symbol, timeframe, since, limit) for exc in self.all_spot_exchanges]
            results = await asyncio.gather(*tasks)
            return results

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        results = loop.run_until_complete(_fetch_all())
        
        all_dfs = []
        successful_sources = []
        for exchange_name, df in results:
            if df is not None and not df.empty:
                all_dfs.append(df)
                successful_sources.append(exchange_name)

        if not all_dfs:
            logger.error("Failed to fetch candle data from ALL sources.")
            return pd.DataFrame()

        logger.info(f"Fusing data from {len(successful_sources)} sources: {successful_sources}")
        return self.aggregate_candle_data(all_dfs)

    def fetch_historical_data(self, symbol: str, timeframe: str, years: float) -> pd.DataFrame:
        """
        Fetch long-term historical data for backtesting/training from the primary SPOT exchange.
        """
        logger.info(f"Fetching historical data for {symbol} exclusively from primary SPOT source: {self.primary_spot_exchange.__class__.__name__}")
        return self.primary_spot_exchange.fetch_historical_data(symbol, timeframe, years)

    def aggregate_candle_data(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        if not dfs:
            return pd.DataFrame()
        combined_df = pd.concat(dfs)
        grouped = combined_df.groupby(combined_df.index)
        agg_df = pd.DataFrame({
            'open': grouped['open'].mean(),
            'high': grouped['high'].max(),
            'low': grouped['low'].min(),
            'close': grouped['close'].mean(),
            'vol': grouped['vol'].sum()
        })
        logger.debug(f"Aggregated {len(dfs)} dataframes into one.")
        return agg_df

    async def fetch_funding_rates_async(self, exchange: Exchange, symbol: str, timeframe: str, since: Optional[int] = None, limit: Optional[int] = None) -> Optional[tuple[str, pd.DataFrame]]:
        exchange_name = f"{exchange.exchange_id}-{exchange.market_type}"
        try:
            df = await asyncio.to_thread(exchange.fetch_funding_rates, symbol, timeframe, since, limit)
            logger.info(f"OK: Successfully fetched {len(df)} funding rates from {exchange_name} for {symbol}")
            return exchange_name, df
        except AttributeError:
            logger.warning(f"FAIL: Exchange {exchange_name} does not support fetch_funding_rates.")
            return exchange_name, None
        except Exception as e:
            logger.warning(f"FAIL: Failed to fetch funding rates from {exchange_name}: {e}")
            return exchange_name, None

    def fetch_funding_rates(self, symbol: str, timeframe: str, since: Optional[int] = None, limit: Optional[int] = 100) -> pd.DataFrame:
        """
        Fetch recent funding rates by aggregating results from all available SWAP exchanges.
        """
        exchange_names = [f"{exc.exchange_id}-{exc.market_type}" for exc in self.all_swap_exchanges]
        logger.info(f"Querying {len(self.all_swap_exchanges)} SWAP sources for recent funding rates: {exchange_names}")
        
        async def _fetch_all():
            tasks = [self.fetch_funding_rates_async(exc, symbol, timeframe, since, limit) for exc in self.all_swap_exchanges]
            results = await asyncio.gather(*tasks)
            return results

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        results = loop.run_until_complete(_fetch_all())
        
        all_dfs = []
        successful_sources = []
        for exchange_name, df in results:
            if df is not None and not df.empty:
                all_dfs.append(df)
                successful_sources.append(exchange_name)

        if not all_dfs:
            logger.warning("Failed to fetch funding rates from ALL sources.")
            return pd.DataFrame()

        logger.info(f"Fusing funding rates from {len(successful_sources)} sources: {successful_sources}")
        return self.aggregate_funding_rates(all_dfs)

    def aggregate_funding_rates(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        if not dfs:
            return pd.DataFrame()
        combined_df = pd.concat(dfs)
        grouped = combined_df.groupby(combined_df.index)
        agg_df = pd.DataFrame({'funding_rate': grouped['funding_rate'].mean()})
        logger.debug(f"Aggregated {len(dfs)} funding rate dataframes into one.")
        return agg_df

    async def fetch_open_interest_async(self, exchange: Exchange, symbol: str, timeframe: str, since: Optional[int] = None, limit: Optional[int] = None) -> Optional[tuple[str, pd.DataFrame]]:
        exchange_name = f"{exchange.exchange_id}-{exchange.market_type}"
        try:
            df = await asyncio.to_thread(exchange.fetch_open_interest, symbol, timeframe, since, limit)
            logger.info(f"OK: Successfully fetched {len(df)} open interest data points from {exchange_name} for {symbol}")
            return exchange_name, df
        except AttributeError:
            logger.warning(f"FAIL: Exchange {exchange_name} does not support fetch_open_interest.")
            return exchange_name, None
        except Exception as e:
            logger.warning(f"FAIL: Failed to fetch open interest from {exchange_name}: {e}")
            return exchange_name, None

    def fetch_open_interest(self, symbol: str, timeframe: str, since: Optional[int] = None, limit: Optional[int] = 100) -> pd.DataFrame:
        """
        Fetch recent open interest by aggregating results from all available SWAP exchanges.
        """
        exchange_names = [f"{exc.exchange_id}-{exc.market_type}" for exc in self.all_swap_exchanges]
        logger.info(f"Querying {len(self.all_swap_exchanges)} SWAP sources for recent open interest: {exchange_names}")
        
        async def _fetch_all():
            tasks = [self.fetch_open_interest_async(exc, symbol, timeframe, since, limit) for exc in self.all_swap_exchanges]
            results = await asyncio.gather(*tasks)
            return results

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        results = loop.run_until_complete(_fetch_all())
        
        all_dfs = []
        successful_sources = []
        for exchange_name, df in results:
            if df is not None and not df.empty:
                all_dfs.append(df)
                successful_sources.append(exchange_name)

        if not all_dfs:
            logger.warning("Failed to fetch open interest from ALL sources.")
            return pd.DataFrame()

        logger.info(f"Fusing open interest from {len(successful_sources)} sources: {successful_sources}")
        return self.aggregate_open_interest(all_dfs)

    def aggregate_open_interest(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        if not dfs:
            return pd.DataFrame()
        combined_df = pd.concat(dfs)
        grouped = combined_df.groupby(combined_df.index)
        agg_df = pd.DataFrame({'open_interest': grouped['open_interest'].sum()})
        logger.debug(f"Aggregated {len(dfs)} open interest dataframes into one.")
        return agg_df

    def fetch_onchain_data(self, query_id: int, params: dict = None) -> pd.DataFrame:
        """
        Fetches on-chain data from Dune Analytics using a specific query ID.
        """
        if not self.dune_client:
            logger.error("Dune Analytics client is not initialized. Cannot fetch on-chain data.")
            return pd.DataFrame()
        
        logger.info(f"Fetching on-chain data from Dune Analytics (Query ID: {query_id})")
        try:
            # The get_query_results method in the client is synchronous.
            # For a fully async application, consider running it in a thread.
            df = self.dune_client.get_query_results(query_id, params)
            if not df.empty:
                logger.info(f"Successfully fetched {len(df)} rows from Dune Analytics.")
            else:
                logger.warning(f"Dune Analytics query {query_id} returned no data.")
            return df
        except Exception as e:
            logger.error(f"An error occurred while fetching data from Dune Analytics: {e}")
            return pd.DataFrame()

    # ==========================================================================
    # The following methods delegate to the primary SPOT exchange for now.
    # ==========================================================================

    def get_balance(self, currency: str) -> float:
        return self.primary_spot_exchange.get_balance(currency)

    def create_order(self, symbol: str, order_type: str, side: str, amount: float, price: Optional[float] = None) -> Dict[str, Any]:
        return self.primary_spot_exchange.create_order(symbol, order_type, side, amount, price)

    def get_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        return self.primary_spot_exchange.get_order(order_id, symbol)

    def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        return self.primary_spot_exchange.cancel_order(order_id, symbol)

    def get_current_price(self, symbol: str) -> float:
        prices = []
        for exchange in self.all_spot_exchanges:
            try:
                prices.append(exchange.get_current_price(symbol))
            except Exception as e:
                logger.warning(f"Could not fetch price from {exchange.__class__.__name__}: {e}")
        
        if not prices:
            raise ConnectionError("Failed to fetch current price from all exchanges.")
            
        return sum(prices) / len(prices)

    def place_oco_order(self, symbol: str, side: str, amount: float, take_profit_price: float, stop_loss_price: float) -> Dict[str, Any]:
        return self.primary_spot_exchange.place_oco_order(symbol, side, amount, take_profit_price, stop_loss_price)
