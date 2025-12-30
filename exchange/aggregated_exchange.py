
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
    async def create_async(cls, primary_exchange_id: str, secondary_exchange_ids: List[str], api_key: str = None, api_secret: str = None, passphrase: str = None, sandbox: bool = True, dune_api_key: str = None, mock: bool = False):
        # Normalize exchange IDs to lowercase to ensure consistency
        primary_exchange_id = primary_exchange_id.lower()
        secondary_exchange_ids = [eid.lower() for eid in secondary_exchange_ids]

        logger.info(f"Asynchronously creating AggregatedExchange for primary: {primary_exchange_id} and secondaries: {secondary_exchange_ids} (Mock: {mock})")
        
        all_exchange_ids = list(set([primary_exchange_id] + secondary_exchange_ids))
        
        # Create coroutines for all exchange creations
        spot_creation_tasks = [ExchangeFactory.create_exchange(ex_id, market_type='spot', api_key=None, api_secret=None, passphrase=None, sandbox=sandbox, mock=mock) for ex_id in all_exchange_ids]
        swap_creation_tasks = [ExchangeFactory.create_exchange(ex_id, market_type='swap', api_key=None, api_secret=None, passphrase=None, sandbox=sandbox, mock=mock) for ex_id in all_exchange_ids]

        # Await them concurrently
        spot_exchanges = await asyncio.gather(*spot_creation_tasks)
        swap_exchanges = await asyncio.gather(*swap_creation_tasks)

        # Asynchronously load markets for all created exchanges
        all_created_exchanges = spot_exchanges + swap_exchanges
        load_tasks = [exc.load() for exc in all_created_exchanges if hasattr(exc, 'load')]
        logger.info(f"Loading markets for {len(load_tasks)} exchange instances...")
        await asyncio.gather(*load_tasks, return_exceptions=True)
        logger.info("Market loading complete.")

        # Initialize Dune Client
        dune_client = None
        try:
            if dune_api_key:
                dune_client = DuneAnalyticsClient(api_key=dune_api_key)
                logger.info("Successfully initialized Dune Analytics client.")
        except Exception as e:
            logger.warning(f"Failed to initialize Dune Analytics client: {e}. On-chain data will be unavailable.")

        primary_spot_exchange = next((ex for ex in spot_exchanges if ex.exchange_id == primary_exchange_id), None)

        if not primary_spot_exchange:
            # Debug info to help diagnose mismatch
            available_ids = [ex.exchange_id for ex in spot_exchanges]
            logger.error(f"Primary exchange '{primary_exchange_id}' not found in available spot exchanges: {available_ids}")
            raise ConnectionError(f"Failed to initialize primary SPOT exchange {primary_exchange_id}")

        return cls(primary_spot_exchange, spot_exchanges, swap_exchanges, dune_client)

    async def close(self):
        """
        Closes all underlying exchange connections.
        """
        logger.info("Closing all exchange connections...")
        all_exchanges = list(set(self.all_spot_exchanges + self.all_swap_exchanges))
        close_tasks = [exc.close() for exc in all_exchanges if hasattr(exc, 'close')]
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        logger.info("All exchange connections have been closed.")

    async def fetch_candles_async(self, exchange: Exchange, symbol: str, timeframe: str, since: Optional[int] = None, limit: Optional[int] = None) -> Optional[tuple[str, pd.DataFrame]]:
        """Asynchronously fetch candles from a single exchange."""
        exchange_name = exchange.__class__.__name__
        try:
            # The underlying fetch_candles is now async
            df = await exchange.fetch_candles(symbol, timeframe, since, limit)
            logger.info(f"OK: Successfully fetched {len(df)} candles from {exchange_name} for {symbol}")
            return exchange_name, df
        except Exception as e:
            logger.warning(f"FAIL: Failed to fetch candles from {exchange_name}: {e}")
            return exchange_name, None

    async def fetch_candles(self, symbol: str, timeframe: str, since: Optional[int] = None, limit: Optional[int] = 100) -> pd.DataFrame:
        """
        Asynchronously fetch recent candle data by aggregating results from all available SPOT exchanges.
        """
        exchange_names = [exc.__class__.__name__ for exc in self.all_spot_exchanges]
        logger.info(f"Querying {len(self.all_spot_exchanges)} SPOT sources for recent candles: {exchange_names}")
        
        tasks = [self.fetch_candles_async(exc, symbol, timeframe, since, limit) for exc in self.all_spot_exchanges]
        results = await asyncio.gather(*tasks)
        
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

    async def fetch_historical_data(self, symbol: str, timeframe: str, years: float) -> pd.DataFrame:
        """
        Fetch long-term historical data for backtesting/training from the primary SPOT exchange.
        """
        logger.info(f"Fetching historical data for {symbol} exclusively from primary SPOT source: {self.primary_spot_exchange.__class__.__name__}")
        return await self.primary_spot_exchange.fetch_historical_data(symbol, timeframe, years)

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
            'volume': grouped['volume'].sum()
        })
        logger.debug(f"Aggregated {len(dfs)} dataframes into one.")
        return agg_df

    async def fetch_funding_rates_async(self, exchange: Exchange, symbol: str, timeframe: str, since: Optional[int] = None, limit: Optional[int] = None) -> Optional[tuple[str, pd.DataFrame]]:
        exchange_name = f"{exchange.exchange_id}-{exchange.market_type}"
        try:
            df = await exchange.fetch_funding_rates(symbol, timeframe, since=since, limit=limit)
            logger.info(f"OK: Successfully fetched {len(df)} funding rates from {exchange_name} for {symbol}")
            return exchange_name, df
        except AttributeError:
            logger.warning(f"FAIL: Exchange {exchange_name} does not support fetch_funding_rates.")
            return exchange_name, None
        except Exception as e:
            logger.warning(f"FAIL: Failed to fetch funding rates from {exchange_name}: {e}")
            return exchange_name, None

    async def fetch_funding_rates(self, symbol: str, timeframe: str, years: Optional[float] = None, since: Optional[int] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch recent funding rates by aggregating results from all available SWAP exchanges.
        """
        exchange_names = [f"{exc.exchange_id}-{exc.market_type}" for exc in self.all_swap_exchanges]
        logger.info(f"Querying {len(self.all_swap_exchanges)} SWAP sources for recent funding rates: {exchange_names}")
        
        tasks = [self.fetch_funding_rates_async(exc, symbol, timeframe, since=since, limit=limit) for exc in self.all_swap_exchanges]
        results = await asyncio.gather(*tasks)
        
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
            df = await exchange.fetch_open_interest(symbol, timeframe, since=since, limit=limit)
            logger.info(f"OK: Successfully fetched {len(df)} open interest data points from {exchange_name} for {symbol}")
            return exchange_name, df
        except AttributeError:
            logger.warning(f"FAIL: Exchange {exchange_name} does not support fetch_open_interest.")
            return exchange_name, None
        except Exception as e:
            logger.warning(f"FAIL: Failed to fetch open interest from {exchange_name}: {e}")
            return exchange_name, None

    async def fetch_open_interest(self, symbol: str, timeframe: str, years: Optional[float] = None, since: Optional[int] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch recent open interest by aggregating results from all available SWAP exchanges.
        If limit is None, all available data will be fetched.
        """
        exchange_names = [f"{exc.exchange_id}-{exc.market_type}" for exc in self.all_swap_exchanges]
        logger.info(f"Querying {len(self.all_swap_exchanges)} SWAP sources for recent open interest: {exchange_names}")
        
        tasks = [self.fetch_open_interest_async(exc, symbol, timeframe, since=since, limit=limit) for exc in self.all_swap_exchanges]
        results = await asyncio.gather(*tasks)
        
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

    async def fetch_long_short_ratio_async(self, exchange: Exchange, symbol: str, timeframe: str, limit: Optional[int] = None) -> Optional[tuple[str, pd.DataFrame]]:
        exchange_name = f"{exchange.exchange_id}-{exchange.market_type}"
        try:
            # Check if method exists (it's custom, not standard CCXT)
            if hasattr(exchange, 'fetch_long_short_ratio'):
                df = await exchange.fetch_long_short_ratio(symbol, timeframe, limit=limit)
                logger.info(f"OK: Successfully fetched {len(df)} long/short ratio records from {exchange_name} for {symbol}")
                return exchange_name, df
            else:
                logger.warning(f"FAIL: Exchange {exchange_name} does not implement fetch_long_short_ratio.")
                return exchange_name, None
        except Exception as e:
            logger.warning(f"FAIL: Failed to fetch long/short ratio from {exchange_name}: {e}")
            return exchange_name, None

    async def fetch_long_short_ratio(self, symbol: str, timeframe: str, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch recent Top Trader Long/Short Ratio by aggregating results from all available SWAP exchanges.
        """
        exchange_names = [f"{exc.exchange_id}-{exc.market_type}" for exc in self.all_swap_exchanges]
        logger.info(f"Querying {len(self.all_swap_exchanges)} SWAP sources for long/short ratio: {exchange_names}")
        
        tasks = [self.fetch_long_short_ratio_async(exc, symbol, timeframe, limit=limit) for exc in self.all_swap_exchanges]
        results = await asyncio.gather(*tasks)
        
        all_dfs = []
        successful_sources = []
        for exchange_name, df in results:
            if df is not None and not df.empty:
                all_dfs.append(df)
                successful_sources.append(exchange_name)

        if not all_dfs:
            logger.info("Fetched 0 long/short ratio records (likely Sandbox limitation).")
            return pd.DataFrame()

        logger.info(f"Fusing long/short ratio from {len(successful_sources)} sources: {successful_sources}")
        return self.aggregate_long_short_ratio(all_dfs)

    def aggregate_long_short_ratio(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        if not dfs:
            return pd.DataFrame()
        combined_df = pd.concat(dfs)
        grouped = combined_df.groupby(combined_df.index)
        # Average the ratios
        agg_df = pd.DataFrame({'toptrader_long_short_ratio': grouped['toptrader_long_short_ratio'].mean()})
        logger.debug(f"Aggregated {len(dfs)} long/short ratio dataframes into one.")
        return agg_df

    def aggregate_open_interest(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        if not dfs:
            return pd.DataFrame()
        combined_df = pd.concat(dfs)
        grouped = combined_df.groupby(combined_df.index)
        
        agg_rules = {}
        if 'open_interest' in combined_df.columns:
            agg_rules['open_interest'] = 'sum'
        if 'open_interest_value' in combined_df.columns:
            agg_rules['open_interest_value'] = 'sum'
            
        agg_df = grouped.agg(agg_rules)
        logger.debug(f"Aggregated {len(dfs)} open interest dataframes into one.")
        return agg_df

    async def fetch_global_long_short_ratio_async(self, exchange: Exchange, symbol: str, timeframe: str, limit: int = 100) -> Optional[tuple[str, pd.DataFrame]]:
        exchange_name = f"{exchange.exchange_id}-{exchange.market_type}"
        try:
            if hasattr(exchange, 'fetch_global_long_short_ratio'):
                df = await exchange.fetch_global_long_short_ratio(symbol, timeframe, limit=limit)
                if df is not None and not df.empty:
                    logger.info(f"OK: Successfully fetched {len(df)} global L/S records from {exchange_name}")
                    return exchange_name, df
            return exchange_name, None
        except Exception as e:
            logger.warning(f"FAIL: Failed to fetch global L/S from {exchange_name}: {e}")
            return exchange_name, None

    async def fetch_global_long_short_ratio(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Fetching Global L/S Ratio from all swap exchanges."""
        tasks = [self.fetch_global_long_short_ratio_async(exc, symbol, timeframe, limit) for exc in self.all_swap_exchanges]
        results = await asyncio.gather(*tasks)
        
        all_dfs = []
        successful_sources = []
        for exchange_name, df in results:
            if df is not None and not df.empty:
                all_dfs.append(df)
                successful_sources.append(exchange_name)
        
        if not all_dfs:
            logger.warning("Fetched 0 Global L/S records (likely Sandbox limitation).")
            return pd.DataFrame()
            
        logger.info(f"Fusing Global L/S from {len(successful_sources)} sources: {successful_sources}")
        return self.aggregate_global_long_short_ratio(all_dfs)

    def aggregate_global_long_short_ratio(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        if not dfs: return pd.DataFrame()
        combined_df = pd.concat(dfs)
        grouped = combined_df.groupby(combined_df.index)
        agg_df = pd.DataFrame({'long_short_ratio': grouped['long_short_ratio'].mean()})
        return agg_df

    async def fetch_taker_buy_sell_vol_ratio_async(self, exchange: Exchange, symbol: str, timeframe: str, limit: int = 100) -> Optional[tuple[str, pd.DataFrame]]:
        exchange_name = f"{exchange.exchange_id}-{exchange.market_type}"
        try:
            if hasattr(exchange, 'fetch_taker_buy_sell_vol_ratio'):
                df = await exchange.fetch_taker_buy_sell_vol_ratio(symbol, timeframe, limit=limit)
                if df is not None and not df.empty:
                    logger.info(f"OK: Successfully fetched {len(df)} taker volume records from {exchange_name}")
                    return exchange_name, df
            return exchange_name, None
        except Exception as e:
            logger.warning(f"FAIL: Failed to fetch taker volume from {exchange_name}: {e}")
            return exchange_name, None

    async def fetch_taker_buy_sell_vol_ratio(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Fetching Taker Buy/Sell Ratio from all swap exchanges."""
        tasks = [self.fetch_taker_buy_sell_vol_ratio_async(exc, symbol, timeframe, limit) for exc in self.all_swap_exchanges]
        results = await asyncio.gather(*tasks)
        
        all_dfs = []
        successful_sources = []
        for exchange_name, df in results:
            if df is not None and not df.empty:
                all_dfs.append(df)
                successful_sources.append(exchange_name)
        
        if not all_dfs:
            logger.warning("Fetched 0 Taker Buy/Sell records (likely Sandbox limitation).")
            return pd.DataFrame()
            
        logger.info(f"Fusing Taker Buy/Sell from {len(successful_sources)} sources: {successful_sources}")
        return self.aggregate_taker_buy_sell_vol_ratio(all_dfs)

    def aggregate_taker_buy_sell_vol_ratio(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        if not dfs: return pd.DataFrame()
        combined_df = pd.concat(dfs)
        grouped = combined_df.groupby(combined_df.index)
        agg_df = pd.DataFrame({'taker_long_short_vol_ratio': grouped['taker_long_short_vol_ratio'].mean()})
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

    async def get_balance(self, currency: str) -> float:
        try:
            return await self.primary_spot_exchange.get_balance(currency)
        except Exception as e:
            logger.warning(f"Failed to fetch balance from primary exchange ({self.primary_spot_exchange.exchange_id}): {e}")
            # If primary fails (e.g. it's binance without keys), try secondaries
            for exchange in self.all_spot_exchanges:
                if exchange.exchange_id == self.primary_spot_exchange.exchange_id:
                    continue
                try:
                    logger.info(f"Attempting to fetch balance from secondary exchange: {exchange.exchange_id}")
                    return await exchange.get_balance(currency)
                except Exception as inner_e:
                     logger.debug(f"Could not fetch balance from {exchange.exchange_id}: {inner_e}")
            
            # If all fail, return 0.0 or re-raise. returning 0.0 is safer for stability but might be misleading.
            # But since this is likely a misconfiguration (Binance primary but no keys), returning 0 is "safe" 
            # as it prevents trading but keeps bot alive.
            logger.error("Could not fetch balance from ANY exchange.")
            return 0.0

    async def create_order(self, symbol: str, order_type: str, side: str, amount: float, price: Optional[float] = None) -> Dict[str, Any]:
        return await self.primary_spot_exchange.create_order(symbol, order_type, side, amount, price)

    async def get_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        return await self.primary_spot_exchange.get_order(order_id, symbol)

    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        return await self.primary_spot_exchange.cancel_order(order_id, symbol)

    async def get_current_price(self, symbol: str) -> float:
        async def _fetch_one_price(exchange):
            try:
                # Assuming the underlying exchange method is now async
                return await exchange.get_current_price(symbol)
            except Exception as e:
                logger.warning(f"Could not fetch price from {exchange.__class__.__name__}: {e}")
                return None

        tasks = [_fetch_one_price(exc) for exc in self.all_spot_exchanges]
        prices = await asyncio.gather(*tasks)
        valid_prices = [p for p in prices if p is not None]

        if not valid_prices:
            raise ConnectionError("Failed to fetch current price from all exchanges.")
            
        return sum(valid_prices) / len(valid_prices)

    async def place_oco_order(self, symbol: str, side: str, amount: float, take_profit_price: float, stop_loss_price: float) -> Dict[str, Any]:
        # Smart Routing: Check if primary is healthy
        if self.is_primary_healthy():
            return await self.primary_spot_exchange.place_oco_order(symbol, side, amount, take_profit_price, stop_loss_price)
        else:
            logger.warning("Primary exchange is unhealthy! Attempting to route OCO order to secondary exchange...")
            # Try to find a secondary exchange that supports OCO
            for exchange in self.all_spot_exchanges:
                if exchange.exchange_id == self.primary_spot_exchange.exchange_id:
                    continue
                try:
                    return await exchange.place_oco_order(symbol, side, amount, take_profit_price, stop_loss_price)
                except Exception as e:
                    logger.warning(f"Failed to route OCO to {exchange.exchange_id}: {e}")
            
            raise ConnectionError("Primary exchange unhealthy and no suitable secondary exchange found for OCO order.")

    def is_primary_healthy(self) -> bool:
        """Checks if the primary exchange is healthy (circuit breaker not tripped)."""
        if hasattr(self.primary_spot_exchange, 'check_circuit_breaker'):
            return self.primary_spot_exchange.check_circuit_breaker()
        return True

    async def hedge_position(self, symbol: str, quantity: float, side: str) -> Dict[str, Any]:
        """
        Places a hedging order on a secondary exchange.
        This is typically called when the primary exchange fails or is unstable.
        
        Args:
            symbol: The symbol to hedge (e.g., 'BTC/USDT')
            quantity: The quantity to hedge
            side: The side of the ORIGINAL position (hedge will be opposite)
        """
        hedge_side = 'sell' if side == 'buy' else 'buy'
        logger.warning(f"INITIATING HEDGE: Attempting to {hedge_side} {quantity} {symbol} on secondary exchanges.")
        
        for exchange in self.all_spot_exchanges:
            # Skip primary exchange
            if exchange.exchange_id == self.primary_spot_exchange.exchange_id:
                continue
                
            try:
                # Check if secondary is healthy (optional but good practice)
                if hasattr(exchange, 'check_circuit_breaker') and not exchange.check_circuit_breaker():
                    logger.warning(f"Skipping secondary exchange {exchange.exchange_id} (Circuit Breaker Active).")
                    continue

                logger.info(f"Placing hedge order on {exchange.exchange_id}...")
                # Market order for immediate execution
                result = await exchange.create_order(symbol, 'market', hedge_side, quantity)
                if result:
                    logger.info(f"HEDGE SUCCESSFUL: Executed on {exchange.exchange_id}. Result: {result}")
                    return result
            except Exception as e:
                logger.error(f"Hedge attempt failed on {exchange.exchange_id}: {e}")
        
        logger.critical("HEDGE FAILED: Could not execute hedge order on ANY secondary exchange!")
        return None
