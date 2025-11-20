import os
from ccxt import async_support
from ccxt.base.errors import (
    BaseError, ExchangeError, NetworkError, RequestTimeout,
    InvalidOrder, OrderNotFound, InsufficientFunds, RateLimitExceeded
)
import pandas as pd
from typing import List, Dict, Any, Optional
from exchange.base import Exchange
import logging
import asyncio

logger = logging.getLogger(__name__)

class CcxtExchange(Exchange):
    def __init__(self, exchange_id: str, api_key: str = None, api_secret: str = None, passphrase: str = None, sandbox: bool = True, market_type: str = 'spot'):
        """
        Initializes the CcxtExchange synchronously. Network I/O is deferred to async methods.
        """
        self.exchange_id = exchange_id.lower()
        self.market_type = market_type
        exchange_class = getattr(async_support, self.exchange_id)
        
        config = {
            'apiKey': api_key,
            'secret': api_secret,
            'timeout': 30000,
            'options': {
                'defaultType': market_type,
            },
        }
        if passphrase:
            config['password'] = passphrase

        https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
        if https_proxy:
            config['httpsProxy'] = https_proxy
            logger.info(f"Applying system HTTPS_PROXY: {https_proxy}")

        self.exchange = exchange_class(config)
        
        if sandbox:
            if 'test' in self.exchange.urls:
                self.exchange.set_sandbox_mode(True)
            else:
                logger.warning(f"{self.exchange.id} does not support sandbox mode. Live mode will be used.")
        
        self.is_loaded = False
        logger.info(f"Initialized CcxtExchange for {self.exchange.id} with market_type='{market_type}' (Sandbox: {sandbox})")

    async def load(self):
        """Asynchronously load markets. Must be called after initialization."""
        if self.is_loaded:
            return
        
        params = {}
        # Specific fix for Bybit v5 API
        if self.exchange_id == 'bybit' and self.market_type == 'swap':
            params['category'] = 'linear'
            logger.info("Bybit swap market detected. Explicitly loading linear markets for v5 API.")

        try:
            await self.exchange.load_markets(params=params)
            self.is_loaded = True
            logger.info(f"Successfully loaded markets for {self.exchange.id} ({self.market_type}).")
        except Exception as e:
            logger.error(f"Failed to load markets for {self.exchange.id}: {e}", exc_info=True)
            # Depending on the strategy, you might want to raise the exception
            # raise e

    async def close(self):
        """Gracefully close the exchange connection."""
        logger.info(f"Closing connection for {self.exchange.id} ({self.market_type})...")
        await self.exchange.close()
        logger.info(f"Connection closed for {self.exchange.id} ({self.market_type}).")

    def _format_symbol(self, symbol: str) -> str:
        return symbol.replace('-', '/')

    def _get_swap_symbol(self, symbol: str) -> str:
        if self.exchange_id == 'okx':
            return symbol + '-SWAP'
        else:
            return symbol.replace('-', '')

    async def fetch_candles(self, symbol: str, timeframe: str, since: Optional[int] = None, limit: Optional[int] = 100) -> pd.DataFrame:
        ccxt_symbol = self._format_symbol(symbol)
        logger.debug(f"Fetching candles for {ccxt_symbol} with timeframe {timeframe}...")
        try:
            ohlcv = await self.exchange.fetch_ohlcv(ccxt_symbol, timeframe.lower(), since, limit)
            if not ohlcv:
                logger.warning(f"No candle data returned for {ccxt_symbol} with timeframe {timeframe}.")
                return pd.DataFrame()

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df.rename(columns={'volume': 'vol'}, inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except BaseError as e:
            logger.error(f"Error fetching candles for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()

    async def fetch_historical_data(self, symbol: str, timeframe: str, years: float) -> pd.DataFrame:
        from datetime import datetime, timedelta

        timeframe_ms = self.exchange.parse_timeframe(timeframe.lower()) * 1000
        since_dt = datetime.utcnow() - timedelta(days=years * 365.25)
        since_ms = int(since_dt.timestamp() * 1000)
        ccxt_symbol = self._format_symbol(symbol) if self.market_type == 'spot' else self._get_swap_symbol(symbol)

        all_candles = []
        logger.info(f"Fetching historical data for {symbol} on {timeframe} since {since_dt.strftime('%Y-%m-%d')}...")

        while True:
            try:
                candles = await self.exchange.fetch_ohlcv(ccxt_symbol, timeframe.lower(), since=since_ms, limit=1000)
                if not candles:
                    break
                all_candles.extend(candles)
                since_ms = candles[-1][0] + timeframe_ms
                await asyncio.sleep(self.exchange.rateLimit / 1000)
            except Exception as e:
                logger.error(f"Error fetching historical data for {symbol}: {e}", exc_info=True)
                break
        
        if not all_candles:
            return pd.DataFrame()

        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df.drop_duplicates(subset=['timestamp'], inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        df.rename(columns={'volume': 'vol'}, inplace=True)
        return df[['open', 'high', 'low', 'close', 'vol']]

    async def get_balance(self, currency: str) -> float:
        try:
            balance = await self.exchange.fetch_balance()
            free_balance = balance.get('free')
            if free_balance is not None:
                return free_balance.get(currency, 0.0)
            
            logger.warning(f"Balance response for {self.exchange.id} is missing 'free' key. Response: {balance}")
            return 0.0
        except BaseError as e:
            logger.error(f"Error fetching balance for {self.exchange.id}: {e}", exc_info=True)
            return 0.0

    async def create_order(self, symbol: str, order_type: str, side: str, amount: float, price: Optional[float] = None) -> Optional[Dict[str, Any]]:
        ccxt_symbol = self._format_symbol(symbol)
        order_details_for_logging = f"Symbol={ccxt_symbol}, Type={order_type}, Side={side}, Amount={amount}, Price={price}"

        try:
            logger.info(f"Attempting to create order on {self.exchange.id}: {order_details_for_logging}")
            return await self.exchange.create_order(ccxt_symbol, order_type, side, amount, price)

        except (RateLimitExceeded, RequestTimeout, NetworkError) as e:
            logger.warning(
                f"Temporary, retryable error creating order on {self.exchange.id}: {type(e).__name__}. "
                f"{order_details_for_logging}. Error: {e}"
            )
            # In a full implementation, this might raise a custom RetryableError
            return None

        except (InvalidOrder, InsufficientFunds) as e:
            logger.critical(
                f"FATAL order logic error on {self.exchange.id}: {type(e).__name__}. "
                f"This is likely a strategy bug and should NOT be retried. "
                f"{order_details_for_logging}. Error: {e}",
                exc_info=True
            )
            return None

        except ExchangeError as e:
            logger.error(
                f"Exchange-level error creating order on {self.exchange.id}: {type(e).__name__}. "
                f"{order_details_for_logging}. Error: {e}",
                exc_info=True
            )
            return None
            
        except BaseError as e:
            logger.error(
                f"Unhandled CCXT BaseError creating order on {self.exchange.id}: {type(e).__name__}. "
                f"{order_details_for_logging}. Error: {e}",
                exc_info=True
            )
            return None

    async def get_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        ccxt_symbol = self._format_symbol(symbol)
        try:
            return await self.exchange.fetch_order(order_id, ccxt_symbol)
        
        except OrderNotFound as e:
            logger.warning(f"Order '{order_id}' not found on {self.exchange.id} for symbol {ccxt_symbol}. It might have been cancelled or never existed. Error: {e}")
            return None

        except (RateLimitExceeded, RequestTimeout, NetworkError) as e:
            logger.warning(f"Temporary error fetching order '{order_id}' on {self.exchange.id}: {type(e).__name__}. Error: {e}")
            # This could also be a retryable case
            return None

        except ExchangeError as e:
            logger.error(f"Exchange-level error fetching order '{order_id}' on {self.exchange.id}: {type(e).__name__}. Error: {e}", exc_info=True)
            return None

        except BaseError as e:
            logger.error(f"Unhandled CCXT BaseError fetching order '{order_id}' on {self.exchange.id}: {type(e).__name__}. Error: {e}", exc_info=True)
            return None

    async def cancel_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        ccxt_symbol = self._format_symbol(symbol)
        try:
            logger.info(f"Attempting to cancel order '{order_id}' on {self.exchange.id} for symbol {ccxt_symbol}")
            return await self.exchange.cancel_order(order_id, ccxt_symbol)

        except OrderNotFound as e:
            # This is often not an error. It means the order was already filled or cancelled.
            logger.info(f"Attempted to cancel order '{order_id}' on {self.exchange.id}, but it was not found (already filled/cancelled?). Error: {e}")
            # Depending on strategy, you might want a specific return value here.
            # For now, we can return a dictionary indicating it's "not found" or "already closed".
            # Let's return None for simplicity, but this is a point for refinement.
            return None

        except (RateLimitExceeded, RequestTimeout, NetworkError) as e:
            logger.warning(f"Temporary, retryable error cancelling order '{order_id}' on {self.exchange.id}: {type(e).__name__}. Error: {e}")
            return None

        except ExchangeError as e:
            logger.error(f"Exchange-level error cancelling order '{order_id}' on {self.exchange.id}: {type(e).__name__}. Error: {e}", exc_info=True)
            return None

        except BaseError as e:
            logger.error(f"Unhandled CCXT BaseError cancelling order '{order_id}' on {self.exchange.id}: {type(e).__name__}. Error: {e}", exc_info=True)
            return None

    async def get_current_price(self, symbol: str) -> float:
        symbol = self._format_symbol(symbol)
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            price = ticker.get('last')
            if price is not None:
                return price
            
            logger.warning(f"Ticker response for {symbol} is missing 'last' key. Response: {ticker}")
            return 0.0
        except BaseError as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}", exc_info=True)
            return 0.0

    async def place_oco_order(self, symbol: str, side: str, amount: float, take_profit_price: float, stop_loss_price: float) -> Dict[str, Any]:
        symbol = self._format_symbol(symbol)
        if not self.exchange.has['createOco']:
            raise NotImplementedError(f"{self.exchange.id} does not support OCO orders.")
        params = {'stopPrice': stop_loss_price, 'stopLimitPrice': stop_loss_price}
        return await self.exchange.create_order(symbol, 'oco', side, amount, take_profit_price, params=params)

    # The following methods are not part of the base Exchange class but are specific to this implementation
    async def fetch_funding_rates(self, symbol: str, timeframe: str, years: Optional[float] = None, since: Optional[int] = None, limit: Optional[int] = None) -> pd.DataFrame:
        if not self.exchange.has['fetchFundingRateHistory']:
            logger.warning(f"Exchange {self.exchange.id} does not support fetching funding rate history.")
            return pd.DataFrame()
        # Implementation would be similar to fetch_historical_data, with await calls
        # For brevity, this is left as an exercise.
        return pd.DataFrame()

    async def fetch_open_interest(self, symbol: str, timeframe: str, years: Optional[float] = None, since: Optional[int] = None, limit: Optional[int] = None) -> pd.DataFrame:
        if not self.exchange.has['fetchOpenInterestHistory']:
            logger.warning(f"Exchange {self.exchange.id} does not support fetching open interest history.")
            return pd.DataFrame()
        # Implementation would be similar to fetch_historical_data, with await calls
        # For brevity, this is left as an exercise.
        return pd.DataFrame()