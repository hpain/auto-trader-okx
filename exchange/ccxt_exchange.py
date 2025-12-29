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
from datetime import datetime, timedelta, timezone

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
            'aiohttp_trust_env': True, # Important: Tell aiohttp to respect env vars
            'options': {
                'defaultType': market_type,
            },
        }
        if passphrase:
            config['password'] = passphrase

        # Explicitly read proxy from env and inject into config
        # This ensures CCXT uses it even if aiohttp auto-discovery fails
        https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
        if https_proxy:
            config['httpsProxy'] = https_proxy
            # config['httpProxy'] = https_proxy # Set both for good measure
            logger.info(f"Applying system HTTPS_PROXY: {https_proxy}")

        self.exchange = exchange_class(config)

        
        if sandbox:
            if 'test' in self.exchange.urls:
                self.exchange.set_sandbox_mode(True)
            else:
                logger.warning(f"{self.exchange.id} does not support sandbox mode. Live mode will be used.")
        
        self.is_loaded = False
        
        # Circuit Breaker State
        self.error_count = 0
        self.last_error_time = None
        self.circuit_breaker_triggered = False
        self.circuit_breaker_reset_time = None
        
        # Circuit Breaker Config
        self.cb_error_threshold = 5  # 5 errors
        self.cb_time_window = 60     # within 60 seconds
        self.cb_cooldown_time = 300  # pause for 5 minutes
        
        logger.info(f"Initialized CcxtExchange for {self.exchange.id} with market_type='{market_type}' (Sandbox: {sandbox})")

    def check_circuit_breaker(self) -> bool:
        """
        Checks if the circuit breaker is active.
        Returns True if trading is allowed (circuit closed), False if broken (circuit open).
        """
        now = datetime.utcnow()
        
        # If triggered, check if cooldown has passed
        if self.circuit_breaker_triggered:
            if now >= self.circuit_breaker_reset_time:
                self.circuit_breaker_triggered = False
                self.error_count = 0
                self.last_error_time = None
                logger.info(f"Circuit breaker cooldown ended. Resuming operations.")
                return True
            else:
                remaining = (self.circuit_breaker_reset_time - now).total_seconds()
                logger.warning(f"Circuit breaker active. Operations paused for {remaining:.0f}s.")
                return False
        
        # Check if error count needs reset (sliding window-ish)
        if self.last_error_time and (now - self.last_error_time).total_seconds() > self.cb_time_window:
            if self.error_count > 0:
                logger.debug("Circuit breaker error window passed. Resetting error count.")
                self.error_count = 0
                
        return True

    def _record_error(self):
        """Records an API error and triggers circuit breaker if threshold reached."""
        now = datetime.utcnow()
        self.last_error_time = now
        self.error_count += 1
        
        if self.error_count >= self.cb_error_threshold:
            self.circuit_breaker_triggered = True
            self.circuit_breaker_reset_time = now + timedelta(seconds=self.cb_cooldown_time)
            logger.critical(f"CIRCUIT BREAKER TRIGGERED! {self.error_count} errors in window. Pausing for {self.cb_cooldown_time}s.")

    async def load(self):
        """Asynchronously load markets. Must be called after initialization."""
        if self.is_loaded:
            return
        
        params = {}
        # Specific fix for Bybit v5 API
        if self.exchange_id == 'bybit' and self.market_type == 'swap':
            params['category'] = 'linear'
            logger.info("Bybit swap market detected. Explicitly loading linear markets for v5 API.")

        retry_count = 3
        for attempt in range(1, retry_count + 1):
            try:
                await self.exchange.load_markets(params=params)
                self.is_loaded = True
                logger.info(f"Successfully loaded markets for {self.exchange.id} ({self.market_type}).")
                return
            except (RequestTimeout, NetworkError) as e:
                if attempt < retry_count:
                    wait_time = 2 * attempt
                    logger.warning(f"Timeout/Network error loading markets for {self.exchange.id} (Attempt {attempt}/{retry_count}): {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.warning(f"Final failure loading markets for {self.exchange.id} after {retry_count} attempts: {e}. Proceeding without full market data.")
            except Exception as e:
                logger.error(f"Failed to load markets for {self.exchange.id}: {e}", exc_info=True)
                return

    async def close(self):
        """Gracefully close the exchange connection."""
        logger.info(f"Closing connection for {self.exchange.id} ({self.market_type})...")
        await self.exchange.close()
        logger.info(f"Connection closed for {self.exchange.id} ({self.market_type}).")

    def _format_symbol(self, symbol: str) -> str:
        return symbol.replace('-', '/')

    def _get_swap_symbol(self, symbol: str) -> str:
        if self.exchange_id == 'okx':
            return symbol.replace('/', '-') + '-SWAP'
        else:
            return symbol.replace('-', '').replace('/', '')

    async def fetch_candles(self, symbol: str, timeframe: str, since: Optional[int] = None, limit: Optional[int] = 100) -> pd.DataFrame:
        ccxt_symbol = self._format_symbol(symbol)
        logger.debug(f"Fetching candles for {ccxt_symbol} with timeframe {timeframe}...")
        try:
            ohlcv = await self.exchange.fetch_ohlcv(ccxt_symbol, timeframe.lower(), since, limit)
            if not ohlcv:
                logger.warning(f"No candle data returned for {ccxt_symbol} with timeframe {timeframe}.")
                return pd.DataFrame()

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except BaseError as e:
            logger.error(f"Error fetching candles for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()

    async def fetch_historical_data(self, symbol: str, timeframe: str, years: float = None, since: int = None) -> pd.DataFrame:
        # from datetime import datetime, timedelta # Moved to top level

        timeframe_ms = self.exchange.parse_timeframe(timeframe.lower()) * 1000
        
        since_ms = since
        since_dt = None
        
        if since_ms is None:
            if years is None:
                # Default to a reasonable lookback if neither is provided, e.g., 1 year
                years = 1.0
            since_dt = datetime.utcnow() - timedelta(days=years * 365.25)
            since_ms = int(since_dt.timestamp() * 1000)
        else:
             since_dt = datetime.fromtimestamp(since_ms / 1000)

        ccxt_symbol = self._format_symbol(symbol) if self.market_type == 'spot' else self._get_swap_symbol(symbol)

        all_candles = []
        logger.info(f"Fetching historical data for {symbol} on {timeframe} since {since_dt.strftime('%Y-%m-%d %H:%M:%S')}...")

        while True:
            try:
                candles = await self.exchange.fetch_ohlcv(ccxt_symbol, timeframe.lower(), since=since_ms, limit=1000)
                if not candles:
                    break
                
                # --- Added Logging for Batch Details ---
                batch_start_time = datetime.fromtimestamp(candles[0][0] / 1000).strftime('%Y-%m-%d %H:%M:%S')
                batch_end_time = datetime.fromtimestamp(candles[-1][0] / 1000).strftime('%Y-%m-%d %H:%M:%S')
                # Note: Requested limit is 1000, but exchanges like OKX may cap at 100 or 300.
                logger.info(f"Fetched batch from [{self.exchange_id}] for {symbol}: {len(candles)} candles | Start: {batch_start_time} -> End: {batch_end_time}")
                # ---------------------------------------

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
        logger.info(f"Successfully fetched {len(df)} historical candles for {symbol} on {timeframe}.")
        return df[['open', 'high', 'low', 'close', 'volume']]

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
            self._record_error()
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
            self._record_error()
            return None
            
        except BaseError as e:
            logger.error(
                f"Unhandled CCXT BaseError creating order on {self.exchange.id}: {type(e).__name__}. "
                f"{order_details_for_logging}. Error: {e}",
                exc_info=True
            )
            self._record_error()
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

        ccxt_symbol = self._format_symbol(symbol) if self.market_type == 'spot' else self._get_swap_symbol(symbol)
        
        # Strategy: Paginated Fetching
        funding_rates = []
        current_since = since
        
        # If since is not provided, use default (1 year)
        if current_since is None:
            if years is None:
                years = 1.0
            since_dt = datetime.now(timezone.utc) - timedelta(days=years * 365.25)
            current_since = int(since_dt.timestamp() * 1000)

        logger.info(f"Fetching funding rates for {symbol} since {datetime.fromtimestamp(current_since/1000).isoformat()}...")

        while True:
            try:
                batch = await self.exchange.fetch_funding_rate_history(ccxt_symbol, since=current_since, limit=1000)
                if not batch:
                    break
                
                funding_rates.extend(batch)
                
                # Update current_since to the last record's timestamp + 1ms to avoid duplication
                last_ts = batch[-1]['timestamp']
                if last_ts <= current_since:
                    # Prevent infinite loop
                    break
                current_since = last_ts + 1
                
                # Check if we've reached current time
                if current_since > int(datetime.now(timezone.utc).timestamp() * 1000):
                    break
                
                await asyncio.sleep(self.exchange.rateLimit / 1000.0)
                
                if limit and len(funding_rates) >= limit:
                    funding_rates = funding_rates[:limit]
                    break
                    
            except (ExchangeError, BaseError) as e:
                logger.warning(f"Error fetching funding rate batch: {e}. Stopping pagination.")
                break
            except Exception as e:
                logger.error(f"Unexpected error fetching funding rates: {e}", exc_info=True)
                break

        # Fallback if paginated fetch failed or returned nothing
        if not funding_rates:
            logger.info("Pagination returned no data. Trying latest fallback...")
            try:
                funding_rates = await self.exchange.fetch_funding_rate_history(ccxt_symbol, limit=limit or 100)
            except Exception as e:
                logger.warning(f"Final fallback failed: {e}")
                return pd.DataFrame()

        if not funding_rates:
            return pd.DataFrame()
        
        df = pd.DataFrame(funding_rates)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        df.rename(columns={'fundingRate': 'funding_rate'}, inplace=True)
        return df[['funding_rate']]

    async def fetch_open_interest(self, symbol: str, timeframe: str, years: Optional[float] = None, since: Optional[int] = None, limit: Optional[int] = None) -> pd.DataFrame:
        if not self.exchange.has['fetchOpenInterestHistory']:
            logger.warning(f"Exchange {self.exchange.id} does not support fetching open interest history.")
            return pd.DataFrame()

        ccxt_symbol = self._format_symbol(symbol) if self.market_type == 'spot' else self._get_swap_symbol(symbol)
        
        # Determine initial 'since'
        if since is None:
            if years is None:
                years = 1.0 / 12.0 # Default to 30 days
            since_dt = datetime.now(timezone.utc) - timedelta(days=years * 365.25)
            since = int(since_dt.timestamp() * 1000)
            
        attempts = [
            ("Original", since),
            ("29 Days", int((datetime.now(timezone.utc) - timedelta(days=29)).timestamp() * 1000)),
            ("7 Days", int((datetime.now(timezone.utc) - timedelta(days=7)).timestamp() * 1000))
        ]
        
        # If original since is already within 30 days, don't retry 30 days again, etc.
        # Filter attempts to avoid redundant retries or retrying with OLDER dates (illogical)
        unique_attempts = []
        seen_timestamps = set()
        for label, ts in attempts:
            if ts not in seen_timestamps and ts >= since: # Only try shorter or equal lookbacks (larger timestamp)
                 seen_timestamps.add(ts)
                 unique_attempts.append((label, ts))
        
        # Ensure at least one attempt (the original) is made if logic filtered everything
        if not unique_attempts:
             unique_attempts = [("Original", since)]

        all_oi_data = []
        
        for label, start_ts in unique_attempts:
            logger.info(f"Attempting to fetch open interest for {symbol} on {timeframe} since {datetime.fromtimestamp(start_ts/1000).isoformat()} ({label})...")
            
            temp_data = []
            current_since = start_ts
            failed = False
            
            batch_limit = 500
            if limit and limit < batch_limit:
                batch_limit = limit

            while True:
                try:
                    batch = await self.exchange.fetch_open_interest_history(ccxt_symbol, timeframe.lower(), since=current_since, limit=batch_limit)
                    
                    if not batch:
                        break
                    
                    temp_data.extend(batch)
                    
                    if limit and len(temp_data) >= limit:
                        temp_data = temp_data[:limit]
                        break

                    last_ts = batch[-1]['timestamp']
                    if last_ts == current_since:
                         break
                    current_since = last_ts + 1
                    
                    if current_since > int(datetime.now(timezone.utc).timestamp() * 1000):
                        break
                        
                    await asyncio.sleep(self.exchange.rateLimit / 1000.0)
                    
                except (ExchangeError, BaseError) as e:
                    # Downgrade expected "too old" errors to INFO during "Original" attempt
                    msg = str(e)
                    is_expected = False
                    if label == "Original":
                        if "-1130" in msg: # Binance: invalid startTime
                            is_expected = True
                        elif "50030" in msg: # OKX: Illegal time range
                            is_expected = True
                    
                    if is_expected:
                        logger.info(f"Exchange limit reached for {label} lookback (Expected): {e}. switching to shorter history.")
                    elif "testnet/sandbox URL" in msg: # Binance Sandbox missing endpoint
                         logger.info(f"Binance Sandbox does not support this endpoint (Expected): {e}. Skipping.")
                         failed = True 
                         break 
                    else:
                        logger.warning(f"Error fetching batch with start_ts={current_since}: {e}")
                    
                    failed = True
                    break
                except Exception as e:
                    logger.error(f"Unexpected error: {e}", exc_info=True)
                    failed = True
                    break
            
            if failed:
                logger.info(f"Fetch with {label} lookback failed. Retrying with shorter history...")
                continue # Try next attempt
            else:
                 # Success!
                 all_oi_data = temp_data
                 logger.info(f"Successfully fetched {len(all_oi_data)} open interest records for {symbol}.")
                 break # specific attempt succeeded, break outer loop

        if not all_oi_data and not failed:
             # If loop finished without data and wasn't marked failed (e.g. empty response)
             # Try fallback to latest
             logger.info("Pagination strategies failed. Attempting to fetch latest open interest (Final Fallback).")
             try:
                 all_oi_data = await self.exchange.fetch_open_interest_history(ccxt_symbol, timeframe.lower(), limit=limit or 100)
             except Exception as e:
                 logger.warning(f"Final fallback fetch failed: {e}")

        if not all_oi_data:
            return pd.DataFrame()
            
        df = pd.DataFrame(all_oi_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        # Standardize column name
        if 'openInterestValue' in df.columns:
             df.rename(columns={'openInterestValue': 'open_interest_value'}, inplace=True)
        if 'openInterestAmount' in df.columns and 'open_interest' not in df.columns:
             df.rename(columns={'openInterestAmount': 'open_interest'}, inplace=True)
        
        cols_to_return = []
        if 'open_interest' in df.columns:
            cols_to_return.append('open_interest')
        if 'open_interest_value' in df.columns:
            cols_to_return.append('open_interest_value')
            
        return df[cols_to_return]

    async def fetch_long_short_ratio(self, symbol: str, timeframe: str, limit: Optional[int] = 100) -> pd.DataFrame:
        """
        Fetches the Top Trader Long/Short Ratio.
        Currently supports Binance and OKX via implicit API methods as CCXT unified support is partial.
        """
        ccxt_symbol = self._format_symbol(symbol) if self.market_type == 'spot' else self._get_swap_symbol(symbol)
        market_id = symbol.replace('/', '') # generic guess
        
        # Determine period string expected by exchange
        period = timeframe # Default
        if self.exchange_id == 'binance':
            # Binance expects '5m', '15m', '1h', etc.
            pass 
        elif self.exchange_id == 'okx':
            # OKX expects '5m', '1H', '4H' etc.
            # Map standard generic intervals to OKX format if needed
            period = timeframe # CCXT usually standardizes this, but raw params might need specific format
        
        data = []
        try:
            if self.exchange_id == 'binance':
                # Binance Futures: fapiDataGetTopLongShortAccountRatio
                # params: symbol, period, limit
                # Symbols for fapi are usually without slash, e.g. BTCUSDT
                f_symbol = symbol.replace('/', '')
                # Note: Correct CCXT mapping verified via dir() is fapiDataGetTopLongShortAccountRatio
                response = await self.exchange.fapiDataGetTopLongShortAccountRatio({
                    'symbol': f_symbol,
                    'period': timeframe,
                    'limit': limit
                })
                data = response
                
            elif self.exchange_id == 'okx':
                # OKX: public_get_rubik_stat_contracts_long_short_account_ratio
                # ccy: BTC, period: 5m
                # Need to extract base currency from symbol (BTC/USDT -> BTC)
                base = symbol.split('/')[0]
                response = await self.exchange.public_get_rubik_stat_contracts_long_short_account_ratio({
                    'ccy': base,
                    'period': timeframe,
                    # pattern: 1: top trader, 0: all trader? API says:
                    # OKX API path is specific. Let's rely on what we know.
                    # Actually standard endpoint returns List
                })
                if response and 'data' in response:
                    data = response['data']
            
            else:
                 logger.warning(f"fetch_long_short_ratio not implemented for {self.exchange_id}")
                 return pd.DataFrame()

        except Exception as e:
            msg = str(e)
            if "testnet/sandbox URL" in msg:
                 logger.info(f"Binance Sandbox does not support Long/Short Ratio (Expected). Skipping.")
            else:
                 logger.warning(f"Failed to fetch Long/Short Ratio from {self.exchange_id}: {e}")
            return pd.DataFrame()

        if not data:
            return pd.DataFrame()

        # Parse Data
        # Binance: [{'symbol': 'BTCUSDT', 'longShortRatio': '1.4500', 'longAccount': '0.5918', 'shortAccount': '0.4082', 'timestamp': 170...}]
        # OKX: [{'ts': '170...', 'ratio': '1.45', 'long': '0.59', 'short': '0.41'}]
        
        records = []
        for item in data:
            record = {}
            if self.exchange_id == 'binance':
                record['timestamp'] = int(item['timestamp'])
                record['long_short_ratio'] = float(item['longShortRatio'])
                record['long_account'] = float(item['longAccount'])
                record['short_account'] = float(item['shortAccount'])
            elif self.exchange_id == 'okx':
                record['timestamp'] = int(item['ts'])
                record['long_short_ratio'] = float(item['ratio'])
                record['long_account'] = float(item['long'])
                record['short_account'] = float(item['short'])
            records.append(record)

        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        df = df.sort_index()
        
        # Rename for consistency with feature engineering whitelist
        df.rename(columns={
            'long_short_ratio': 'toptrader_long_short_ratio',
             # Add other derived cols if available
        }, inplace=True)

        return df[['toptrader_long_short_ratio']]
        
        df = df[~df.index.duplicated(keep='first')]
        
        if 'open_interest' in df.columns:
            logger.info(f"Successfully fetched {len(df)} open interest records for {symbol}.")
            return df[['open_interest']]
        else:
            logger.warning(f"Open interest data from {self.exchange.id} missing expected columns. Available: {df.columns}")
            return pd.DataFrame()