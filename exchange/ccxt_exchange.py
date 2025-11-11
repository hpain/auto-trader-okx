import os
import ccxt
import pandas as pd
from typing import List, Dict, Any, Optional
from exchange.base import Exchange
import logging
import asyncio

logger = logging.getLogger(__name__)

class CcxtExchange(Exchange):
    def __init__(self, exchange_id: str, api_key: str = None, api_secret: str = None, passphrase: str = None, sandbox: bool = True, market_type: str = 'spot'):
        """
        Initializes the CcxtExchange.
        :param exchange_id: The ID of the exchange to connect to (e.g., 'binance', 'okx').
        :param market_type: The market type to use, e.g., 'spot', 'swap'.
        """
        self.exchange_id = exchange_id.lower()
        self.market_type = market_type
        exchange_class = getattr(ccxt, self.exchange_id)
        
        config = {
            'apiKey': api_key,
            'secret': api_secret,
            'timeout': 30000,  # 30 seconds
            'options': {
                'defaultType': market_type,
            },
        }
        if passphrase:
            config['password'] = passphrase

        # --- START PROXY FIX ---
        https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
        http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
        
        if https_proxy:
            config['httpsProxy'] = https_proxy
            logger.info(f"Applying system HTTPS_PROXY: {https_proxy}")
        elif http_proxy:
            config['httpProxy'] = http_proxy
            logger.info(f"Applying system HTTP_PROXY: {http_proxy}")
        # --- END PROXY FIX ---

        self.exchange = exchange_class(config)
        
        if sandbox:
            if 'test' in self.exchange.urls:
                self.exchange.set_sandbox_mode(True)
            else:
                logger.warning(f"{self.exchange.id} does not support sandbox mode through ccxt. Live mode will be used.")

        # --- START BYBIT V5 MARKET LOADING FIX ---
        if self.exchange_id == 'bybit' and self.market_type == 'swap':
            try:
                logger.info("Bybit swap market detected. Explicitly loading linear markets for v5 API.")
                self.exchange.load_markets(params={'category': 'linear'})
                logger.info("Successfully loaded linear markets for Bybit.")
            except Exception as e:
                logger.error(f"Failed to explicitly load linear markets for Bybit: {e}", exc_info=True)
        # --- END BYBIT V5 MARKET LOADING FIX ---

        logger.info(f"Initialized CcxtExchange for {self.exchange.id} with market_type='{market_type}' (Sandbox: {sandbox})")

    def _format_symbol(self, symbol: str) -> str:
        """Convert internal symbol format (e.g., BTC-USDT) to ccxt format (e.g., BTC/USDT)."""
        return symbol.replace('-', '/')

    def _get_swap_symbol(self, symbol: str) -> str:
        """Converts an internal spot-like symbol (e.g., BTC-USDT) to a CCXT-compatible swap symbol."""
        if self.exchange_id == 'okx':
            return symbol + '-SWAP'
        else:  # For binance, bybit, etc.
            return symbol.replace('-', '')

    def fetch_candles(self, symbol: str, timeframe: str, since: Optional[int] = None, limit: Optional[int] = 100) -> pd.DataFrame:
        """
        Fetch historical OHLCV candles from the exchange.
        """
        ccxt_symbol = self._format_symbol(symbol)
        timeframe = timeframe.lower()
        logger.debug(f"Fetching candles for {ccxt_symbol} with timeframe {timeframe}...")
        ohlcv = self.exchange.fetch_ohlcv(ccxt_symbol, timeframe, since, limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df.rename(columns={'volume': 'vol'}, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    def fetch_historical_data(self, symbol: str, timeframe: str, years: float) -> pd.DataFrame:
        """
        Fetch a large amount of historical data using a generic, forward-paginating approach.
        This method is compatible with the majority of exchanges in ccxt.
        """
        import time
        from datetime import datetime, timedelta

        internal_symbol = symbol
        timeframe = timeframe.lower()
        ccxt_symbol = self._format_symbol(symbol) if self.market_type == 'spot' else self._get_swap_symbol(symbol)
        
        timeframe_duration_ms = self.exchange.parse_timeframe(timeframe) * 1000
        
        since_dt = datetime.utcnow() - timedelta(days=years * 365.25)
        since_ms = int(since_dt.timestamp() * 1000)

        all_candles = []
        logger.info(f"Fetching historical data for {internal_symbol} on timeframe {timeframe} since {since_dt.strftime('%Y-%m-%d %H:%M:%S')} (using generic forward pagination). Market type: {self.market_type}")

        page_no = 1
        max_retries = 5
        retries = 0

        while True:
            try:
                candles = self.exchange.fetch_ohlcv(ccxt_symbol, timeframe, since=since_ms, limit=1000)
                
                if not candles:
                    logger.info("OK: No more data returned from exchange, fetch complete.")
                    break

                last_candle_ts = candles[-1][0]
                
                chunk_df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                chunk_df['timestamp'] = pd.to_datetime(chunk_df['timestamp'], unit='ms', utc=True)
                newest, oldest = chunk_df['timestamp'].max(), chunk_df['timestamp'].min()
                logger.info(f"OK 第 {page_no} 页: {len(chunk_df)} 条, 最新: {newest}, 最旧: {oldest}")
                page_no += 1

                all_candles.extend(candles)

                since_ms = last_candle_ts + timeframe_duration_ms
                
                retries = 0

                time.sleep(self.exchange.rateLimit / 1000)

            except ccxt.NetworkError as e:
                retries += 1
                if retries <= max_retries:
                    logger.warning(f"WARN: A network error occurred: {e}. Retrying in 5 seconds... (Attempt {retries}/{max_retries})")
                    time.sleep(5)
                else:
                    logger.critical(f"CRITICAL: Exceeded max retries ({max_retries}) for network errors. Aborting data fetch for {internal_symbol}.")
                    break
            except Exception as e:
                logger.critical(f"CRITICAL: An unexpected error occurred while fetching data for {internal_symbol}: {e}", exc_info=True)
                break
        
        if not all_candles:
            logger.warning("No historical data was fetched.")
            return pd.DataFrame()

        full_df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        full_df.drop_duplicates(subset=['timestamp'], inplace=True)
        full_df['timestamp'] = pd.to_datetime(full_df['timestamp'], unit='ms', utc=True)
        full_df.set_index('timestamp', inplace=True)
        full_df.sort_index(inplace=True)
        
        full_df.rename(columns={'volume': 'vol'}, inplace=True)

        logger.info(f"DONE 返回 {len(full_df)} 条有效K线 ({internal_symbol}, {timeframe}, {years}y)")
        
        return full_df[['open', 'high', 'low', 'close', 'vol']]

    def fetch_funding_rates(self, symbol: str, timeframe: str, years: Optional[float] = None, since: Optional[int] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetches recent funding rates and filters them by the requested time window.
        """
        import time
        from datetime import datetime, timedelta

        if not self.exchange.has['fetchFundingRateHistory']:
            logger.warning(f"Exchange {self.exchange.id} does not support fetching funding rate history.")
            return pd.DataFrame()

        ccxt_symbol = self._get_swap_symbol(symbol)
        
        fetch_limit = 1000
        logger.info(f"Fetching latest {fetch_limit} funding rates for {symbol} and filtering by date.")

        all_data = []
        try:
            all_data = self.exchange.fetch_funding_rate_history(ccxt_symbol, since=None, limit=fetch_limit)
        except Exception as e:
            logger.error(f"Error fetching funding rates for {symbol} from {self.exchange.id}: {e}", exc_info=True)
            return pd.DataFrame()

        if not all_data:
            logger.warning(f"No funding rates fetched for {symbol}.")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df.rename(columns={'fundingRate': 'funding_rate'}, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.drop_duplicates(subset=['timestamp'], inplace=True)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        since_dt = None
        if since is not None:
            since_dt = pd.to_datetime(since, unit='ms', utc=True)
        elif years is not None:
            since_dt = datetime.utcnow().replace(tzinfo=pd.Timestamp.utcnow().tz) - timedelta(days=years * 365.25)

        if since_dt:
            df = df[df.index >= since_dt]
            logger.info(f"Filtered funding rates, returning {len(df)} entries since {since_dt.strftime('%Y-%m-%d')}.")

        return df[['funding_rate']]

    def fetch_open_interest(self, symbol: str, timeframe: str, years: Optional[float] = None, since: Optional[int] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetches recent open interest and filters it by the requested time window.
        """
        import time
        from datetime import datetime, timedelta

        if not self.exchange.has['fetchOpenInterestHistory']:
            logger.warning(f"Exchange {self.exchange.id} does not support fetching open interest history.")
            return pd.DataFrame()

        ccxt_symbol = self._get_swap_symbol(symbol)

        fetch_limit = 1000
        logger.info(f"Fetching latest {fetch_limit} open interest data points for {symbol} and filtering by date.")

        all_data = []
        try:
            params = {}
            if self.exchange_id == 'bybit' and self.market_type == 'swap':
                params['category'] = 'linear'
            all_data = self.exchange.fetch_open_interest_history(ccxt_symbol, since=None, limit=fetch_limit, params=params)
        except Exception as e:
            logger.error(f"Error fetching open interest for {symbol} from {self.exchange.id}: {e}", exc_info=True)
            return pd.DataFrame()

        if not all_data:
            logger.warning(f"No open interest fetched for {symbol}.")
            return pd.DataFrame()

        processed_data = []
        for entry in all_data:
            # Correctly parse the OI value. Fallback to 'openInterestAmount' if 'Value' is not available.
            oi_value = entry.get('openInterestValue')
            if oi_value is None:
                oi_value = entry.get('openInterestAmount')

            if oi_value is not None:
                processed_data.append({
                    'timestamp': entry['timestamp'],
                    'open_interest': oi_value
                })

        if not processed_data:
            logger.warning(f"No valid open interest entries found for {symbol} after processing.")
            return pd.DataFrame()

        df = pd.DataFrame(processed_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.drop_duplicates(subset=['timestamp'], inplace=True)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        since_dt = None
        if since is not None:
            since_dt = pd.to_datetime(since, unit='ms', utc=True)
        elif years is not None:
            since_dt = datetime.utcnow().replace(tzinfo=pd.Timestamp.utcnow().tz) - timedelta(days=years * 365.25)

        if since_dt:
            df = df[df.index >= since_dt]
            logger.info(f"Filtered open interest, returning {len(df)} entries since {since_dt.strftime('%Y-%m-%d')}.")

        return df[['open_interest']]

    def get_balance(self, currency: str) -> float:
        """
        Get the balance for a specific currency.
        """
        balance = self.exchange.fetch_balance()
        return balance['free'].get(currency, 0.0)

    def create_order(self, symbol: str, order_type: str, side: str, amount: float, price: Optional[float] = None) -> Dict[str, Any]:
        """
        Create a new order.
        """
        symbol = self._format_symbol(symbol)
        return self.exchange.create_order(symbol, order_type, side, amount, price)

    def get_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Get details of a specific order.
        """
        symbol = self._format_symbol(symbol)
        return self.exchange.fetch_order(order_id, symbol)

    def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Cancel an order.
        """
        symbol = self._format_symbol(symbol)
        return self.exchange.cancel_order(order_id, symbol)

    def get_current_price(self, symbol: str) -> float:
        """
        Get the current price of a symbol.
        """
        symbol = self._format_symbol(symbol)
        ticker = self.exchange.fetch_ticker(symbol)
        return ticker['last']

    def place_oco_order(self, symbol: str, side: str, amount: float, take_profit_price: float, stop_loss_price: float) -> Dict[str, Any]:
        """
        Place a One-Cancels-the-Other order.
        """
        symbol = self._format_symbol(symbol)
        if not self.exchange.has['createOco']:
            raise NotImplementedError(f"{self.exchange.id} does not support OCO orders through ccxt.")

        params = {
            'stopPrice': stop_loss_price,
            'stopLimitPrice': stop_loss_price,
        }
        
        return self.exchange.create_order(
            symbol=symbol,
            type='oco',
            side=side,
            amount=amount,
            price=take_profit_price,
            params=params
        )