import requests
import time
import hmac
import hashlib
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
import json

from exchange.base import Exchange

class BinanceExchange(Exchange):
    def __init__(self, api_key: str = None, api_secret: str = None, passphrase: str = None, sandbox: bool = False):
        self.api_key = api_key
        self.secret_key = api_secret
        # Binance doesn't use a passphrase, but we keep it for compatibility with the base class
        self.passphrase = passphrase 
        self.sandbox = sandbox
        if self.sandbox:
            # Binance has a specific testnet URL
            self.base_url = "https://testnet.binance.vision"
        else:
            self.base_url = "https://api.binance.com"
        # Use a session object for connection pooling and performance
        self.session = requests.Session()

    def _get_timestamp(self):
        return int(time.time() * 1000)

    def _sign(self, params: dict):
        query_string = '&'.join([f"{key}={value}" for key, value in params.items()])
        signature = hmac.new(self.secret_key.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
        return signature

    def _request(self, method, path, params=None, body=None, authenticated=False):
        url = self.base_url + path
        headers = {}
        if self.api_key:
            headers['X-MBX-APIKEY'] = self.api_key

        if params is None:
            params = {}

        if authenticated:
            params['timestamp'] = self._get_timestamp()
            query_string_to_sign = '&'.join([f"{key}={value}" for key, value in params.items()])
            signature = hmac.new(self.secret_key.encode('utf-8'), query_string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
            params['signature'] = signature

        try:
            if method.upper() == 'GET':
                response = self.session.get(url, headers=headers, params=params, timeout=30)
            elif method.upper() == 'POST':
                # For POST, params are sent as URL query parameters
                response = self.session.post(url, headers=headers, params=params, json=body, timeout=30)
            elif method.upper() == 'DELETE':
                # For DELETE, params are also sent as URL query parameters
                response = self.session.delete(url, headers=headers, params=params, timeout=30)
            else:
                raise ValueError("Unsupported HTTP method")
            
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e.response.status_code} {e.response.text}")
            # Try to return JSON error body if possible
            try:
                return e.response.json()
            except json.JSONDecodeError:
                return {'error': e.response.text}
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON decode failed: {e}. Response text: {response.text}")
            return None

    def fetch_candles(self, symbol: str, timeframe: str, since: Optional[int] = None, limit: Optional[int] = None, endTime: Optional[int] = None) -> pd.DataFrame:
        path = "/api/v3/klines"
        params = {
            'symbol': symbol.replace('-', ''),
            'interval': timeframe.lower(),
        }
        if limit:
            params['limit'] = limit
        if since:
            params['startTime'] = since
        if endTime:
            params['endTime'] = endTime

        data = self._request('GET', path, params=params)
        
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms', utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
           df[col] = df[col].astype(float)
        df.rename(columns={'volume': 'vol'}, inplace=True)
        df = df.set_index('timestamp')
        df = df.sort_index() # Sort by the new DatetimeIndex
        return df[['open', 'high', 'low', 'close', 'vol']]
    
    def get_balance(self, currency: str) -> float:
        path = '/api/v3/account'
        response = self._request('GET', path, authenticated=True)
        if response and 'balances' in response:
            for balance in response['balances']:
                if balance['asset'] == currency.upper():
                    return float(balance['free'])
        return 0.0

    def create_order(self, symbol: str, order_type: str, side: str, amount: float, price: Optional[float] = None) -> Dict[str, Any]:
        path = '/api/v3/order'
        params = {
            'symbol': symbol.replace('-', ''),
            'side': side.upper(),
            'type': order_type.upper(),
            'quantity': str(amount),
        }
        if order_type.upper() == 'LIMIT':
            if price is None:
                raise ValueError("Price must be specified for LIMIT orders")
            params['price'] = str(price)
            params['timeInForce'] = 'GTC'

        return self._request('POST', path, params=params, authenticated=True)

    def get_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        path = '/api/v3/order'
        params = {
            'symbol': symbol.replace('-', ''),
            'orderId': order_id,
        }
        return self._request('GET', path, params=params, authenticated=True)

    def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        path = '/api/v3/order'
        params = {
            'symbol': symbol.replace('-', ''),
            'orderId': order_id,
        }
        return self._request('DELETE', path, params=params, authenticated=True)
    
    def get_current_price(self, symbol: str) -> float:
        path = '/api/v3/ticker/price'
        params = {'symbol': symbol.replace('-', '')}
        data = self._request('GET', path, params=params)
        if data and 'price' in data:
            return float(data['price'])
        return 0.0

    def fetch_historical_data(self, symbol: str, timeframe: str, years: float) -> pd.DataFrame:
        """
        Fetch a large amount of historical data from Binance.
        """
        print(f"币安交易所获取历史数据")
        print(f"Fetching historical data for {symbol} on timeframe {timeframe} for the last {years} years (using robust pagination).")
        
        # Binance API limit per request (max 1000)
        limit = 1000
        max_pages = 5000 # Safety break
        
        # Calculate the start date
        target_time = pd.Timestamp.utcnow() - pd.Timedelta(days=years * 365.25)

        all_data = []
        # We fetch backwards from the current time
        end_time_param = int(datetime.utcnow().timestamp() * 1000)

        for page_no in range(1, max_pages + 1):
            # Binance's `endTime` is exclusive for the kline at that exact millisecond.
            chunk = self.fetch_candles(symbol=symbol, timeframe=timeframe, endTime=end_time_param, limit=limit)
            
            if chunk.empty:
                print("OK 数据获取完毕")
                break
            
            # --- 关键修复：匹配 data/binance.py 的日志输出 ---
            newest, oldest = chunk.index.max(), chunk.index.min()
            print(f"OK 第 {page_no} 页: {len(chunk)} 条, 最新: {newest}, 最旧: {oldest}")
            # --- 日志修复结束 ---

            all_data.append(chunk)
            
            # The oldest timestamp in the current chunk becomes the end time for the next request
            oldest_ts_in_chunk = chunk.index.min()
            
            if oldest_ts_in_chunk < target_time:
                print(f"OK: Reached target start time of {target_time}.")
                break # 提前退出，因为我们已经拿到了比目标更早的数据
                
            # The next request's end time must be the timestamp of the oldest record we just got.
            # Crucial fix: Subtract 1ms to avoid re-fetching the same candle and getting stuck in a loop.
            end_time_param = int(oldest_ts_in_chunk.timestamp() * 1000) - 1
            
            # Binance has rate limits, so a small sleep is good practice
            time.sleep(0.2) 

        if not all_data:
            print("No historical data fetched.")
            return pd.DataFrame()

        # Concatenate all chunks and remove duplicates
        full_df = pd.concat(all_data)
        full_df = full_df[~full_df.index.duplicated(keep='first')]
        full_df = full_df.sort_index()
        
        # --- 关键修复：匹配 data/binance.py 的日志输出 ---
        print(f"DONE 返回 {len(full_df)} 条有效K线 ({symbol}, {timeframe}, {years}y)")
        # --- 日志修复结束 ---

        return full_df


    def place_oco_order(self, symbol: str, side: str, amount: float, take_profit_price: float, stop_loss_price: float) -> Dict[str, Any]:
        path = '/api/v3/order/oco'
        
        # Binance requires the stop price to be different for buy and sell orders.
        # For SELL OCO, stopPrice must be below the market price. Here it acts as a stop-loss.
        # For BUY OCO, stopPrice must be above the market price.
        # This implementation assumes a standard stop-loss scenario for a long position (side=SELL)
        # or a short position (side=BUY).

        params = {
            'symbol': symbol.replace('-', ''),
            'side': side.upper(),
            'quantity': str(amount),
            'price': str(take_profit_price),  # This is the take-profit limit order price
            'stopPrice': str(stop_loss_price), # This is the trigger price for the stop-loss
            # The stopLimitPrice is the price at which the stop-loss order will be placed once triggered.
            # Setting it slightly lower (for sell) or higher (for buy) than the stopPrice can ensure it gets filled.
            'stopLimitPrice': str(stop_loss_price), 
            'stopLimitTimeInForce': 'GTC', # Good-Til-Canceled
        return self._request('POST', path, params=params, authenticated=True)

    def fetch_funding_rates(self, symbol: str, timeframe: str, years: Optional[float] = None, since: Optional[int] = None, limit: Optional[int] = None) -> pd.DataFrame:
        path = "/fapi/v1/fundingRate" # Futures Funding Rate History
        params = {
            'symbol': symbol.replace('-', ''),
        }
        if limit:
            params['limit'] = limit
        if since:
            params['startTime'] = since

        # Binance API for funding rate does not use 'timeframe' in the same way as klines.
        # It returns historical funding rates. We will fetch based on 'since' or 'limit'.
        # 'years' would require multiple requests and pagination, similar to fetch_historical_data.
        # For simplicity, initially, we'll focus on 'since' and 'limit'.

        data = self._request('GET', path, params=params)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms', utc=True)
        df['fundingRate'] = df['fundingRate'].astype(float)
        df = df.rename(columns={'fundingTime': 'timestamp', 'fundingRate': 'rate'})
        df = df.set_index('timestamp')
        df = df.sort_index()
        return df[['rate']]

    def fetch_open_interest(self, symbol: str, timeframe: str, years: Optional[float] = None, since: Optional[int] = None, limit: Optional[int] = None) -> pd.DataFrame:
        path = "/fapi/v1/openInterest" # Futures Open Interest
        params = {
            'symbol': symbol.replace('-', ''),
            'interval': timeframe # Binance API supports m, h, d for open interest
        }
        if limit:
            params['limit'] = limit
        if since:
            params['startTime'] = since

        data = self._request('GET', path, params=params)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['openInterest'] = df['openInterest'].astype(float)
        df = df.rename(columns={'openInterest': 'oi'})
        df = df.set_index('timestamp')
        df = df.sort_index()
        return df[['oi']]

