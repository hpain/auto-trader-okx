import requests, time
import pandas as pd
import os
from datetime import datetime
import hmac
import base64
import json
import logging
from urllib.parse import urlencode
from typing import List, Dict, Any, Optional

from exchange.base import Exchange

class OKXExchange(Exchange):
    def __init__(self, api_key: str = None, api_secret: str = None, passphrase: str = None, sandbox: bool = False):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key
        self.secret_key = api_secret
        self.passphrase = passphrase
        self.sandbox = sandbox
        self.base_url = "https://www.okx.com"

    def _get_timestamp(self):
        return datetime.utcnow().isoformat()[:-3] + 'Z'

    def _sign(self, timestamp, method, request_path, body=''):
        if body:
            body = json.dumps(body)
        message = timestamp + method.upper() + request_path + body
        secret_key_str = str(self.secret_key) if self.secret_key is not None else ""
        mac = hmac.new(bytes(secret_key_str, encoding='utf-8'), bytes(message, encoding='utf-8'), digestmod='sha256')
        d = mac.digest()
        return base64.b64encode(d)

    def _request(self, method, request_path, params=None, body=None, authenticated=False):
        url = self.base_url + request_path
        if params:
            url += '?' + urlencode(params)

        headers = {}
        
        if authenticated:
            if not self.api_key or not self.secret_key or not self.passphrase:
                self.logger.error("API key, secret, or passphrase not configured for authenticated request.")
                raise ValueError("API key, secret and passphrase required for authenticated requests")
            
            timestamp = datetime.utcnow().isoformat()[:-3] + 'Z'
            body_str = json.dumps(body) if body else ''
            
            # The message to sign is a concatenation of timestamp, method, request path, and body.
            message = timestamp + method.upper() + request_path + body_str
            signature = self._sign(timestamp, method, request_path, body_str)
            
            headers = {
                'Content-Type': 'application/json',
                'OK-ACCESS-KEY': self.api_key,
                'OK-ACCESS-SIGN': signature,
                'OK-ACCESS-TIMESTAMP': timestamp,
                'OK-ACCESS-PASSPHRASE': self.passphrase,
            }
            self.logger.debug(f"Request URL: {url}")
            self.logger.debug(f"Request Headers: {headers}")
            self.logger.debug(f"Message to Sign: '{message}'")
            self.logger.debug(f"Signature: {signature}")

        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, timeout=10)
            elif method.upper() == 'POST':
                # For POST requests, send the dictionary 'body' as JSON
                response = requests.post(url, headers=headers, json=body, timeout=10)
            else:
                self.logger.error(f"Unsupported HTTP method: {method}")
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            # Some successful OKX API calls might return an empty body.
            if response.content:
                return response.json()
            return None
        except requests.exceptions.HTTPError as e:
            # Log the detailed error message from the exchange
            self.logger.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during request: {e}")
            raise

    def fetch_candles(self, symbol: str, timeframe: str, since: Optional[int] = None, limit: Optional[int] = None) -> pd.DataFrame:
        VALID_BARS = {"1m", "3m", "5m", "15m", "30m", "1H", "2H", "4H", "6H", "12H", "1D", "1W", "1M"}
        if timeframe not in VALID_BARS:
           raise ValueError(f"Invalid timeframe: {timeframe}, must be one of {VALID_BARS}")

        params = {'instId': symbol, 'bar': timeframe}
        if limit:
            params['limit'] = limit
        if since:
            # OKX uses 'after' for pagination, which is a timestamp. 'since' can be converted.
            params['after'] = since

        data = self._request('GET', '/api/v5/market/candles', params=params)
        
        if not data or 'data' not in data:
            return pd.DataFrame()

        df = pd.DataFrame(data['data'], columns=["timestamp", "open", "high", "low", "close", "vol", "vol_ccy", "vol_ccy_quote", "confirm"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        df = df.sort_values("timestamp").reset_index(drop=True)
        for col in ["open", "high", "low", "close", "vol"]:
           df[col] = df[col].astype(float)

        return df

    def fetch_historical_data(self, symbol: str, timeframe: str, years: float) -> pd.DataFrame:
        """
        Fetch a large amount of historical data for backtesting or training.
        """
        all_data = []
        end_time = None
        limit = 100  # OKX API每页最多100条

        # Calculate the total number of candles to fetch
        # This is a rough estimation
        if 'm' in timeframe:
            minutes = int(timeframe.replace('m', ''))
            total_candles = years * 365 * 24 * (60 / minutes)
        elif 'H' in timeframe:
            hours = int(timeframe.replace('H', ''))
            total_candles = years * 365 * (24 / hours)
        else: # 'D', 'W', 'M'
            total_candles = years * 365 # good enough approximation

        fetched = 0

        while fetched < total_candles:
            params = {
                'instId': symbol,
                'bar': timeframe,
                'limit': str(limit)
            }
            if end_time:
                params['after'] = end_time

            data = self._request('GET', '/api/v5/market/history-candles', params=params)

            if not data or "data" not in data or len(data["data"]) == 0:
                break

            batch = data["data"]
            all_data.extend(batch)
            fetched += len(batch)
            end_time = batch[0][0]
            time.sleep(0.2)

        df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "vol", "vol_ccy", "vol_ccy_quote", "confirm"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        df = df.sort_values("timestamp").reset_index(drop=True)
        for col in ["open", "high", "low", "close", "vol"]:
           df[col] = df[col].astype(float)

        return df

    def get_balance(self, currency: str) -> float:
        response = self._request('GET', '/api/v5/account/balance', authenticated=True)
        if response and response.get("code") == "0" and response.get("data"):
            for detail in response['data'][0]['details']:
                if detail['ccy'] == currency:
                    return float(detail['availBal'])
        return 0.0

    def create_order(self, symbol: str, order_type: str, side: str, amount: float, price: Optional[float] = None) -> Dict[str, Any]:
        body = {
            "instId": symbol,
            "tdMode": "cash",
            "side": side,
            "ordType": order_type,
            "sz": str(amount)
        }
        if order_type == 'limit' and price is not None:
            body['px'] = str(price)
            
        response = self._request('POST', '/api/v5/trade/order', body=body, authenticated=True)
        return response

    def get_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        params = {'instId': symbol, 'ordId': order_id}
        return self._request('GET', '/api/v5/trade/order', params=params, authenticated=True)

    def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        body = {'instId': symbol, 'ordId': order_id}
        return self._request('POST', '/api/v5/trade/cancel-order', body=body, authenticated=True)

    def get_current_price(self, symbol: str) -> float:
        params = {'instId': symbol}
        data = self._request('GET', '/api/v5/market/ticker', params=params)
        if data and 'data' in data and len(data['data']) > 0:
            return float(data['data'][0]['last'])
        return 0.0

    def place_oco_order(self, symbol: str, side: str, amount: float, take_profit_price: float, stop_loss_price: float) -> Dict[str, Any]:
        """下单并附带止盈和止损"""
        self.logger.info(f"准备下 OCO 订单: {side.upper()} {amount} {symbol}, TP: {take_profit_price}, SL: {stop_loss_price}")
        body = {
            "instId": symbol,
            "tdMode": "cash",
            "side": side,
            "ordType": "oco",
            "sz": str(amount),
            "tpTriggerPx": str(take_profit_price),
            "tpOrdPx": "-1",  # 以市价执行止盈
            "slTriggerPx": str(stop_loss_price),
            "slOrdPx": "-1"   # 以市价执行止损
        }
        response = self._request('POST', '/api/v5/trade/order', body=body, authenticated=True)
        self.logger.info(f"OCO 下单响应: {response}")
        return response

    def fetch_funding_rates(self, symbol: str, timeframe: str, years: Optional[float] = None, since: Optional[int] = None, limit: Optional[int] = None) -> pd.DataFrame:
        # OKX funding rate history endpoint
        path = "/api/v5/public/funding-rate-history"
        params = {
            'instId': symbol
        }
        if limit:
            params['limit'] = limit
        if since:
            params['after'] = since # OKX uses 'after' for pagination

        data = self._request('GET', path, params=params)

        if not data or 'data' not in data:
            return pd.DataFrame()

        df = pd.DataFrame(data['data'], columns=['instId', 'fundingRate', 'fundingTime'])
        df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms', utc=True)
        df['fundingRate'] = df['fundingRate'].astype(float)
        df = df.rename(columns={'fundingTime': 'timestamp', 'fundingRate': 'rate'})
        df = df.set_index('timestamp')
        df = df.sort_index()
        return df[['rate']]

    def fetch_open_interest(self, symbol: str, timeframe: str, years: Optional[float] = None, since: Optional[int] = None, limit: Optional[int] = None) -> pd.DataFrame:
        # OKX open interest endpoint
        path = "/api/v5/market/open-interest"
        params = {
            'instId': symbol,
            'instType': 'SWAP' # Assuming we are interested in swap open interest
        }
        # OKX open interest endpoint does not support 'since' or 'limit' for historical data directly
        # It provides current open interest or a limited history via 'bar' parameter for candles.
        # For historical open interest, we might need to fetch candles and extract OI from there if available,
        # or use a different endpoint if OKX provides one.
        # For now, we will fetch the current open interest.

        data = self._request('GET', path, params=params)

        if not data or 'data' not in data:
            return pd.DataFrame()

        # The open interest endpoint returns current open interest, not historical series.
        # If historical open interest is needed, a different approach or endpoint is required.
        # For now, we return a DataFrame with the latest open interest.
        oi_data = data['data'][0]
        df = pd.DataFrame([{
            'timestamp': pd.to_datetime(oi_data['ts'], unit='ms', utc=True),
            'oi': float(oi_data['oi'])
        }])
        df = df.set_index('timestamp')
        return df[['oi']]