import requests, time
import pandas as pd
import os
from datetime import datetime
import hmac
import base64
import json

class OKXClient:
    def __init__(self, api_key, secret_key, passphrase, flag):
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.flag = flag
        # flag: 1 for live trading, 0 for demo trading
        self.base_url = "https://www.okx.com"

    def _get_timestamp(self):
        return datetime.utcnow().isoformat()[:-3] + 'Z'

    def _sign(self, timestamp, method, request_path, body=''):
        if body:
            body = json.dumps(body)
        message = timestamp + method.upper() + request_path + body
        mac = hmac.new(bytes(self.secret_key, encoding='utf-8'), bytes(message, encoding='utf-8'), digestmod='sha256')
        d = mac.digest()
        return base64.b64encode(d)

    def _request(self, method, request_path, params=None, body=None, authenticated=False):
        url = self.base_url + request_path
        if params:
            url += '?' + '&'.join([f'{k}={v}' for k, v in params.items()])

        headers = {}
        # 模拟盘交易 header
        if self.flag == '0':
            headers['x-simulated-trading'] = '1'

        if authenticated:
            timestamp = self._get_timestamp()
            headers['OK-ACCESS-KEY'] = self.api_key
            headers['OK-ACCESS-SIGN'] = self._sign(timestamp, method, request_path, body if body else '')
            headers['OK-ACCESS-TIMESTAMP'] = timestamp
            headers['OK-ACCESS-PASSPHRASE'] = self.passphrase
            headers['Content-Type'] = 'application/json'

        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=headers, data=json.dumps(body) if body else None)
            else:
                raise ValueError("Unsupported HTTP method")
            
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"请求失败: {e}")
            return None
        except json.JSONDecodeError:
            print(f"JSON解码失败: {response.text}")
            return None

    def get_kline(self, symbol="BTC-USDT", interval="1h", limit=100):
        VALID_BARS = {"1m", "3m", "5m", "15m", "30m", "1H", "2H", "4H", "6H", "12H", "1D", "1W", "1M"}
        if interval not in VALID_BARS:
           raise ValueError(f"无效的 interval 参数: {interval}, 必须是 {VALID_BARS}")

        params = {'instId': symbol, 'bar': interval, 'limit': limit}
        data = self._request('GET', '/api/v5/market/candles', params=params)
        
        if not data or 'data' not in data:
            return None

        df = pd.DataFrame(data['data'], columns=["timestamp", "open", "high", "low", "close", "vol", "vol_ccy", "vol_ccy_quote", "confirm"])
        df["close"] = df["close"].astype(float)
        return df[::-1]

    def place_order(self, symbol, side, quantity, order_type='market'):
        print(f"准备下单: {side.upper()} {quantity} {symbol}")
        body = {
            "instId": symbol,
            "tdMode": "cash",
            "side": side,
            "ordType": order_type,
            "sz": str(quantity)
        }
        response = self._request('POST', '/api/v5/trade/order', body=body, authenticated=True)
        print(f"下单响应: {response}")
        return response

    def place_oco_order(self, symbol, side, quantity, take_profit_price, stop_loss_price):
        """下单并附带止盈和止损"""
        print(f"准备下 OCO 订单: {side.upper()} {quantity} {symbol}, TP: {take_profit_price}, SL: {stop_loss_price}")
        body = {
            "instId": symbol,
            "tdMode": "cash",
            "side": side,
            "ordType": "oco",
            "sz": str(quantity),
            "tpTriggerPx": str(take_profit_price),
            "tpOrdPx": "-1",  # 以市价执行止盈
            "slTriggerPx": str(stop_loss_price),
            "slOrdPx": "-1"   # 以市价执行止损
        }
        response = self._request('POST', '/api/v5/trade/order', body=body, authenticated=True)
        print(f"OCO 下单响应: {response}")
        return response

    def get_account_balance(self):
        """获取账户余额信息"""
        return self._request('GET', '/api/v5/account/balance', authenticated=True)

    def get_usdt_equity(self):
        """获取以USDT计价的总权益"""
        data = self.get_account_balance()
        if data and data.get("code") == "0" and data.get("data"):
            # totalEq provides the total equity in USDT
            return float(data["data"][0]["totalEq"])
        return None

    def mock_order(self, action, symbol, qty):
        print(f"[模拟交易] {datetime.now()}：{action.upper()} {qty} {symbol}")

    def get_candlesticks(self, instId, bar="1H", limit=100, after=None, before=None):
        params = {"instId": instId, "bar": bar, "limit": limit}
        if after:
            params["after"] = after
        if before:
            params["before"] = before
        
        return self._request('GET', '/api/v5/market/candles', params=params)
