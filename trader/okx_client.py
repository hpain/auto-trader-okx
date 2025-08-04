import requests, time
import pandas as pd
from datetime import datetime

class OKXClient:
    def __init__(self, api_key, secret_key, passphrase, flag):
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.base_url = "https://www.okx.com" if flag == "1" else "https://www.okx.com"

    def get_kline(self, symbol="BTC-USDT", interval="1h", limit=100):
        url = f"{self.base_url}/api/v5/market/candles?instId={symbol}&bar={interval}&limit={limit}"
        resp = requests.get(url)
        data = resp.json()['data']
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "vol", "vol_ccy", "vol_ccy_quote", "confirm", "ts"])
        df["close"] = df["close"].astype(float)
        return df[::-1]

    def mock_order(self, action, symbol, qty):
        print(f"[模拟交易] {datetime.now()}：{action.upper()} {qty} {symbol}")