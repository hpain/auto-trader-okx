# data/okx.py

import requests
import pandas as pd
import time

def fetch_ohlcv(symbol, interval='1h', limit=100):
    # 示例用OKX的公共REST API拉取历史K线
    granularity_map = {
        '1m': 60,
        '5m': 300,
        '15m': 900,
        '1h': 3600,
        '4h': 14400,
        '1d': 86400
    }

    granularity = granularity_map.get(interval)
    if granularity is None:
        raise ValueError(f"Unsupported interval: {interval}")

    url = f"https://www.okx.com/api/v5/market/candles?instId={symbol}&bar={interval}&limit={limit}"
    resp = requests.get(url)
    data = resp.json()

    if 'data' not in data:
        raise ValueError("API response error:", data)

    # OKX返回的顺序是[时间, 开盘, 最高, 最低, 收盘, 成交量, ...]
    df = pd.DataFrame(data['data'], columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df = df.iloc[::-1]  # 翻转为时间升序

    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

    df = df.rename(columns={'ts': 'timestamp'})
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

