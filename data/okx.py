# data/okx.py

import requests
import pandas as pd
import time

'''
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

    #print(f"📡 正在连接 OKX API 获取 {symbol} 的 K线数据...")

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

'''

def fetch_ohlcv(symbol, interval='1H', limit=50):
    print(f"📡 正在连接 OKX API 获取 {symbol} 的 K线数据...")

    url = f"https://www.okx.com/api/v5/market/candles"
    params = {
        'instId': symbol,
        'bar': interval,
        'limit': limit
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # HTTP错误抛出异常
        data = response.json()

        if 'data' not in data or not data['data']:
            print("⚠️ API 返回数据为空")
            return None

        # 解析为DataFrame
        df = pd.DataFrame(data['data'], columns=['ts', 'open', 'high', 'low', 'close', 'volume', '_1', '_2'])
        df = df.iloc[::-1]  # 翻转为时间升序

        df['timestamp'] = pd.to_datetime(df['ts'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        print(f"✅ 成功获取数据，共 {len(df)} 条记录。")
        print(df.tail(3))  # 打印最后几行以确认数据
        return df

    except requests.exceptions.RequestException as e:
        print(f"❌ 网络请求失败：{e}")
        return None

    except Exception as e:
        print(f"❌ 数据解析或转换失败：{e}")
        return None

