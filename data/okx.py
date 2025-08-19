# data/okx.py

import requests
import pandas as pd
import time
import os

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

def normalize_interval(interval):
    # 自动将常见小写写法转换为OKX支持的格式
    mapping = {
        '1h': '1H',
        '4h': '4H',
        '1d': '1D',
        '1w': '1W',
        '1m': '1m',
        '5m': '5m',
        '15m': '15m',
        '30m': '30m',
    }
    return mapping.get(interval.lower(), interval)
    
def fetch_ohlcv(symbol, interval='1h', limit=50):
    print(f"📡 正在连接 OKX API 获取 {symbol} 的 K线数据...")

    url = f"https://www.okx.com/api/v5/market/candles"
    params = {
        'instId': symbol,
        'bar': normalize_interval(interval),
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
        #df = pd.DataFrame(data['data'], columns=['ts', 'open', 'high', 'low', 'close', 'volume', '_1', '_2'])
        # 原始数据 DataFrame（不指定列名）
        df = pd.DataFrame(data['data'])
        df = df.iloc[::-1]  # 翻转为时间升序

        # 👉 打印列数和列内容
        print(f"✅ 获取到的原始数据共有 {df.shape[1]} 列")
        print("📋 列名示意：", [f"Column {i}" for i in range(df.shape[1])])
        print("📊 原始样本行：", df.iloc[0].to_list())

        # 只取前6列为 ['ts', 'open', 'high', 'low', 'close', 'volume']
        df = df.iloc[:, :6]
        df.columns = ['ts', 'open', 'high', 'low', 'close', 'volume']
        
        
        
        df['timestamp'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        print("📊 数据预处理成功，最后几条：")
        print(df.tail(3))

         
        '''
        

        df['timestamp'] = pd.to_datetime(df['ts'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        print(f"✅ 成功获取数据，共 {len(df)} 条记录。")
        print(df.tail(3))  # 打印最后几行以确认数据
        '''
        return df

    except requests.exceptions.RequestException as e:
        print(f"❌ 网络请求失败：{e}")
        return None

    except Exception as e:
        print(f"❌ 数据解析或转换失败：{e}")
        return None


def get_klines(client, symbol, interval, limit=1000, max_candles=10000, save=True):
    """
    分页拉取OKX历史K线，自动拼接，返回DataFrame
    :param client: OKXClient
    :param symbol: 交易对，例如 "BTC-USDT"
    :param interval: K线周期，例如 "1m", "5m", "1H"
    :param limit: 单次请求数量（OKX最大1000）
    :param max_candles: 最多获取多少根K线
    :param save: 是否保存到 data/history/ 目录
    """
    all_data = []
    end_time = None
    fetched = 0

    while fetched < max_candles:
        resp = client.get_candlesticks(
            instId=symbol,
            bar=interval,
            limit=limit,
            after=end_time
        )

        if not resp or "data" not in resp or len(resp["data"]) == 0:
            print("⚠️ API 返回数据为空或出错")
            break

        batch = resp["data"]
        all_data.extend(batch)
        fetched += len(batch)

        # 下一次请求的结束时间（取最后一根K线的ts）
        end_time = batch[-1][0]
        time.sleep(0.2)  # 防止触发API限速

        if len(batch) < limit:
            break

    # 转换为 DataFrame
    df = pd.DataFrame(all_data, columns=[
        "ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"
    ])

    # 时间戳转换
    df["ts"] = pd.to_datetime(df["ts"].astype(float), unit="ms")
    df = df.sort_values("ts").reset_index(drop=True)

    # 只保留常用字段
    #df = df[["ts", "open", "high", "low", "close", "vol"]].astype(float)
    df = df[["ts", "open", "high", "low", "close", "vol"]].copy()

    # 转换数值列
    for col in ["open", "high", "low", "close", "vol"]:
       df[col] = df[col].astype(float)

    # 确保时间列为 datetime
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")

    # 按时间升序
    df = df.sort_values("ts").reset_index(drop=True)


    print(f"✅ 成功获取 {len(df)} 根K线数据 ({symbol}, {interval})")

    # 保存数据
    if save:
        os.makedirs("data/history", exist_ok=True)
        path = f"data/history/{symbol.replace('-', '')}_{interval}.csv"
        df.to_csv(path, index=False)
        print(f"💾 已保存到 {path}")

    return df
