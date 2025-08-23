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


def get_klines(client, symbol, interval, years=1, limit=1000, save=True, keep_ts_float=False):
    """
    按年限拉取 OKX 历史K线（正向分页 + 严格窗口 + 分页日志）
    - 从 target_time 往现在，用 after 游标推进
    - 每页打印 第几页 / 条数 / 最早-最晚 / 距 now 差距
    - 严格截断到 window，以及条数护栏
    """
    import os, time
    import pandas as pd

    def normalize_interval(iv):
        m = {
            '1m':'1m','5m':'5m','15m':'15m','30m':'30m',
            '1h':'1H','4h':'4H','1d':'1D','1w':'1W'
        }
        return m.get(iv.lower(), iv)

    # 1）计算窗口
    now_utc = pd.Timestamp.utcnow().tz_localize("UTC")
    target_time = (now_utc - pd.Timedelta(days=years * 365)).ceil(interval.lower()).tz_convert("UTC")

    # 2）准备 after 游标：要拿到 >= target_time 的第一根，减 1ms
    after_ms = int((target_time - pd.Timedelta(milliseconds=1)).timestamp() * 1000)

    all_rows = []
    page_no = 1

    while True:
        params = {
            "instId": symbol,
            "bar": normalize_interval(interval),
            "limit": limit,
            "after": str(after_ms)
        }
        resp = client.get_candlesticks(**params)
        data = resp.get("data") or []
        if not data:
            print("⚠️ API 返回空或出错")
            break

        # 转时间戳
        ts = pd.to_datetime([int(r[0]) for r in data], unit="ms", utc=True)
        earliest, latest = ts.min(), ts.max()

        # 分页日志：页号 / 条数 / 范围 / 距离 now 差距
        delta_hrs = (now_utc - latest).total_seconds() / 3600
        print(f"📄 第{page_no}页: {len(data)}根, 范围 {earliest} ~ {latest}, 离 now {delta_hrs:.1f}h")

        # 收集 >= target_time 的行
        for row, t in zip(data, ts):
            if t >= target_time:
                all_rows.append(row)

        # 如果 latest 已经 >= now_utc（拉到最新），或本页条数 < limit，直接停
        if latest >= now_utc or len(data) < limit:
            print("✅ 拉取到最新或不足一页，停止")
            break

        # 推进 after 游标：本页最新时间 + 1ms
        after_ms = int((latest + pd.Timedelta(milliseconds=1)).timestamp() * 1000)
        page_no += 1
        time.sleep(0.2)

    # DataFrame 化 & 类型转换
    df = pd.DataFrame(all_rows, columns=[
        "ts","open","high","low","close","vol",
        "volCcy","volCcyQuote","confirm"
    ])
    df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms", utc=True)
    for c in ["open","high","low","close","vol"]:
        df[c] = df[c].astype(float)
    if keep_ts_float:
        df["ts_float"] = df["ts"].view("int64") / 1e9

    # 排序 + 去重 + 严格窗口截断
    df = df.sort_values("ts").drop_duplicates(["ts"]).reset_index(drop=True)
    df = df[df["ts"] >= target_time].reset_index(drop=True)

    # 条数护栏（可选）
    bars_per_day = {
        "1m":1440, "5m":288, "15m":96, "30m":48,
        "1h":24, "4h":6, "1d":1, "1w":1/7
    }
    iv = interval.lower()
    if iv in bars_per_day:
        expected = int(years * 365 * bars_per_day[iv] + 2)  # +2根缓冲
        if len(df) > expected:
            print(f"⚠️ 超出预期{expected}根，裁剪多余{len(df)-expected}根")
            df = df.tail(expected).reset_index(drop=True)

    print(f"✅ 完成获取 {len(df)}根K线 ({symbol},{interval}), 窗口从{target_time}到{now_utc}")
    if save:
        os.makedirs("data/history", exist_ok=True)
        path = f"data/history/{symbol.replace('-','')}_{interval}_{years}y.csv"
        df.to_csv(path, index=False)
        print(f"💾 已保存 {path}")

    return df


"""
def get_klines(client, symbol, interval, limit=1000, max_candles=10000, save=True):
    '''
    分页拉取OKX历史K线，自动拼接，返回DataFrame
    :param client: OKXClient
    :param symbol: 交易对，例如 "BTC-USDT"
    :param interval: K线周期，例如 "1m", "5m", "1H"
    :param limit: 单次请求数量（OKX最大1000）
    :param max_candles: 最多获取多少根K线
    :param save: 是否保存到 data/history/ 目录
    '''
    all_data = []
    end_time = None
    fetched = 0
    count =0
    while fetched < max_candles:
        '''
        resp = client.get_candlesticks(
            instId=symbol,
            bar=normalize_interval(interval),
            limit=limit,
            after=end_time
        )
        '''
        params = {
            "instId": symbol,
            "bar": normalize_interval(interval),
            "limit": 1000
        }
        if end_time:
            params["before"] = end_time  # 翻页：从上次最早的时间往前取

        resp = client.get_candlesticks(**params)

        if not resp or "data" not in resp or len(resp["data"]) == 0:
            print("⚠️ API 返回数据为空或出错")
            break

        batch = resp["data"]
        all_data.extend(batch)
        fetched += len(batch)

        # 下一次请求的结束时间（取最后一根K线的ts）
        end_time = batch[-1][0]
        time.sleep(0.2)  # 防止触发API限速

        c_len = len(batch)
        print(f"len(batch):{c_len}, count: {count} ")
        count += 1
        '''
        c_len = len(batch)
        if c_len < limit:
            print(f"len(batch):{c_len}, limit: {limit} ")
            break
       '''
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

    """
