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

def get_klines(client, symbol, interval, years=1, limit=300, save=True, keep_ts_float=False, max_pages=5000):
    """
    稳健抓取 OKX 历史 K 线（分页+去重+防死循环+按时间正序）
    - 第 1 页使用 /market/candles
    - 后续使用 /market/history-candles
    - 每页 before = 上页最旧 K 线时间戳 - 1ms
    - 命中目标时间后截断
    - 支持保存 CSV
    """
    import os, time, requests
    import pandas as pd

    # 规范化 interval
    def normalize_interval(iv):
        m = {
            '1m':'1m','5m':'5m','15m':'15m','30m':'30m',
            '1h':'1H','4h':'4H','1d':'1D','1w':'1W'
        }
        return m.get(iv.lower(), iv)

    bar = normalize_interval(interval)

    # 目标时间
    target_time = pd.Timestamp.utcnow() - pd.Timedelta(days=years*365)
    if target_time.tzinfo is None:
        target_time = target_time.tz_localize('UTC')
    else:
        target_time = target_time.tz_convert('UTC')

    # 保存路径
    cache_dir = "data/history"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = f"{cache_dir}/{symbol.replace('-','')}_{interval}_{years}y.csv"

    session = requests.Session()
    all_rows = []
    before_ts = None
    page_no = 1
    last_oldest_ts = None

    while page_no <= max_pages:
        # 第 1 页用 candles，后续用 history-candles
        endpoint = "/api/v5/market/candles" if page_no == 1 and before_ts is None else "/api/v5/market/history-candles"
        url = f"{client.base_url}{endpoint}"
        params = {"instId": symbol, "bar": bar, "limit": limit}
        if before_ts is not None:
            params["before"] = str(int(before_ts))

        try:
            r = session.get(url, params=params, timeout=15)
            r.raise_for_status()
            resp = r.json()
        except Exception as e:
            print(f"⚠️ 请求异常: {type(e).__name__}: {e}")
            break

        if resp.get("code") not in ("0", 0, None) or not resp.get("data"):
            print(f"⚠️ API 返回空/错误: {resp}")
            break

        batch = resp["data"]
        if not batch:
            break

        # OKX 返回倒序（最新→最旧）
        ts_arr = pd.to_datetime([int(row[0]) for row in batch], unit='ms', utc=True)
        newest, oldest = ts_arr.max(), ts_arr.min()
        diff_h = (oldest - target_time).total_seconds() / 3600
        print(f"📄 第 {page_no} 页 via {endpoint.split('/')[-1]}: {len(batch)} 根, 最旧 {oldest}, 最新 {newest}, 距目标 {diff_h:.1f} 小时")

        # 获取当前页最旧 K 线原始毫秒时间戳
        cur_oldest_ts = int(batch[-1][0])

        # 游标前进保护
        if last_oldest_ts is not None and cur_oldest_ts >= last_oldest_ts:
            # 微调游标避免重复
            cur_oldest_ts = last_oldest_ts - 1
        last_oldest_ts = cur_oldest_ts

        # 命中目标时间 → 截断
        batch = [row for row in batch if pd.to_datetime(int(row[0]), unit="ms", utc=True) >= target_time]
        all_rows.extend(batch)

        if oldest <= target_time:
            print(f"✅ 命中目标时间，已收集到 {target_time.date()} 及之后数据")
            break

        # 下一页游标
        before_ts = cur_oldest_ts
        page_no += 1
        time.sleep(0.18)  # 适度限速

    if not all_rows:
        print("❌ 没有抓到任何 K 线")
        return pd.DataFrame(columns=["ts","open","high","low","close","vol"])

    # DataFrame 化
    df = pd.DataFrame(all_rows)
    df.columns = ["ts","open","high","low","close","vol","volCcy","volCcyQuote","confirm"][:df.shape[1]]
    df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms", utc=True)
    for col in ["open","high","low","close","vol"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=["open","high","low","close"]).sort_values("ts").drop_duplicates(subset=["ts"]).reset_index(drop=True)
    df = df[df["ts"] >= target_time].reset_index(drop=True)

    # 数量护栏
    bars_per_day = {"1m":1440,"5m":288,"15m":96,"30m":48,"1h":24,"4h":6,"1d":1,"1w":1/7}
    if interval.lower() in bars_per_day:
        expected = int(years*365*bars_per_day[interval.lower()] + 96)  # 留冗余
        if len(df) > expected:
            df = df.tail(expected).reset_index(drop=True)

    if keep_ts_float:
        df["ts_float"] = df["ts"].view("int64")/1e9

    print(f"✅ 成功获取 {len(df)} 根 K 线 ({symbol}, {interval}, {years}y)")
    if save:
        df.to_csv(cache_path, index=False)
        print(f"💾 已保存到 {cache_path}")

    return df

import os
import time
import requests
import pandas as pd

def get_klines_bian(client, symbol, interval, years=1, limit=1000,
                    save=True, keep_ts_float=False, max_pages=5000,
                    ignore_local=False):
    """
    稳健抓取 Binance 历史 K 线（分页 + 去重 + 防死循环 + 合并本地历史）
    参数保持与 OKX 版本一致，可无缝替换。
    """
    import os, time, pandas as pd, requests

    API_URL = "https://api.binance.com/api/v3/klines"
    symbol = symbol.replace("-", "")  # Binance 不接受中横线
    columns = ["ts", "o", "h", "l", "c", "v", "ct", "qv", "tbuv", "tqav", "trades", "ignore"]

    # 目标起始时间（UTC）
    target_time = pd.Timestamp.utcnow() - pd.Timedelta(days=years * 365)
    target_time = (target_time.tz_localize("UTC")
                   if target_time.tzinfo is None
                   else target_time.tz_convert("UTC"))

    # 保存路径
    cache_dir = "data/history"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = f"{cache_dir}/binance_{symbol}_{interval}_{years}y.csv"

    # ===== 先尝试加载本地历史 =====
    if not ignore_local and os.path.exists(cache_path):
        try:
            df_local = pd.read_csv(cache_path, parse_dates=["ts"])
            print(f"检测到本地历史: {len(df_local)} 条, 时间范围 {df_local['ts'].min()} → {df_local['ts'].max()}")
        except Exception as e:
            print(f"⚠️ 本地文件读取失败: {e}")
            df_local = pd.DataFrame(columns=["ts"])
    else:
        if ignore_local:
            print("⚠️ 已启用 ignore_local，忽略本地 CSV，直接全量抓取")
        df_local = pd.DataFrame(columns=["ts"])

    # ===== 开始抓取（断点续传或全量）=====
    fetched_timestamps = set(int(x.timestamp() * 1000) for x in df_local['ts']) if not df_local.empty else set()
    before_param = None
    total_fetched = 0
    all_rows = []

    if not df_local.empty and not ignore_local:
        # 从最早时间往前拉补缺
        before_param = int(df_local['ts'].min().timestamp() * 1000) - 1

    session = requests.Session()
    page_no = 1

    while page_no <= max_pages:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if before_param is not None:
            params["endTime"] = before_param

        try:
            r = session.get(API_URL, params=params, timeout=15)
            r.raise_for_status()
            batch = r.json()
        except Exception as e:
            print(f"❌ 请求失败: {e}")
            break

        if not batch:
            print("✅ 数据获取完毕")
            break

        # 去重
        new_rows = []
        for row in batch:
            ts = int(row[0])
            if ts not in fetched_timestamps:
                new_rows.append(row)
                fetched_timestamps.add(ts)

        if not new_rows:
            print("⚠️ 本批全重复，结束")
            break

        # 转 DataFrame（批次降序）
        df_new = pd.DataFrame(new_rows, columns=columns)
        df_new['ts'] = pd.to_datetime(df_new['ts'], unit='ms', utc=True)
        df_new = df_new.sort_values("ts", ascending=False)

        newest, oldest = df_new['ts'].max(), df_new['ts'].min()
        print(f"📥 第 {page_no} 页: {len(df_new)} 条, 最新: {newest}, 最旧: {oldest}")

        # 命中目标时间
        if oldest <= target_time:
            df_new = df_new[df_new['ts'] >= target_time]
            all_rows.extend(df_new.values.tolist())
            total_fetched += len(df_new)
            print(f"✅ 命中目标时间，收集完成 ({len(df_new)} 条)")
            break

        all_rows.extend(df_new.values.tolist())
        total_fetched += len(df_new)
        before_param = int(oldest.timestamp() * 1000) - 1
        page_no += 1
        time.sleep(0.45)

    # ===== 合并历史与新数据 =====
    df_new_all = pd.DataFrame(all_rows, columns=columns)
    df_new_all['ts'] = pd.to_datetime(df_new_all['ts'], unit='ms', utc=True)

    if not df_local.empty and not ignore_local:
        df = pd.concat([df_local, df_new_all], ignore_index=True)
    else:
        df = df_new_all

    # 清洗
    df = df.sort_values("ts").drop_duplicates(subset=["ts"]).reset_index(drop=True)
    for col in ["o", "h", "l", "c", "v"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=["o", "h", "l", "c"])

    if not keep_ts_float and "ct" in df.columns:
        df = df.drop(columns=['ct', 'qv', 'tbuv', 'tqav', 'trades', 'ignore'], errors='ignore')
    elif keep_ts_float:
        df['ts_float'] = df['ts'].view('int64') / 1e9

    if save:
        df.to_csv(cache_path, index=False)
        print(f"💾 已保存到 {cache_path}")

    print(f"🏁 返回 {len(df)} 条有效K线 ({symbol}, {interval}, {years}y)")
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
