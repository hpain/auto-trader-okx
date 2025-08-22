import requests, time
import pandas as pd
import os
from datetime import datetime

class OKXClient:
    def __init__(self, api_key, secret_key, passphrase, flag):
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.base_url = "https://www.okx.com" if flag == "1" else "https://www.okx.com"

    def get_kline(self, symbol="BTC-USDT", interval="1h", limit=100):
        VALID_BARS = {"1m", "3m", "5m", "15m", "30m", "1H", "2H", "4H", "6H", "12H", "1D", "1W", "1M"}
        if interval not in VALID_BARS:
           raise ValueError(f"无效的 interval 参数: {interval}, 必须是 {VALID_BARS}")

        url = f"{self.base_url}/api/v5/market/candles?instId={symbol}&bar={interval}&limit={limit}"
        resp = requests.get(url)
        data = resp.json()['data']
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "vol", "vol_ccy", "vol_ccy_quote", "confirm", "ts"])
        df["close"] = df["close"].astype(float)
        return df[::-1]

    def mock_order(self, action, symbol, qty):
        print(f"[模拟交易] {datetime.now()}：{action.upper()} {qty} {symbol}")

    def get_candlesticks(self, instId, bar="1H", limit=100, after=None, before=None):
        url = f"{self.base_url}/api/v5/market/candles"
        print(f"url:  {url}")
        params = {"instId": instId, "bar": bar, "limit": limit}
        if after:
            params["after"] = after
        if before:
            params["before"] = before

        #print(f"params: {params}")
        resp = requests.get(url, params=params)
        print(f"resp: {resp}")
        if resp.status_code != 200:
            print(f"❌ 请求失败: {resp.status_code}, {resp.text}")
            return None
        return resp.json()

        
    def fetch_historical_klines(self, instId, bar="1H", limit=100, max_pages=50, save_path="data/history"):
        """
        分页拉取历史K线，并缓存到CSV
        :param instId: 交易对，比如 "BTC-USDT"
        :param bar: 时间周期
        :param limit: 每次请求的数据量 (最大100)
        :param max_pages: 拉取多少页
        :param save_path: CSV 保存路径
        """
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, f"{instId.replace('-', '_')}_{bar}.csv")

        all_data = []

        # 如果本地有缓存，先加载
        if os.path.exists(file_path):
            print(f"📂 发现本地缓存 {file_path}，正在加载...")
            df = pd.read_csv(file_path, parse_dates=["ts"])
            all_data.extend(df.values.tolist())
            last_timestamp = int(df["ts"].iloc[-1].timestamp() * 1000)  # 转毫秒
        else:
            last_timestamp = None

        print("⏳ 正在从OKX拉取历史数据...")
        for i in range(max_pages):
            data = self.get_candlesticks(instId, bar=bar, limit=limit, before=last_timestamp)
            if not data or "data" not in data or len(data["data"]) == 0:
                print("✅ 已经拉取到尽头")
                break

            rows = data["data"]
            all_data.extend(rows)
            last_timestamp = int(rows[-1][0])  # 下一页起点
            time.sleep(0.2)  # 避免触发API限流

        # 转成DataFrame
        df = pd.DataFrame(all_data, columns=[
            "ts", "o", "h", "l", "c", "vol", "volCcy", "volCcyQuote", "confirm"
        ])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)

        # 保存到CSV
        df.to_csv(file_path, index=False)
        print(f"✅ 历史数据已保存到 {file_path}, 共 {len(df)} 条")
        return df
