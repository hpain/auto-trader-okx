# data/okx.py
import pandas as pd
import time

def get_klines(client, symbol, interval, limit=100):
    """
    获取K线数据
    :param client: OKXClient 实例
    :param symbol: 交易对
    :param interval: K线周期
    :param limit: 数据条数
    :return: DataFrame
    """
    print(f"正在连接 OKX API 获取 {symbol} 的 K线数据...")
    df = client.get_kline(symbol=symbol, interval=interval, limit=limit)
    if df is not None and not df.empty:
        print(f"成功获取 {len(df)} 条K线数据")
    else:
        print("未获取到K线数据")
    return df

def get_historical_klines(client, symbol, interval, years=1):
    """
    获取指定年份的历史K线数据（分页）
    """
    all_data = []
    end_time = None
    limit = 100  # OKX API每页最多100条
    total_candles = years * 365 * 24 * (60 / int(interval.replace('H', '').replace('m', '')))
    fetched = 0

    while fetched < total_candles:
        data = client.get_candlesticks(
            instId=symbol,
            bar=interval,
            limit=limit,
            after=end_time
        )
        if not data or "data" not in data or len(data["data"]) == 0:
            break

        batch = data["data"]
        all_data.extend(batch)
        fetched += len(batch)
        end_time = batch[-1][0]
        time.sleep(0.2)

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "vol", "vol_ccy", "vol_ccy_quote", "confirm"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "vol"]:
       df[col] = df[col].astype(float)

    return df
