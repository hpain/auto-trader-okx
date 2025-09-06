# data/binance.py
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
    interval = interval.lower()
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
    if not df_new_all.empty:
        df_new_all['ts'] = pd.to_datetime(df_new_all['ts'], unit='ms', utc=True)

    if not df_local.empty and not ignore_local:
        df = pd.concat([df_local, df_new_all], ignore_index=True)
    else:
        df = df_new_all

    # 清洗
    if not df.empty:
        df = df.sort_values("ts").drop_duplicates(subset=["ts"]).reset_index(drop=True)
        for col in ["o", "h", "l", "c", "v"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=["o", "h", "l", "c"])

        if not keep_ts_float and "ct" in df.columns:
            df = df.drop(columns=['ct', 'qv', 'tbuv', 'tqav', 'trades', 'ignore'], errors='ignore')
        elif keep_ts_float:
            df['ts_float'] = df['ts'].view('int64') / 1e9

    if save and not df.empty:
        df.to_csv(cache_path, index=False)
        print(f"💾 已保存到 {cache_path}")

    print(f"🏁 返回 {len(df)} 条有效K线 ({symbol}, {interval}, {years}y)")
    return df
