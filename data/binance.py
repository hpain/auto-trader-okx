# data/binance.py
import os
import time
import requests
import pandas as pd
from config import config
from utils.data_normalization import normalize_binance_df

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
    cache_dir = config["paths"]["history_data_dir"]
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = f"{cache_dir}/binance_{symbol}_{interval}_{years}y.csv"

    # ===== 先尝试加载本地历史 =====
    if not ignore_local and os.path.exists(cache_path):
        try:
            df_local = pd.read_csv(cache_path, parse_dates=["ts"])
            # attempt to normalize if columns use short names
            if 'o' in df_local.columns:
                df_local = normalize_binance_df(df_local)
            print(f"检测到本地历史: {len(df_local)} 条, 时间范围 {df_local['ts'].min()} → {df_local['ts'].max()}")
        except Exception as e:
            print(f"WARN 本地文件读取失败: {e}")
            df_local = pd.DataFrame(columns=["ts"])
    else:
        if ignore_local:
            print("WARN 已启用 ignore_local，忽略本地 CSV，直接全量抓取")
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
            print(f"FAIL 请求失败: {e}")
            break

        if not batch:
            print("OK 数据获取完毕")
            break

        # 去重
        new_rows = []
        for row in batch:
            ts = int(row[0])
            if ts not in fetched_timestamps:
                new_rows.append(row)
                fetched_timestamps.add(ts)

        if not new_rows:
            print("WARN 本批全重复，结束")
            break

        # 转 DataFrame（批次降序）
        df_new = pd.DataFrame(new_rows, columns=columns)
        df_new['ts'] = pd.to_datetime(df_new['ts'], unit='ms', utc=True)
        df_new = df_new.sort_values("ts", ascending=False)

        newest, oldest = df_new['ts'].max(), df_new['ts'].min()
        print(f"OK 第 {page_no} 页: {len(df_new)} 条, 最新: {newest}, 最旧: {oldest}")

        all_rows.extend(df_new.values.tolist())
        total_fetched += len(df_new)

        # 命中目标时间
        if oldest <= target_time:
            print(f"OK 命中目标时间，停止抓取")
            break

        before_param = int(oldest.timestamp() * 1000) - 1
        page_no += 1
        time.sleep(0.45)

    # ===== 合并历史与新数据 =====
    df_new_all = pd.DataFrame(all_rows, columns=columns)
    if not df_new_all.empty:
        df_new_all['ts'] = pd.to_datetime(df_new_all['ts'], unit='ms', utc=True)

    # --- 修复后的合并与清洗逻辑 ---
    # 如果忽略本地缓存，则最终数据就是新抓取的数据
    if ignore_local:
        df_final = df_new_all
    # 否则，合并本地数据和新抓取的数据
    else:
        dfs_to_concat = [df for df in [df_local, df_new_all] if not df.empty]
        if dfs_to_concat:
            df_final = pd.concat(dfs_to_concat, ignore_index=True)
        else:
            df_final = pd.DataFrame()

    # 清洗
    if not df_final.empty:
        # --- 关键修复：在设置索引前，先排序并彻底去重 ---
        # 1. 确保 'ts' 列是 datetime 类型
        df_final['ts'] = pd.to_datetime(df_final['ts'])
        # 2. 排序并基于 'ts' 列移除重复行，保留第一个出现的记录
        df_final = df_final.sort_values("ts").drop_duplicates(subset=["ts"], keep='first')
        # 3. 筛选出目标时间范围内的数据
        df_final = df_final[df_final['ts'] >= target_time].reset_index(drop=True)
        for col in ["o", "h", "l", "c", "v"]:
            if col in df_final.columns:
                df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
        
        required_cols = ["o", "h", "l", "c"]
        if all(col in df_final.columns for col in required_cols):
            df_final = df_final.dropna(subset=required_cols)
        else:
            print("WARN: Price data is missing required columns (o, h, l, c).")
            df_final = pd.DataFrame() # Make df empty

    # 在返回之前，删除不需要的列
    if not df_final.empty and not keep_ts_float and "ct" in df_final.columns:
        df_final = df_final.drop(columns=['ct', 'qv', 'tbuv', 'tqav', 'trades', 'ignore'], errors='ignore')

    # --- 关键修复：在返回前统一进行列名归一化 ---
    if not df_final.empty and 'o' in df_final.columns:
        print("Normalizing column names before returning...")
        df_final = normalize_binance_df(df_final)

    if save and not df_final.empty:
        # 保存完整、干净的数据
        df_to_save = df_final.copy()
        if 'ts_float' in df_to_save.columns:
             df_to_save = df_to_save.drop(columns=['ts_float'])
        df_to_save.to_csv(cache_path, index=False)
        print(f"SAVE 已保存到 {cache_path}")

    print(f"DONE 返回 {len(df_final)} 条有效K线 ({symbol}, {interval}, {years}y)")

    # --- 修复: 在返回前将 'ts' 列设置为主索引 ---
    # 下游的特征工程函数需要一个 DatetimeIndex。
    if not df_final.empty and 'ts' in df_final.columns:
        df_final = df_final.set_index('ts')

    return df_final
