# utils/data_normalization.py
import pandas as pd

def normalize_binance_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 Binance 原始K线数据的列名映射为统一标准:
    open, high, low, close, vol
    """
    col_map = {
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "vol",
        0: "open_time",  # 如果是列表转 DataFrame 形式
        1: "open",
        2: "high",
        3: "low",
        4: "close",
        5: "vol"
    }
    return df.rename(columns={k: col_map.get(k, k) for k in df.columns})

