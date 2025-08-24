# utils/data_normalization.py
import pandas as pd

def normalize_binance_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 Binance 原始K线数据标准化为特征工程可用格式:
    - 列名统一为 open, high, low, close, vol
    - 数值列转换为 float
    """
    # 列名映射：兼容 API 返回的短名或数字索引
    col_map = {
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "vol",
        0: "open_time",
        1: "open",
        2: "high",
        3: "low",
        4: "close",
        5: "vol"
    }
    df = df.rename(columns={k: col_map.get(k, k) for k in df.columns})

    # 将关键数值列转为 float，忽略无法转换的值
    num_cols = ["open", "high", "low", "close", "vol"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


