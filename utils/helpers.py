import pandas as pd
import numpy as np

def check_data_length(df: pd.DataFrame, min_length: int = 200) -> bool:
    """
    检查输入数据是否满足最小长度要求
    """
    if df is None or df.empty:
        return False
    return len(df) >= min_length

def handle_insufficient_data(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    处理数据长度不足的情况，返回一个填充了NaN的DataFrame
    """
    if df is None or df.empty:
        # 如果原始df为空，创建一个空的DataFrame，但包含所需的列
        # 这里我们无法确定索引，所以只能返回一个空的带列名的DataFrame
        result = pd.DataFrame(columns=feature_names)
        return result
        
    result = pd.DataFrame(index=df.index)
    for feature in feature_names:
        result[feature] = np.nan
    return result
