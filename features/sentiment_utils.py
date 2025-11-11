"""
情感分析相关的工具函数
"""
import pandas as pd

def merge_price_and_sentiment(price_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    合并价格数据和情感数据，确保正确对齐并处理边缘情况。

    Args:
        price_df: 价格数据DataFrame
        sentiment_df: 情感数据DataFrame

    Returns:
        合并后的DataFrame
    """
    # 检查输入数据
    if sentiment_df is None or sentiment_df.empty:
        result = price_df.copy()
        result['sent_mean'] = 0.5
        result['sent_median'] = 0.5
        result['count'] = 0
        return result

    # 确保索引是时间戳类型
    if not isinstance(sentiment_df.index, pd.DatetimeIndex):
        sentiment_df.index = pd.to_datetime(sentiment_df.index)
    if not isinstance(price_df.index, pd.DatetimeIndex):
        price_df.index = pd.to_datetime(price_df.index)

    # 合并数据
    result = pd.merge(
        price_df,
        sentiment_df,
        left_index=True,
        right_index=True,
        how='left'
    )

    # 使用前向填充处理缺失值，然后用中性值（0.5）填充剩余的NaN
    for col in ['sent_mean', 'sent_median']:
        if col in result.columns:
            result[col] = result[col].fillna(method='ffill').fillna(0.5)
    
    # 计数列使用0填充
    if 'count' in result.columns:
        result['count'] = result['count'].fillna(0)

    return result