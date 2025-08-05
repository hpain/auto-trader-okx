import pandas as pd

def generate_signal(df):
    if df is None or len(df) < 20:
        # 数据本身就不够长
        return 'hold'

    df['ma_short'] = df['close'].rolling(window=5).mean()
    df['ma_long'] = df['close'].rolling(window=20).mean()

    # 删除所有包含 NaN 的行
    df = df.dropna()

    if len(df) < 2:
        # 删除 NaN 后行数还不够
        return 'hold'

    # 提取最后两行用于比较
    prev = df.iloc[-2]
    curr = df.iloc[-1]

    if prev['ma_short'] < prev['ma_long'] and curr['ma_short'] > curr['ma_long']:
        return 'buy'
    elif prev['ma_short'] > prev['ma_long'] and curr['ma_short'] < curr['ma_long']:
        return 'sell'
    else:
        return 'hold'
