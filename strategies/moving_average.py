import pandas as pd

def generate_signal(df: pd.DataFrame, short=7, long=25):
    df['ma_short'] = df['close'].rolling(short).mean()
    df['ma_long'] = df['close'].rolling(long).mean()
    if df['ma_short'].iloc[-2] < df['ma_long'].iloc[-2] and df['ma_short'].iloc[-1] > df['ma_long'].iloc[-1]:
        return "buy"
    elif df['ma_short'].iloc[-2] > df['ma_long'].iloc[-2] and df['ma_short'].iloc[-1] < df['ma_long'].iloc[-1]:
        return "sell"
    else:
        return "hold"