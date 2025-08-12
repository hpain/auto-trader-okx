from strategies.moving_average import generate_signal, plot_moving_averages
from trader.okx_client import OKXClient
import yaml
'''
def run():
    with open("config/settings.yaml", "r") as f:
        config = yaml.safe_load(f)
    client = OKXClient(**config["okx"])
    df = client.get_kline(config["trade"]["symbol"], config["trade"]["interval"])
    signal = generate_signal(df)

    if signal in ["buy", "sell"]:
        client.mock_order(signal, config["trade"]["symbol"], config["trade"]["order_size"])
    else:
        print("暂无交易信号")

'''

from data.okx import fetch_ohlcv  # 假设你在这里获取数据

def run():
    from config import config
    from data.okx import get_klines

    client = OKXClient(**config["okx"])
    symbol = config["trade"]["symbol"]
    interval = config["trade"]["interval"]
    df = get_klines(client, symbol, interval)
    
    if df is None or df.empty:
        print("❌ API 返回数据为空，终止运行")
        return
    
    # 计算均线
    df['ma_short'] = df['close'].rolling(window=5).mean()
    df['ma_long'] = df['close'].rolling(window=20).mean()

    # 生成交易信号
    signal = generate_signal(df)
    print(f"📈 生成交易信号: {signal}")

    # 画图保存
    plot_moving_averages(df, symbol)
    '''
    # ✅ 打印数据结构和后几行
    print("\n📊 市场数据预览：")
    print(df.tail(5))           # 打印后5行
    print("\n数据总行数:", len(df))
    print("字段列表:", df.columns.tolist())

   
'''
