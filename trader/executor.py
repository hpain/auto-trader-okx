from strategies.moving_average import generate_signal
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
# trader/executor.py
from strategies.moving_average import generate_signal
from data.okx import fetch_ohlcv  # 假设你在这里获取数据

def run():
    df = fetch_ohlcv("BTC-USDT", interval="1h", limit=50)

    # ✅ 打印数据结构和后几行
    print("\n📊 市场数据预览：")
    print(df.tail(5))           # 打印后5行
    print("\n数据总行数:", len(df))
    print("字段列表:", df.columns.tolist())

    signal = generate_signal(df)
    print("📈 生成交易信号:", signal)
