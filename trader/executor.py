from strategies.moving_average import generate_signal
from trader.okx_client import OKXClient
import yaml

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