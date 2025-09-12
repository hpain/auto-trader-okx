# config/__init__.py
import os
import yaml

def _load_config():
    """
    Loads configuration from settings.yaml and overrides with environment variables.
    """
    _here = os.path.dirname(__file__)
    _yaml_path = os.path.join(_here, "settings.yaml")

    if not os.path.exists(_yaml_path):
        # In a Docker environment, the file might not exist, which is fine.
        # We can rely solely on environment variables.
        config = {
            "okx": {},
            "trade": {},
            "paths": {
                "model_dir": "models",
                "feature_cache_dir": "data/cache",
                "history_data_dir": "data/history"
            }
        }
    else:
        with open(_yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    # --- Override with Environment Variables ---
    # OKX settings
    config["okx"]["api_key"] = os.getenv("OKX_API_KEY", config.get("okx", {}).get("api_key"))
    config["okx"]["secret_key"] = os.getenv("OKX_SECRET_KEY", config.get("okx", {}).get("secret_key"))
    config["okx"]["passphrase"] = os.getenv("OKX_PASSPHRASE", config.get("okx", {}).get("passphrase"))
    config["okx"]["flag"] = os.getenv("OKX_FLAG", config.get("okx", {}).get("flag", "0"))

    # Trade settings
    config["trade"]["symbol"] = os.getenv("TRADE_SYMBOL", config.get("trade", {}).get("symbol", "BTC-USDT"))
    config["trade"]["interval"] = os.getenv("TRADE_INTERVAL", config.get("trade", {}).get("interval", "1H"))
    
    # Ensure quantity is parsed as a float
    trade_quantity_str = os.getenv("TRADE_QUANTITY")
    if trade_quantity_str:
        config["trade"]["quantity"] = float(trade_quantity_str)
    elif "quantity" not in config.get("trade", {}):
        config["trade"]["quantity"] = 0.001


    return config

config = _load_config()

__all__ = ["config"]
