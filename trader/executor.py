# trader/executor.py
import json
import os
import logging
from datetime import datetime
import pandas as pd
from joblib import load

from config import config
from data.okx import get_klines
from strategies.moving_average import generate_signal as ma_signal, plot_moving_averages
from trader.okx_client import OKXClient
from utils.logger import setup_logger

def _write_status(status_data: dict):
    """Write the status dictionary to a JSON file."""
    try:
        # Ensure all values are serializable, especially datetime objects
        for key, value in status_data.items():
            if isinstance(value, (pd.Timestamp, datetime)):
                status_data[key] = value.isoformat()

        with open("status.json", "w", encoding="utf-8") as f:
            json.dump(status_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logging.error(f"Failed to write status.json file: {e}", exc_info=True)

def _load_model():
    try:
        model_dir = config["paths"]["model_dir"]
        model_path = os.path.join(model_dir, "best_model.pkl")
        meta_path = os.path.join(model_dir, "metadata.json")
        pipe = load(model_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return pipe, meta
    except FileNotFoundError:
        return None, None
    except Exception:
        logging.error("Failed to load model due to an unexpected error:", exc_info=True)
        return None, None

def calculate_quantity(client: OKXClient, symbol: str, df: pd.DataFrame) -> float:
    """Calculate trade quantity based on the configured strategy."""
    strategy = config.get("trade", {}).get("position_sizing_strategy", "fixed")
    
    if strategy == "fixed":
        quantity = config.get("trade", {}).get("quantity", 0.001)
        logging.info(f"Using fixed position sizing. Quantity: {quantity}")
        return quantity

    if strategy == "fractional":
        risk_per_trade = config.get("trade", {}).get("risk_per_trade", 0.01)
        stop_loss_pct = config.get("trade", {}).get("stop_loss_pct", 0.02)
        
        equity = client.get_usdt_equity()
        if equity is None:
            logging.error("Could not retrieve equity. Falling back to fixed quantity.")
            return config.get("trade", {}).get("quantity", 0.001)

        current_price = df['close'].iloc[-1]
        stop_loss_price = current_price * (1 - stop_loss_pct)
        risk_per_unit = current_price - stop_loss_price
        
        if risk_per_unit <= 0:
            logging.warning("Risk per unit is zero or negative. Cannot calculate fractional quantity.")
            return 0.0

        amount_to_risk = equity * risk_per_trade
        quantity = amount_to_risk / risk_per_unit
        
        logging.info(f"Equity: ${equity:.2f}, Risk: {risk_per_trade:.2%}, Price: ${current_price:.2f}, SL: ${stop_loss_price:.2f}")
        logging.info(f"Calculated fractional quantity: {quantity:.6f}")
        return quantity

    logging.warning(f"Unknown position sizing strategy: '{strategy}'. Defaulting to fixed.")
    return config.get("trade", {}).get("quantity", 0.001)

def run():
    setup_logger()
    status_report = {
        "last_update": datetime.utcnow().isoformat(),
        "signal": "N/A",
        "outcome": "Cycle starting",
        "account_equity": "N/A"
    }

    client = OKXClient(
        api_key=config["okx"]["api_key"],
        secret_key=config["okx"]["secret_key"],
        passphrase=config["okx"]["passphrase"],
        flag=config["okx"]["flag"],
    )

    try:
        symbol = config.get("trade", {}).get("symbol", "BTC-USDT")
        interval = config.get("trade", {}).get("interval", "1H")

        df = get_klines(client, symbol, interval, limit=400)
        if df is None or df.empty:
            logging.error("Failed to get kline data, aborting run.")
            status_report["outcome"] = "Error: Failed to get kline data"
            return

        quantity = calculate_quantity(client, symbol, df)
        if quantity <= 0:
            logging.warning("Calculated quantity is 0, no trade will be placed.")
            # Do not return, to allow signal generation for status report

        df['ma_short'] = df['close'].rolling(window=5, min_periods=5).mean()
        df['ma_long'] = df['close'].rolling(window=20, min_periods=20).mean()
        plot_moving_averages(df, symbol)

        pipe, meta = _load_model()
        sig = "hold"

        if not pipe or not meta:
            logging.info("Model not found, using MA fallback strategy.")
            sig = ma_signal(df)
        else:
            feature_cols = meta["feature_cols"]
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                logging.warning(f"Model feature columns missing: {missing}. Using MA fallback.")
                sig = ma_signal(df)
            else:
                x = df[feature_cols].tail(1)
                try:
                    proba = float(pipe.predict_proba(x)[:, 1][0])
                    conf_threshold = meta["best_params"].get("confidence_threshold", 0.5)
                    
                    logging.info(f"Model probability: {proba:.4f}, Confidence Threshold: {conf_threshold:.4f}")
                    if proba >= conf_threshold:
                        sig = "buy"
                    else:
                        sig = "hold"

                except Exception as e:
                    logging.error(f"Model inference failed: {e}. Using MA fallback.", exc_info=True)
                    sig = ma_signal(df)

        logging.info(f"Final Signal: {sig.upper()}")
        status_report["signal"] = sig.upper()

        if sig == "buy" and quantity > 0:
            current_price = df['close'].iloc[-1]
            stop_loss_pct = config.get("trade", {}).get("stop_loss_pct", 0.02)
            take_profit_pct = config.get("trade", {}).get("take_profit_pct", 0.05)

            stop_loss_price = current_price * (1 - stop_loss_pct)
            take_profit_price = current_price * (1 + take_profit_pct)
            
            order_result = client.place_oco_order(symbol, "buy", quantity, take_profit_price, stop_loss_price)
            status_report["outcome"] = f"Placed BUY order for {quantity} {symbol}. Response: {order_result}"

        elif sig == "sell" and quantity > 0:
            logging.info("Sell signal received. Placing a simple market sell order.")
            order_result = client.place_order(symbol, "sell", quantity)
            status_report["outcome"] = f"Placed SELL order for {quantity} {symbol}. Response: {order_result}"
        else:
            logging.info("Signal is 'hold' or quantity is zero, no order placed.")
            status_report["outcome"] = "Signal is 'hold' or quantity is zero, no order placed."

    except Exception as e:
        logging.exception(f"An unhandled error occurred during the run cycle: {e}")
        status_report["outcome"] = f"Error: {e}"
    finally:
        logging.info("Updating final status report.")
        try:
            equity = client.get_usdt_equity()
            status_report["account_equity"] = equity if equity is not None else "Failed to retrieve"
        except Exception as e:
            logging.error(f"Could not retrieve account equity for status report: {e}")
            status_report["account_equity"] = "Error retrieving equity"
        
        status_report["last_update"] = datetime.utcnow().isoformat() + "Z"
        _write_status(status_report)
        logging.info("Status file 'status.json' has been updated.")
