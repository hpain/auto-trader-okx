import time
import logging
import re

from trader.executor import run as run_trade_cycle
from utils.logger import setup_logger
from config import config

def get_interval_seconds(interval_str: str) -> int:
    """Parse interval string like '1H', '15m', '4h' into seconds."""
    match = re.match(r"(\d+)([HhMm])", interval_str)
    if not match:
        logging.warning(f"Invalid interval format: '{interval_str}'. Defaulting to 1 hour.")
        return 3600

    value, unit = int(match.group(1)), match.group(2).lower()

    if unit == 'h':
        return value * 3600
    if unit == 'm':
        return value * 60
    
    logging.warning(f"Unknown interval unit: '{unit}'. Defaulting to 1 hour.")
    return 3600

def main():
    """Main function to run the trading bot in a continuous loop."""
    setup_logger()
    
    logging.info("==================================================")
    logging.info("           STARTING AUTO TRADER BOT             ")
    logging.info("==================================================")

    interval_str = config.get("trade", {}).get("interval", "1H")
    sleep_duration = get_interval_seconds(interval_str)

    while True:
        try:
            logging.info(f"--- Starting new trade cycle (Interval: {interval_str}) ---")
            run_trade_cycle()
            logging.info("--- Trade cycle finished successfully. ---")

        except Exception as e:
            # Using logging.exception to automatically capture and log the traceback
            logging.exception(f"An unexpected error occurred in the trade cycle: {e}")
        
        finally:
            logging.info(f"Sleeping for {sleep_duration} seconds until the next cycle.")
            time.sleep(sleep_duration)

if __name__ == "__main__":
    main()
