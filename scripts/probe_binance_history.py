
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_url(url):
    try:
        r = requests.head(url) # Use HEAD to save bandwidth
        if r.status_code == 200:
            logger.info(f"[EXISTS] {url}")
            return True
        else:
            logger.warning(f"[MISSING] {url} (Status: {r.status_code})")
            return False
    except Exception as e:
        logger.error(f"Error: {e}")
        return False

if __name__ == "__main__":
    base = "https://data.binance.vision/data/futures/um/daily"
    
    # Dates to probe
    dates = ["2020-01-01", "2020-05-01", "2020-09-01", "2020-10-01", "2021-01-01"]
    
    logger.info("--- Probing Metrics (OI, Ratio) ---")
    for d in dates:
        url = f"{base}/metrics/BTCUSDT/BTCUSDT-metrics-{d}.zip"
        check_url(url)
        
    logger.info("--- Probing Premium Index (Funding Rate) ---")
    for d in dates:
        url = f"{base}/premiumIndex/BTCUSDT/BTCUSDT-premiumIndex-{d}.zip"
        check_url(url)
