
import requests
import zipfile
import io
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_url(url):
    logger.info(f"Checking URL: {url}")
    try:
        r = requests.get(url)
        if r.status_code != 200:
            logger.error(f"Failed to download: {r.status_code}")
            return

        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            file_list = z.namelist()
            logger.info(f"Files in zip: {file_list}")
            
            for fname in file_list:
                if fname.endswith('.csv'):
                    with z.open(fname) as f:
                        df = pd.read_csv(f)
                        logger.info(f"--- columns in {fname} ---")
                        logger.info(df.columns.tolist())
                        logger.info(f"--- First 2 rows ---")
                        print(df.head(2))
                        
                        # Check for target columns
                        has_oi = any('open_interest' in c.lower() for c in df.columns)
                        has_fr = any('funding' in c.lower() for c in df.columns)
                        has_ratio = any('ratio' in c.lower() for c in df.columns)
                        
                        logger.info(f"Contains OI? {has_oi}")
                        logger.info(f"Contains Funding? {has_fr}")
                        logger.info(f"Contains Ratio? {has_ratio}")

    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    # 1. User provided date
    inspect_url("https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-2021-12-01.zip")
    
    # 2. Check Early 2020 date (e.g. May 2020) where we previously thought data was missing
    # Naming convention: usually BTCUSDT-metrics-2020-05-01.zip
    inspect_url("https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-2020-05-01.zip")
