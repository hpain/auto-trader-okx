import datetime
import subprocess
import sys
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    start_year = 2020
    current_year = datetime.datetime.now().year
    # Determine end year. If we are early in the year, we might want data up to now.
    # The fetch script works by years. So we pass current_year.
    end_year = current_year
    
    symbols = ["BTCUSDT", "ETHUSDT"]
    
    for symbol in symbols:
        logger.info(f"🚀 Starting Full History Download for {symbol} ({start_year}-{end_year})...")
        logger.info("This process may take a while depending on your internet connection.")
        
        cmd = [
            sys.executable, "scripts/fetch_full_dataset.py",
            "--symbol", symbol,
            "--start_year", str(start_year),
            "--end_year", str(end_year),
            "--output_dir", "data/history"
        ]
        
        try:
            subprocess.check_call(cmd)
            logger.info(f"✅ Successfully fetched full history for {symbol}")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to fetch history for {symbol}. Error: {e}")
            # Ask user if they want to continue to next symbol? 
            # For automation, we just log and continue or exit.
            # Let's continue.

if __name__ == "__main__":
    main()
