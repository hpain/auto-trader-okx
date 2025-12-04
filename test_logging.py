import logging
import os
import sys

print("Starting test_logging.py")
try:
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/test_log.log"),
            logging.StreamHandler()
        ]
    )
    logging.info("Test log message")
    print("Test print message")
except Exception as e:
    print(f"Error: {e}")
