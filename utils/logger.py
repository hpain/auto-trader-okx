import logging
import os
from logging.handlers import RotatingFileHandler
import sys

def setup_logger():
    """
    Set up the root logger to output to console and a rotating file.
    This function is designed to be idempotent and avoid conflicts with other libraries.
    """
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, "trader.log")

    log_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(module)s - %(message)s"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers by checking if handlers of the same type already exist.
    # This is more robust than clearing all handlers.
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_formatter)
        root_logger.addHandler(console_handler)

    if not any(isinstance(h, RotatingFileHandler) for h in root_logger.handlers):
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8'
        )
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)

    # Use a flag on the logger itself to ensure the configuration message is logged only once.
    if not getattr(root_logger, '_configured', False):
        logging.info("Logger has been configured.")
        root_logger._configured = True
