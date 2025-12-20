import yaml
from typing import Dict, Any
from pathlib import Path
import time
import logging
import os

logger = logging.getLogger(__name__)

def load_config(file_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file, with a retry mechanism to handle
    filesystem delays.
    Also overrides configuration with environment variables if present.

    :param file_path: The path to the YAML file.
    :return: A dictionary containing the configuration.
    """
    path = Path(file_path)
    
    # Retry mechanism for filesystem race conditions, especially in tests
    for attempt in range(3):
        if path.is_file():
            break
        logger.warning(f"Config file not found on attempt {attempt + 1}. Retrying in 10ms...")
        time.sleep(0.01)
    else:
        # If file doesn't exist, we might still want to proceed if env vars are set
        # But generally we expect a base config file.
        # For now, we'll log a warning and return an empty dict or raise error.
        # Given existing logic raises error, we keep it, but maybe we can just return empty dict 
        # and let env vars fill it? 
        # For safety, let's keep raising error but maybe loose it if we decide to go full env-var.
        raise FileNotFoundError(f"Configuration file not found after multiple attempts at: {file_path}")

    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}

    # Override with Environment Variables
    _override_with_env(config)
    
    return config

def _override_with_env(config: Dict[str, Any]):
    """
    Helper to override config dict with environment variables.
    """
    # OKX Credentials
    if os.getenv('OKX_API_KEY'):
        config.setdefault('okx', {})['api_key'] = os.getenv('OKX_API_KEY')
    if os.getenv('OKX_SECRET_KEY'):
        config.setdefault('okx', {})['secret_key'] = os.getenv('OKX_SECRET_KEY')
    if os.getenv('OKX_PASSPHRASE'):
        config.setdefault('okx', {})['passphrase'] = os.getenv('OKX_PASSPHRASE')
    if os.getenv('OKX_FLAG'):
        config.setdefault('okx', {})['flag'] = os.getenv('OKX_FLAG')

    # General Trading
    if os.getenv('TRADE_SYMBOL'):
        config.setdefault('trading', {})['symbol'] = os.getenv('TRADE_SYMBOL')
    if os.getenv('TRADE_INTERVAL'):
        config.setdefault('trading', {})['interval'] = os.getenv('TRADE_INTERVAL')
