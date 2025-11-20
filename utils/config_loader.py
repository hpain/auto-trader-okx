import yaml
from typing import Dict, Any
from pathlib import Path
import time
import logging

logger = logging.getLogger(__name__)

def load_config(file_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file, with a retry mechanism to handle
    filesystem delays.

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
        raise FileNotFoundError(f"Configuration file not found after multiple attempts at: {file_path}")

    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
