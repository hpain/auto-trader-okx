
import yaml
from typing import Dict, Any
from pathlib import Path

def load_config(file_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file.

    :param file_path: The path to the YAML file.
    :return: A dictionary containing the configuration.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {file_path}")

    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
