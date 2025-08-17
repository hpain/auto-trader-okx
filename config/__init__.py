


# config/__init__.py
import os
import yaml

_here = os.path.dirname(__file__)
_yaml_path = os.path.join(_here, "settings.yaml")

if not os.path.exists(_yaml_path):
    raise FileNotFoundError(f"Config file not found: {_yaml_path}")

with open(_yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

__all__ = ["config"]
