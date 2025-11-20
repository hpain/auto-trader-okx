import pytest
import yaml
from utils.config_loader import load_config

# Fixture to create a temporary config file for testing
@pytest.fixture
def temp_config_file(tmp_path):
    config_data = {
        'okx': {
            'api_key': 'yaml_api_key',
            'secret_key': 'yaml_secret_key',
            'passphrase': 'yaml_passphrase',
        },
        'trading': {
            'symbol': 'BTC-USDT'
        }
    }
    config_path = tmp_path / "settings.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    return str(config_path)

def test_load_from_yaml_only(temp_config_file):
    """
    Tests that configuration is loaded correctly from the YAML file.
    """
    config = load_config(temp_config_file)
    assert config['okx']['api_key'] == 'yaml_api_key'
    assert config['okx']['secret_key'] == 'yaml_secret_key'
    assert config['trading']['symbol'] == 'BTC-USDT'

def test_file_not_found():
    """
    Tests that FileNotFoundError is raised for a non-existent file.
    """
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_file.yaml")