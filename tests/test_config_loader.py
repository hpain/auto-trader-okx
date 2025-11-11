import pytest
import os
import yaml
from utils.config_loader import load_config
import utils.config_loader as config_loader_module

# Fixture to create a temporary config file for testing
@pytest.fixture
def temp_config_file(tmp_path):
    config_data = {
        'okx': {
            'api_key': 'yaml_api_key',
            'secret_key': 'yaml_secret_key',
            'passphrase': 'yaml_passphrase',
            'flag': '0'
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
    Tests that configuration is loaded correctly from the YAML file
    when no environment variables are set.
    """
    config = load_config(temp_config_file)
    assert config['okx']['api_key'] == 'yaml_api_key'
    assert config['okx']['secret_key'] == 'yaml_secret_key'
    assert config['trading']['symbol'] == 'BTC-USDT'

def test_override_with_env_vars(temp_config_file, monkeypatch):
    """
    Tests that environment variables override the values from the YAML file.
    """
    # Set environment variables
    monkeypatch.setenv("OKX_API_KEY", "env_api_key")
    monkeypatch.setenv("OKX_PASSPHRASE", "env_passphrase")

    config = load_config(temp_config_file)

    # Assert that env vars took precedence
    assert config['okx']['api_key'] == 'env_api_key'
    assert config['okx']['passphrase'] == 'env_passphrase'
    # Assert that non-overridden values are still loaded from YAML
    assert config['okx']['secret_key'] == 'yaml_secret_key'
    assert config['okx']['flag'] == '0'

def test_load_dotenv_is_called_and_works(temp_config_file, monkeypatch):
    """
    Tests the effect of the load_dotenv call by mocking it.
    This avoids filesystem-related race conditions in tests.
    """
    # This mock simulates that load_dotenv() has found a .env file
    # and has set the environment variables accordingly.
    def mock_load_dotenv(*args, **kwargs):
        os.environ['OKX_API_KEY'] = 'dotenv_api_key'
        os.environ['OKX_FLAG'] = '1'

    # Replace the real load_dotenv with our mock function
    monkeypatch.setattr(config_loader_module, 'load_dotenv', mock_load_dotenv)

    # Load the config
    config = load_config(temp_config_file)

    # Check if the values set by our mock_load_dotenv were picked up
    assert config['okx']['api_key'] == 'dotenv_api_key'
    assert config['okx']['flag'] == '1'
    # Check that a value not in the mock still falls back to yaml
    assert config['okx']['secret_key'] == 'yaml_secret_key'

def test_file_not_found():
    """
    Tests that FileNotFoundError is raised for a non-existent file.
    """
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_file.yaml")
