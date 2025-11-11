# config/__init__.py
import os
from utils.config_loader import load_config

class ConfigProxy:
    _config = None
    _loaded = False

    def _load_config(self):
        """
        Private method to load the configuration on first access.
        """
        if not self._loaded:
            # This is the core of the lazy loading. The expensive `load_config`
            # is only called when a config attribute is first accessed.
            _current_dir = os.path.dirname(os.path.abspath(__file__))
            _root_dir = os.path.dirname(_current_dir)
            _settings_path = os.path.join(_root_dir, 'config', 'settings.yaml')

            # Fallback to the example file if the main one doesn't exist.
            if not os.path.exists(_settings_path):
                _settings_path = os.path.join(_root_dir, 'config', 'settings.yaml.example')

            self._config = load_config(_settings_path)
            self._loaded = True

    def __getattr__(self, name):
        """
        Magic method to intercept attribute access.
        e.g., config.some_key
        """
        self._load_config()
        # Return the attribute from the loaded config dictionary.
        # This might raise an AttributeError if the key doesn't exist, which is expected.
        return self._config[name]

    def __getitem__(self, key):
        """
        Magic method to intercept item access (like a dictionary).
        e.g., config['some_key']
        """
        self._load_config()
        return self._config[key]

    def get(self, key, default=None):
        """
        Provides a safe way to get a value, similar to dict.get().
        """
        self._load_config()
        return self._config.get(key, default)

# Instantiate the proxy.
# When other modules `from config import config`, they get this instance.
config = ConfigProxy()
