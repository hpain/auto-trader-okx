import os
import pandas as pd
import tempfile
from utils.load_history import load_history_csv


def test_load_history_short_names():
    # Create a temporary CSV with short column names
    df = pd.DataFrame({
        'ts': ['2025-01-01T00:00:00Z','2025-01-01T01:00:00Z'],
        'o': [100.0, 101.0],
        'h': [101.0, 102.0],
        'l': [99.0, 100.5],
        'c': [100.5, 101.5],
        'v': [10, 12]
    })
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    tmp_path = tmp.name
    tmp.close()
    try:
        df.to_csv(tmp_path, index=False)
        loaded = load_history_csv(tmp.name)
        assert 'open' in loaded.columns and 'close' in loaded.columns
        assert pd.api.types.is_datetime64_any_dtype(loaded['ts'])
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
