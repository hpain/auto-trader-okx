"""
Utilities to load and normalize history CSVs in a single place.
This ensures consistent column names and numeric typing across the project.
"""
import os
import pandas as pd


def load_history_csv(path: str) -> pd.DataFrame:
    """Load a single CSV and normalize column names/types.
    Returns a DataFrame with columns like ['ts','open','high','low','close','vol'] (ts not indexed).
    """
    df = pd.read_csv(path)
    try:
        from .data_normalization import normalize_binance_df
    except Exception:
        # fallback: try import relative from repo root
        import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from utils.data_normalization import normalize_binance_df
    df = normalize_binance_df(df)
    # Ensure ts is datetime
    if 'ts' in df.columns:
        df['ts'] = pd.to_datetime(df['ts'], utc=True)
    return df


def load_latest_history_in_dir(dir_path: str) -> str:
    """Return path to the latest CSV file in dir_path, or None if none found."""
    if not os.path.exists(dir_path):
        return None
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.csv')]
    if not files:
        return None
    files = sorted(files)
    return files[-1]
