"""Regenerate models/metadata.json.feature_cols by running the feature pipeline on the latest history CSV.

Usage:
    python tools/regenerate_metadata.py

This will backup existing metadata.json -> metadata.json.bak and write a new metadata.json with updated feature_cols.
"""
import os
import json
import shutil
import sys

# Ensure project root is on path when running from tools/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from features.feature_engineering import generate_features, make_supervised
from utils.load_history import load_latest_history_in_dir, load_history_csv
from config import config

MODELS_DIR = os.path.join(ROOT, "models")
META_PATH = os.path.join(MODELS_DIR, "metadata.json")


def main():
    history_dir = config["paths"]["history_data_dir"]
    # Try to find a suitable price CSV (with open/close columns). Iterate files from newest to oldest.
    files = []
    if os.path.exists(history_dir):
        files = [os.path.join(history_dir, f) for f in os.listdir(history_dir) if f.endswith('.csv')]
        files = sorted(files, reverse=True)
    chosen = None
    for candidate in files:
        try:
            df_try = load_history_csv(candidate)
            if 'open' in df_try.columns or 'close' in df_try.columns:
                chosen = candidate
                df = df_try
                break
        except Exception:
            continue
    if chosen is None:
        print("No suitable price CSV found in", history_dir)
        return
    print("Using history file:", chosen)
    # generate features
    news_csv = os.path.join(history_dir, "sample_crypto_news.csv")
    dfm = generate_features(df, news_csv_path=news_csv)
    # supervised dataset
    data = make_supervised(dfm, horizon=1, threshold=0.005)
    non_feature_cols = ["ts", "dt", "y", "future_high", "future_low", "future_close", "future_ret", "date", "timestamp", "vol_ccy", "vol_ccy_quote", "confirm"]
    feature_cols = [c for c in data.columns if c not in non_feature_cols]

    # Backup old metadata
    if os.path.exists(META_PATH):
        shutil.copyfile(META_PATH, META_PATH + ".bak")
        try:
            with open(META_PATH + ".bak", "r", encoding="utf-8") as f:
                old = json.load(f)
        except Exception:
            old = None
    else:
        old = None

    new_meta = {
        "model_type": "lgb_classifier",
        "feature_cols": feature_cols,
        "best_params": old.get("best_params", {}) if old else {},
        "n_samples": len(data),
        "best_score": old.get("best_score", 0) if old else 0,
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(new_meta, f, ensure_ascii=False, indent=2)
    print(f"WROTE new metadata with {len(feature_cols)} features to {META_PATH}")


if __name__ == '__main__':
    main()
