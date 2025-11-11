#!/usr/bin/env python3
"""Diagnostic script for model training results

Run from project root:
    python tools/diagnose_models.py

This script reads:
 - models/trials_summary.csv
 - models/metadata.json
 - models/best_trial_details.json
 - data/history/*.csv (if exists)
 
And prints a short diagnostic summary useful for locating why recent trials have low scores.
"""

import os
import sys
import json
import ast
import math
from statistics import mean

import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR = os.path.join(ROOT, "data", "history")


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_params_field(s):
    # tries multiple fallbacks to parse a params string which might be a JSON string
    if not isinstance(s, str):
        return {}
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return {}


def summary_trials(csv_path):
    df = pd.read_csv(csv_path)
    print("\n--- trials_summary.csv ---")
    print("count:", len(df))

    if "overall_sharpe" in df.columns:
        print("overall_sharpe: mean %.4f, median %.4f, max %.4f, min %.4f" % (
            df["overall_sharpe"].mean(), df["overall_sharpe"].median(), df["overall_sharpe"].max(), df["overall_sharpe"].min()
        ))
    else:
        print("overall_sharpe column missing")

    # parse learning_rate from params
    lrs = []
    bad_trials = 0
    zero_sharpe = 0
    for idx, row in df.iterrows():
        params = parse_params_field(row.get("params", "{}"))
        lr = params.get("learning_rate")
        if lr is not None:
            lrs.append(lr)
        if row.get("overall_sharpe") is not None and row.get("overall_sharpe") <= 0:
            zero_sharpe += 1
        if row.get("stability_score") is not None and (isinstance(row.get("stability_score"), str) or math.isnan(row.get("stability_score"))):
            bad_trials += 1
    if lrs:
        print("learning_rate: count %d, mean %.4f, min %.4f, max %.4f" % (len(lrs), mean(lrs), min(lrs), max(lrs)))
    print("trials with overall_sharpe <= 0:", zero_sharpe)
    print("trials with missing/NaN stability_score:", bad_trials)

    # analyze fold_sharpes if available
    if "fold_sharpes" in df.columns:
        fold_vals = []
        for v in df["fold_sharpes"].fillna("[]"):
            parsed = None
            # try JSON first, then literal_eval
            try:
                parsed = json.loads(v)
            except Exception:
                try:
                    parsed = ast.literal_eval(v)
                except Exception:
                    parsed = []
            if isinstance(parsed, (list, tuple)):
                for x in parsed:
                    try:
                        xv = float(x)
                        if np.isfinite(xv):
                            fold_vals.append(xv)
                    except Exception:
                        continue
        if fold_vals:
            print("fold_sharpes composite: mean %.4f median %.4f min %.4f max %.4f" % (
                np.mean(fold_vals), np.median(fold_vals), np.min(fold_vals), np.max(fold_vals)
            ))
    return df


def inspect_metadata(p):
    print("\n--- metadata.json ---")
    m = load_json(p)
    for k in ["model_type", "n_samples", "best_params", "feature_cols", "best_score"]:
        if k in m:
            val = m[k]
            if k == "feature_cols":
                print("feature_cols: count", len(val))
            elif k == "best_params":
                print("best_params:", {k: v for k, v in val.items() if k in ("n_estimators","learning_rate","num_leaves","entry_threshold","take_profit_pct")})
            else:
                print(f"{k}:", val)
    return m


def inspect_best_trials(p):
    print("\n--- best_trial_details.json ---")
    b = load_json(p)
    # print a short summary of recent trials
    atb = b.get("all_time_best")
    if atb:
        print("all_time_best: ts %s, overall_sharpe %s, stability_score %s" % (atb.get("timestamp"), atb.get("overall_sharpe"), atb.get("stability_score")))
    recent = b.get("recent_trials", [])
    print("recent_trials count:", len(recent))
    for r in recent[:5]:
        print("  - ts %s, overall_sharpe %.4f, stability %.4f, params sample:" % (r.get("timestamp"), r.get("overall_sharpe", 0), r.get("stability_score", 0)))
        pp = r.get("params", {})
        print("     ", {k: pp.get(k) for k in ("n_estimators","learning_rate","num_leaves","entry_threshold","take_profit_pct")})
    return b


def load_history_sample(data_dir):
    print("\n--- data/history sample ---")
    if not os.path.exists(data_dir):
        print("history dir not found:", data_dir)
        return None
    try:
        from utils.load_history import load_latest_history_in_dir, load_history_csv
    except Exception:
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from utils.load_history import load_latest_history_in_dir, load_history_csv
    p = load_latest_history_in_dir(data_dir)
    if p is None:
        print("no csv files in history dir")
        return None
    print("loading sample history file:", p)
    df = load_history_csv(p)
    print("shape:", df.shape)

    # Debug: print columns after normalization
    try:
        print("columns:", list(df.columns))
    except Exception:
        pass

    if 'close' in df.columns:
        df['future_return_1'] = df['close'].pct_change().shift(-1)
        s = df['future_return_1'].dropna()
        print("future_return_1: count %d mean %.6f std %.6f SNR %.4f" % (len(s), s.mean(), s.std(), abs(s.mean())/(s.std()+1e-12)))
        print("quantiles:", s.quantile([0.01,0.05,0.1,0.5,0.9,0.95,0.99]).to_dict())
    else:
        print("no close column in sample file")
    return df


def feature_overlap_and_drift(df_hist, feature_cols):
    print("\n--- feature overlap & drift ---")
    if df_hist is None:
        print("no history sample provided")
        return
    cols = [c for c in feature_cols if c in df_hist.columns]
    print("feature overlap: %d / %d" % (len(cols), len(feature_cols)))
    if not cols:
        return
    n = len(df_hist)
    head = df_hist[cols].iloc[:max(200, n//10)]
    tail = df_hist[cols].iloc[-max(200, n//10):]
    shifts = []
    for c in cols:
        try:
            mm = abs(head[c].mean() - tail[c].mean()) / (head[c].std() + 1e-12)
            if mm > 0.5:
                shifts.append((c, mm, head[c].mean(), tail[c].mean()))
        except Exception:
            continue
    print("features with mean shift > 0.5 std (sample):", len(shifts))
    print(shifts[:20])
    # missing percentage
    missing = {c: float(df_hist[c].isna().sum())/len(df_hist) for c in cols}
    missing_sorted = sorted(missing.items(), key=lambda x: -x[1])[:10]
    print("top missing features:", missing_sorted)


if __name__ == '__main__':
    # run diagnostics
    csvp = os.path.join(MODELS_DIR, 'trials_summary.csv')
    if os.path.exists(csvp):
        df_trials = summary_trials(csvp)
    else:
        print('No trials_summary.csv found at', csvp)

    metap = os.path.join(MODELS_DIR, 'metadata.json')
    if os.path.exists(metap):
        meta = inspect_metadata(metap)
    else:
        print('No metadata.json at', metap)
        meta = {}

    bestp = os.path.join(MODELS_DIR, 'best_trial_details.json')
    if os.path.exists(bestp):
        best = inspect_best_trials(bestp)
    else:
        print('No best_trial_details.json at', bestp)
        best = {}

    hist = load_history_sample(DATA_DIR)
    if meta.get('feature_cols') and hist is not None:
        feature_overlap_and_drift(hist, meta.get('feature_cols'))

    print('\n=== DIAGNOSTIC COMPLETE ===')
