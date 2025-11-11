import sys
import os
import json
import ast
import random
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from joblib import load

# Ensure project root is on sys.path so relative imports work when running this script
BASE = Path(__file__).resolve().parents[1]
REPO_ROOT = str(BASE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
TRIALS_CSV = BASE / 'models' / 'trials_summary.csv'
META_JSON = BASE / 'models' / 'metadata.json'
OUT_DIR = BASE / 'models'

# imports from project
from features.feature_engineering import generate_features, make_supervised
from utils.data_normalization import normalize_binance_df
from data.binance import get_klines_bian
from utils.backtest import run_backtest
import logging
logging.basicConfig(level=logging.INFO)

# read trials
print('Reading trials...')
df = pd.read_csv(TRIALS_CSV)

# parse params column
def parse_params(s):
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return {}

# select zero-sharpe trials
zero_df = df[df['overall_sharpe'] == 0].copy()
print(f"Found {len(zero_df)} zero-sharpe trials")

# summarize confidence_threshold distribution
ct_stats = zero_df['confidence_threshold'].describe()
print('\nconfidence_threshold stats for zero-sharpe trials:')
print(ct_stats)

# parse params for zero-sharpe
params_list = zero_df['params'].apply(parse_params).tolist()
# collect simple histograms
from collections import Counter, defaultdict
lr_vals = [round(p.get('learning_rate', None), 4) for p in params_list]
nl_vals = [p.get('num_leaves', None) for p in params_list]
ne_vals = [p.get('n_estimators', None) for p in params_list]

print('\nSample param distributions (top counts):')
print('learning_rate sample:', Counter(lr_vals).most_common(10))
print('num_leaves sample:', Counter(nl_vals).most_common(10))
print('n_estimators sample:', Counter(ne_vals).most_common(10))

# choose up to 3 candidates: highest stability among zero-sharpe, plus 2 random (if available)
candidates = []
if not zero_df.empty:
    sorted_zero = zero_df.sort_values('stability_score', ascending=False)
    candidates.append(sorted_zero.iloc[0])
    # add up to 2 random different
    remaining = sorted_zero.iloc[1:]
    n_rand = min(2, len(remaining))
    if n_rand > 0:
        rand_idx = random.sample(list(remaining.index), n_rand)
        for i in rand_idx:
            candidates.append(sorted_zero.loc[i])

print(f"\nSelected {len(candidates)} trials for replay: {[int(c['trial_number']) for c in candidates]}")

# Load / generate feature dataset (reuse research/evolve pipeline choices)
from config import config
symbol = config.get('trade', {}).get('symbol', 'BTC-USDT')
interval = config.get('trade', {}).get('interval', '1H')
years = 3

# Try cache path like research/evolve, but fall back to regenerate
import hashlib
config_str = f"{symbol}-{interval}-{years}-" + str(BASE / config['paths']['history_data_dir'])
config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:10]
cache_dir = Path(config['paths']['feature_cache_dir'])
feature_cache_path = cache_dir / f"features_{config_hash}.parquet"

if feature_cache_path.exists():
    print('Loading feature cache:', feature_cache_path)
    dfm = pd.read_parquet(feature_cache_path)
else:
    print('No feature cache found, loading local history CSV...')
    # Try common history filenames in data/history
    hist_dir = Path(BASE) / 'data' / 'history'
    hist_candidates = [
        hist_dir / f'binance_{symbol.replace("/","")}_{interval}_{years}y.csv',
        hist_dir / f'binance_BTCUSDT_1h_3y.csv',
        hist_dir / f'binance_BTCUSDT_1h_2y.csv',
        hist_dir / f'binance_BTCUSDT_1h_1y.csv',
    ]
    dfp = None
    for p in hist_candidates:
        if p.exists():
            print('Loading history file:', p)
            dfp = pd.read_csv(p)
            # ensure ts column parsed and set as index
            if 'ts' in dfp.columns:
                dfp['ts'] = pd.to_datetime(dfp['ts'])
                dfp = dfp.set_index('ts')
            break
    if dfp is None:
        raise SystemExit('No local history CSV found in data/history; cannot proceed')
    # normalize and generate features
    try:
        dfp = normalize_binance_df(dfp)
    except Exception:
        # normalize_binance_df expects certain columns; try simple renaming fallback
        dfp = dfp.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'vol'})
    dfm = generate_features(dfp)
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        dfm.to_parquet(feature_cache_path)
        print('Saved feature cache to', feature_cache_path)
    except Exception:
        print('Could not save feature cache (parquet write failed), continuing without cache')

# Build supervised data with default profit threshold 0.005 (same as default in research/evolve)
data = make_supervised(dfm, horizon=1, threshold=0.005)

# get feature columns from metadata if present, else construct
if META_JSON.exists():
    meta = json.loads(META_JSON.read_text(encoding='utf-8'))
    feature_cols = meta.get('feature_cols', [c for c in data.columns if c not in ['y','future_high','future_low','future_close','future_ret']])
else:
    feature_cols = [c for c in data.columns if c not in ['y','future_high','future_low','future_close','future_ret']]

# drop NaNs
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(subset=feature_cols + ['y'], inplace=True)

print('Final data rows:', len(data), 'feature_cols:', len(feature_cols))

# helper to replay a single trial
import lightgbm as lgb

def replay_trial_row(row):
    params = parse_params(row['params'])
    conf = float(row['confidence_threshold']) if not pd.isna(row['confidence_threshold']) else params.get('confidence_threshold', 0.95)
    print('\n--- Replaying trial', int(row['trial_number']), 'conf_thresh=', conf, 'params=', params)

    # Ensure params safe for LGBMClassifier
    params_train = {k: v for k, v in params.items() if k in ['n_estimators','learning_rate','num_leaves']}
    model = lgb.LGBMClassifier(random_state=42, verbose=-1, **params_train)

    tscv = TimeSeriesSplit(n_splits=5)
    fold_summaries = []
    total_trade_count = 0
    total_success = 0
    returns_concat = []
    trade_logs = []

    for i, (train_idx, test_idx) in enumerate(tscv.split(data)):
        X_train = data.iloc[train_idx][feature_cols]
        y_train = data.iloc[train_idx]['y'].astype(int)
        X_test = data.iloc[test_idx][feature_cols]

        # Train
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            print('Model training failed on fold', i, 'error:', e)
            continue

        preds = pd.Series(model.predict(X_test), index=X_test.index)
        probs = model.predict_proba(X_test)[:, 1]

        _total_ret, _max_dd, success_rate, trade_count, returns_series = run_backtest(
            predictions=preds,
            probabilities=probs,
            test_data=data.loc[X_test.index],
            confidence_threshold=conf,
            stop_loss_pct=0.02,
            take_profit_pct=0.05,
        )
        fold_summaries.append({'fold': i, 'trade_count': trade_count, 'success_rate': success_rate, 'total_ret': _total_ret})
        total_trade_count += trade_count
        total_success += success_rate * trade_count
        returns_concat.append(returns_series)

        # build small trade log from returns_series: include indices where returns != 0
        nz = returns_series[returns_series != 0]
        if not nz.empty:
            # returns_series may be a default-indexed Series (0..n-1). Align positions to X_test.index
            log_entries = []
            if len(returns_series) == len(X_test):
                idx_list = list(X_test.index)
                nz_positions = [int(pos) for pos in np.where(returns_series.values != 0)[0]]
                for pos in nz_positions[:20]:
                    idx = idx_list[pos]
                    rowdata = data.loc[idx]
                    entry_price = rowdata['close']
                    future_close = rowdata['future_close']
                    realized = float(returns_series.iloc[pos])
                    log_entries.append({'index': str(idx), 'entry_price': float(entry_price), 'future_close': float(future_close), 'realized_return': realized})
            else:
                # fallback: try to use nz.index directly
                for idx in nz.index[:20]:
                    try:
                        rowdata = data.loc[idx]
                        entry_price = rowdata['close']
                        future_close = rowdata['future_close']
                        realized = float(nz.loc[idx])
                        log_entries.append({'index': str(idx), 'entry_price': float(entry_price), 'future_close': float(future_close), 'realized_return': realized})
                    except Exception:
                        continue
            trade_logs.append({'fold': i, 'sample_trades': log_entries})

    avg_success_rate = (total_success / total_trade_count) if total_trade_count > 0 else 0.0
    print('Fold summaries:', fold_summaries)
    print('Total trade count across folds:', total_trade_count)
    print('Average success rate:', avg_success_rate)
    print('Sample trade logs (first non-zero trades, up to 20 per fold):')
    for tl in trade_logs:
        print(tl)

# run replays
for r in candidates:
    replay_trial_row(r)

print('\nDone')
