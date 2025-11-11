import json
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
TRIALS = BASE / 'models' / 'trials_summary.csv'
BEST = BASE / 'models' / 'best_trial_details.json'
META = BASE / 'models' / 'metadata.json'

print('Reading', TRIALS)
df = pd.read_csv(TRIALS)
print('\nTrials count:', len(df))

for col in ['overall_sharpe','stability_score']:
    if col in df.columns:
        print(f"\n{col} stats:")
        print(df[col].describe())

zero_sharpe = df['overall_sharpe'] == 0
print('\nZero overall_sharpe count:', zero_sharpe.sum())

# parse fold_sharpes and params
import ast

def safe_parse(x):
    try:
        return ast.literal_eval(x)
    except Exception:
        try:
            return json.loads(x)
        except Exception:
            return None

print('\nTop 5 trials by stability_score:')
print(df.sort_values('stability_score', ascending=False).head(5)[['trial_number','stability_score','overall_sharpe','confidence_threshold']])

# parse best jsons
print('\nReading best_trial_details.json')
with open(BEST,'r',encoding='utf-8') as f:
    best = json.load(f)
print('all_time_best stability_score:', best.get('all_time_best',{}).get('stability_score'))

print('\nReading metadata.json')
with open(META,'r',encoding='utf-8') as f:
    meta = json.load(f)
print('model_type:', meta.get('model_type'))
print('n_samples:', meta.get('n_samples'))
print('feature_cols count:', len(meta.get('feature_cols',[])))
print('best_params:', meta.get('best_params'))

print('\nDone')
