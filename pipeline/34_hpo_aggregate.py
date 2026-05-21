"""
34_hpo_aggregate.py — Aggregates HPO best configs in hpo_summary.csv
=====================================================================
"""
import os, json, functools
import pandas as pd
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

forecasters = ['tft', 'lgb', 'lgb_nolags', 'mlp', 'mlp_nolags']
rows = []
for fc in forecasters:
    path = os.path.join(RESULTS_DIR, f'hpo_{fc}_best.json')
    if not os.path.exists(path):
        print(f'SKIP {fc}: {path} not found')
        continue
    with open(path) as f:
        best = json.load(f)
    row = {
        'forecaster': fc,
        'best_trial': best['best_trial'],
        'best_val_wape_med': best['best_value'],
        'n_trials': best['n_trials'],
        **{f'hp_{k}': v for k, v in best['best_params'].items()},
    }
    rows.append(row)

if not rows:
    print('No HPO results found.')
    exit(1)

df = pd.DataFrame(rows)
out_path = os.path.join(RESULTS_DIR, 'hpo_summary.csv')
df.to_csv(out_path, index=False)
print(f'Saved: {out_path}')
print('\nSummary:')
print(df.to_string(index=False))
