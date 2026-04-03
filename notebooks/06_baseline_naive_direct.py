"""
06_baseline_naive_direct.py — Seasonal Naive Direct Forecast
=============================================================
PINN-Retail: Physics-Informed Neural Networks per Demand Forecasting
di Prodotti Deperibili.

Passo 3e: baseline naive in modalità direct forecast.
Predizione: per ogni serie, il profilo dell'ultimo giorno disponibile
prima dell'orizzonte di forecast è usato come predizione per TUTTI i
giorni dell'orizzonte.

- Val (giorni 84-90): profilo = S_obs(giorno 83) per tutti i 7 giorni
- Test (giorni 91-97): profilo = S_obs(giorno 90) per tutti i 7 giorni

A differenza del naive one-step-ahead (notebook 02), qui NON si usano
osservazioni del periodo di forecast.

Eseguire con: freshnet/bin/python notebooks/06_baseline_naive_direct.py
"""

import sys
import os
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.evaluation.metrics import compute_metrics, format_metrics_table

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Caricamento dati
# ---------------------------------------------------------------------------
print('=' * 72)
print('  SEASONAL NAIVE — DIRECT FORECAST')
print('=' * 72)

print('\n1. Caricamento dati...')
df_train = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df_eval = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_eval.parquet'))

df_train['dt_parsed'] = pd.to_datetime(df_train['dt'])
df_eval['dt_parsed'] = pd.to_datetime(df_eval['dt'])

df_full = pd.concat([df_train, df_eval], ignore_index=True)
df_full = df_full.sort_values(['store_id', 'product_id', 'dt_parsed']).reset_index(drop=True)

all_dates = sorted(df_full['dt_parsed'].unique())
date_to_day = {d: i + 1 for i, d in enumerate(all_dates)}
df_full['day_num'] = df_full['dt_parsed'].map(date_to_day)

n_series = df_full.groupby(['store_id', 'product_id']).ngroups
print(f'  Train: {len(df_train):,} righe, giorni 1-90')
print(f'  Eval:  {len(df_eval):,} righe, giorni 91-97')
print(f'  Full:  {len(df_full):,} righe, {len(all_dates)} giorni, {n_series:,} serie')

del df_train, df_eval

# ---------------------------------------------------------------------------
# 2. Direct forecast: profilo fisso per ogni orizzonte
# ---------------------------------------------------------------------------
print('\n2. Calcolo predizioni direct forecast...')
print('   Val:  profilo = S_obs(giorno 83) -> applicato a giorni 84-90')
print('   Test: profilo = S_obs(giorno 90) -> applicato a giorni 91-97')

pooled_full = {
    split: {
        sub: {'sae': 0., 'sao': 0., 'se': 0., 'so': 0., 'n': 0}
        for sub in ['overall', 'instock', 'stockout']
    }
    for split in ['val', 'test']
}
per_series_records = {s: [] for s in ['val', 'test']}

groups = df_full.groupby(['store_id', 'product_id'], sort=False)
n_groups = len(groups)

for i, ((sid, pid), grp) in enumerate(groups):
    if (i + 1) % 10000 == 0:
        print(f'    ... {i+1:,}/{n_groups:,} serie')

    grp_s = grp.sort_values('day_num')
    sales = np.array(grp_s['hours_sale'].tolist(), dtype=np.float64)  # (N, 24)
    stock = np.array(grp_s['hours_stock_status'].tolist())
    days = grp_s['day_num'].values

    for split_name, anchor_day, d_min, d_max in [
        ('val', 83, 84, 90),
        ('test', 90, 91, 97),
    ]:
        # Anchor: last day before the forecast horizon
        anchor_idx = np.where(days == anchor_day)[0]
        if len(anchor_idx) == 0:
            continue
        anchor_idx = anchor_idx[0]
        profile = sales[anchor_idx]  # (24,)

        # Target days
        target_mask = (days >= d_min) & (days <= d_max)
        if not target_mask.any():
            continue

        n_target = target_mask.sum()
        p = np.tile(profile, (n_target, 1))  # (n_target, 24)
        o = sales[target_mask]
        s = stock[target_mask]

        p_flat = p.ravel()
        o_flat = o.ravel()
        s_flat = s.ravel()
        err = p_flat - o_flat

        # Pooled accumulation
        for sub, smask in [('overall', np.ones(len(p_flat), dtype=bool)),
                           ('instock', s_flat == 0),
                           ('stockout', s_flat == 1)]:
            acc = pooled_full[split_name][sub]
            ef = err[smask]
            of = o_flat[smask]
            acc['sae'] += np.abs(ef).sum()
            acc['sao'] += np.abs(of).sum()
            acc['se'] += ef.sum()
            acc['so'] += of.sum()
            acc['n'] += int(smask.sum())

        # Per-series metrics
        m = compute_metrics(p, o, s)
        m['store_id'] = sid
        m['product_id'] = pid
        per_series_records[split_name].append(m)

# ---------------------------------------------------------------------------
# 3. Build results
# ---------------------------------------------------------------------------
print('\n3. Risultati...')

pooled_results = {}
for split_name in ['val', 'test']:
    r = {}
    for sub in ['overall', 'instock', 'stockout']:
        acc = pooled_full[split_name][sub]
        r[f'wape_{sub}'] = acc['sae'] / acc['sao'] if acc['sao'] > 0 else np.nan
        r[f'wpe_{sub}'] = acc['se'] / acc['so'] if acc['so'] > 0 else np.nan
        r[f'n_{sub}'] = acc['n']
    pooled_results[split_name] = r

per_series_dfs = {}
for split_name in ['val', 'test']:
    ps = pd.DataFrame(per_series_records[split_name])
    per_series_dfs[split_name] = ps
    out_path = os.path.join(RESULTS_DIR, f'naive_direct_{split_name}_per_series.parquet')
    ps.to_parquet(out_path, index=False)
    print(f'  Salvato: {out_path} ({len(ps):,} serie)')

# ---------------------------------------------------------------------------
# 4. Tabella risultati pooled
# ---------------------------------------------------------------------------
print(format_metrics_table(pooled_results, model_name='Naive Direct Forecast'))

# ---------------------------------------------------------------------------
# 5. Distribuzione per-serie
# ---------------------------------------------------------------------------
METRIC_COLS = ['wape_overall', 'wape_instock', 'wape_stockout',
               'wpe_overall', 'wpe_instock', 'wpe_stockout']
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]

print('\n' + '=' * 72)
print('  5. DISTRIBUZIONE METRICHE PER-SERIE')
print('=' * 72)

print(f'\n  {"Split":<8} {"Metric":<16} {"Mean":>8} {"Median":>8} '
      f'{"Std":>8} {"Q5":>8} {"Q25":>8} {"Q75":>8} {"Q95":>8} {"Valid":>7}')
print('  ' + '-' * 96)

for split_name, ps in per_series_dfs.items():
    for col in METRIC_COLS:
        vals = ps[col].dropna()
        if len(vals) == 0:
            continue
        qs = np.quantile(vals, QUANTILES)
        print(f'  {split_name:<8} {col:<16} {vals.mean():>8.4f} {vals.median():>8.4f} '
              f'{vals.std():>8.4f} {qs[0]:>8.4f} {qs[1]:>8.4f} {qs[2]:>8.4f} {qs[3]:>8.4f} '
              f'{len(vals):>7,}')

# ---------------------------------------------------------------------------
# 6. Figure
# ---------------------------------------------------------------------------
print('\n6. Generazione figure...')

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Naive Direct Forecast — Distribuzione per-serie', fontsize=14)

for j, (split_name, ps) in enumerate(per_series_dfs.items()):
    ax = axes[0, j]
    vals = ps['wape_overall'].dropna()
    vals_clipped = vals.clip(upper=vals.quantile(0.99))
    ax.hist(vals_clipped, bins=80, color='steelblue', alpha=0.7, edgecolor='none')
    ax.axvline(vals.median(), color='red', linestyle='--', linewidth=1.5,
               label=f'median={vals.median():.3f}')
    ax.set_title(f'WAPE overall — {split_name}')
    ax.set_xlabel('WAPE')
    ax.set_ylabel('N serie')
    ax.legend(fontsize=8)

    ax = axes[1, j]
    vals = ps['wpe_overall'].dropna()
    vals_clipped = vals.clip(lower=vals.quantile(0.01), upper=vals.quantile(0.99))
    ax.hist(vals_clipped, bins=80, color='darkorange', alpha=0.7, edgecolor='none')
    ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
    ax.axvline(vals.median(), color='red', linestyle='--', linewidth=1.5,
               label=f'median={vals.median():.3f}')
    ax.set_title(f'WPE overall — {split_name}')
    ax.set_xlabel('WPE')
    ax.set_ylabel('N serie')
    ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig26_naive_direct_distributions.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Salvata: fig26_naive_direct_distributions.png')

print('\n' + '=' * 72)
print('  DONE')
print('=' * 72)
