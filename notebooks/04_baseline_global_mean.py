"""
04_baseline_global_mean.py — Global Mean Profile Baseline
=========================================================
PINN-Retail: Physics-Informed Neural Networks per Demand Forecasting
di Prodotti Deperibili.

Passo 3c: baseline profilo medio globale.
Predizione: D_pred(day, hour) = mean_d∈train S_obs(d, hour)
Per ogni serie, un unico vettore di 24 valori (profilo medio su tutti i
giorni di training) usato come predizione per ogni giorno.

Metriche: WAPE e WPE su overall / in-stock / stockout.
Split: train (giorni 2-83), val (giorni 84-90), test (eval, giorni 91-97).
Profilo calcolato su giorni 1-83 (training completo).

Eseguire con: freshnet/bin/python notebooks/04_baseline_global_mean.py
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
print('  GLOBAL MEAN PROFILE BASELINE')
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
n_days = len(all_dates)
print(f'  Train: {len(df_train):,} righe, giorni 1-90')
print(f'  Eval:  {len(df_eval):,} righe, giorni 91-97')
print(f'  Full:  {len(df_full):,} righe, {n_days} giorni, {n_series:,} serie')

del df_train, df_eval

# ---------------------------------------------------------------------------
# 2. Calcolo profilo medio globale + metriche
# ---------------------------------------------------------------------------
print('\n2. Calcolo profilo medio globale e metriche...')
print('   Val:  profilo = media hours_sale su giorni 1-83 (solo train)')
print('   Test: profilo = media hours_sale su giorni 1-90 (train+val)')

pooled_full = {
    split: {
        sub: {'sae': 0., 'sao': 0., 'se': 0., 'so': 0., 'n': 0}
        for sub in ['overall', 'instock', 'stockout']
    }
    for split in ['train', 'val', 'test']
}
per_series_records = {s: [] for s in ['train', 'val', 'test']}

groups = df_full.groupby(['store_id', 'product_id'], sort=False)
n_groups = len(groups)

for i, ((sid, pid), grp) in enumerate(groups):
    if (i + 1) % 10000 == 0:
        print(f'    ... {i+1:,}/{n_groups:,} serie')

    grp_s = grp.sort_values('day_num')
    sales = np.array(grp_s['hours_sale'].tolist(), dtype=np.float64)  # (N, 24)
    stock = np.array(grp_s['hours_stock_status'].tolist())
    days = grp_s['day_num'].values
    N = len(days)

    # Training data: days 1-83
    train_mask = days <= 83

    if not train_mask.any():
        continue

    # Profile for train/val: mean across days 1-83
    profile_trainval = sales[train_mask].mean(axis=0)  # (24,)

    # Profile for test: mean across days 1-90 (train+val)
    trainval_mask = days <= 90
    profile_test = sales[trainval_mask].mean(axis=0)  # (24,)

    for split_name, d_min, d_max in [('train', 2, 83), ('val', 84, 90), ('test', 91, 97)]:
        mask = (days >= d_min) & (days <= d_max)
        if not mask.any():
            continue

        # Use appropriate profile
        profile = profile_test if split_name == 'test' else profile_trainval
        n_days_split = mask.sum()
        p = np.tile(profile, (n_days_split, 1))
        o = sales[mask]
        s = stock[mask]

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

# Pooled results
pooled_results = {}
for split_name in ['train', 'val', 'test']:
    r = {}
    for sub in ['overall', 'instock', 'stockout']:
        acc = pooled_full[split_name][sub]
        r[f'wape_{sub}'] = acc['sae'] / acc['sao'] if acc['sao'] > 0 else np.nan
        r[f'wpe_{sub}'] = acc['se'] / acc['so'] if acc['so'] > 0 else np.nan
        r[f'n_{sub}'] = acc['n']
    pooled_results[split_name] = r

# Per-series DataFrames
per_series_dfs = {}
for split_name in ['train', 'val', 'test']:
    ps = pd.DataFrame(per_series_records[split_name])
    per_series_dfs[split_name] = ps
    out_path = os.path.join(RESULTS_DIR, f'global_mean_{split_name}_per_series.parquet')
    ps.to_parquet(out_path, index=False)
    print(f'  Salvato: {out_path} ({len(ps):,} serie)')

# ---------------------------------------------------------------------------
# 4. Tabella risultati pooled
# ---------------------------------------------------------------------------
print(format_metrics_table(pooled_results, model_name='Global Mean Profile'))

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
# 6. Confronto con baseline precedenti
# ---------------------------------------------------------------------------
print('\n' + '=' * 72)
print('  6. CONFRONTO CON BASELINE PRECEDENTI')
print('=' * 72)

print(f'\n  {"Split":<8} {"Model":<24} {"WAPE_pooled":>12} {"WAPE_med_ps":>14} '
      f'{"WPE_pooled":>12}')
print('  ' + '-' * 74)

baseline_files = {
    'Global Mean': 'global_mean',
}

for split_name in ['train', 'val', 'test']:
    for model_label, prefix in baseline_files.items():
        if model_label == 'Global Mean':
            wape_p = pooled_results[split_name]['wape_overall']
            wpe_p = pooled_results[split_name]['wpe_overall']
            med = per_series_dfs[split_name]['wape_overall'].median()
        else:
            path = os.path.join(RESULTS_DIR, f'{prefix}_{split_name}_per_series.parquet')
            if not os.path.exists(path):
                continue
            ps = pd.read_parquet(path)
            med = ps['wape_overall'].median()
            wape_p = np.nan
            wpe_p = np.nan

        print(f'  {split_name if model_label == list(baseline_files.keys())[0] else "":<8} '
              f'{model_label:<24} '
              f'{wape_p:>12.4f} {med:>14.4f} {wpe_p:>12.4f}')
    print()

# ---------------------------------------------------------------------------
# 7. Figure distribuzioni
# ---------------------------------------------------------------------------
print('7. Generazione figure distribuzioni...')

# Fig 20: Histograms WAPE_overall e WPE_overall per split
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Global Mean Profile — Distribuzione per-serie', fontsize=14)

for j, (split_name, ps) in enumerate(per_series_dfs.items()):
    # WAPE overall
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

    # WPE overall
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
fig.savefig(os.path.join(FIG_DIR, 'fig20_global_mean_per_series_distributions.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Salvata: fig20_global_mean_per_series_distributions.png')

# Fig 21: Boxplot comparativi per split
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Global Mean Profile — Confronto split', fontsize=14)

data_wape = [ps['wape_overall'].dropna().clip(upper=ps['wape_overall'].dropna().quantile(0.99))
             for ps in per_series_dfs.values()]
bp = axes[0].boxplot(data_wape, tick_labels=list(per_series_dfs.keys()), patch_artist=True)
for patch, color in zip(bp['boxes'], ['#4C72B0', '#55A868', '#C44E52']):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[0].set_title('WAPE overall per split')
axes[0].set_ylabel('WAPE')

data_wpe = [ps['wpe_overall'].dropna().clip(
    lower=ps['wpe_overall'].dropna().quantile(0.01),
    upper=ps['wpe_overall'].dropna().quantile(0.99))
    for ps in per_series_dfs.values()]
bp = axes[1].boxplot(data_wpe, tick_labels=list(per_series_dfs.keys()), patch_artist=True)
for patch, color in zip(bp['boxes'], ['#4C72B0', '#55A868', '#C44E52']):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[1].axhline(0, color='black', linestyle='-', linewidth=0.8)
axes[1].set_title('WPE overall per split')
axes[1].set_ylabel('WPE')

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig21_global_mean_per_series_boxplots.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Salvata: fig21_global_mean_per_series_boxplots.png')

print('\n' + '=' * 72)
print('  DONE')
print('=' * 72)
