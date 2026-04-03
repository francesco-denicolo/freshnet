"""
05_baseline_dow_mean.py — Day-of-Week Mean Profile Baseline
===========================================================
PINN-Retail: Physics-Informed Neural Networks per Demand Forecasting
di Prodotti Deperibili.

Passo 3d: baseline profilo medio per giorno della settimana.
Predizione: D_pred(day, hour) = mean_{d∈train, dow(d)=dow(day)} S_obs(d, hour)
Per ogni serie, 7 vettori di 24 valori (uno per giorno della settimana)
usati come predizione per il corrispondente giorno.

Metriche: WAPE e WPE su overall / in-stock / stockout.
Split: train (giorni 2-83), val (giorni 84-90), test (eval, giorni 91-97).
Profili calcolati su giorni 1-83 (training completo).

Eseguire con: freshnet/bin/python notebooks/05_baseline_dow_mean.py
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
print('  DAY-OF-WEEK MEAN PROFILE BASELINE')
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

# Day of week (0=Monday, 6=Sunday)
df_full['dow'] = df_full['dt_parsed'].dt.dayofweek

n_series = df_full.groupby(['store_id', 'product_id']).ngroups
n_days = len(all_dates)
print(f'  Train: {len(df_train):,} righe, giorni 1-90')
print(f'  Eval:  {len(df_eval):,} righe, giorni 91-97')
print(f'  Full:  {len(df_full):,} righe, {n_days} giorni, {n_series:,} serie')

# Day-of-week distribution in training (days 1-83)
train_dates_info = df_full[df_full['day_num'] <= 83][['dt_parsed', 'dow']].drop_duplicates()
dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
dow_counts = train_dates_info['dow'].value_counts().sort_index()
print('\n  Distribuzione DoW nel training (giorni 1-83):')
for dow_val, count in dow_counts.items():
    print(f'    {dow_names[dow_val]}: {count} giorni')

del df_train, df_eval

# ---------------------------------------------------------------------------
# 2. Calcolo profili per DoW + metriche
# ---------------------------------------------------------------------------
print('\n2. Calcolo profili per giorno della settimana e metriche...')
print('   Val:  7 profili DoW calcolati su giorni 1-83 (solo train)')
print('   Test: 7 profili DoW calcolati su giorni 1-90 (train+val)')

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

n_missing_dow = 0  # Count series×DoW with no training data

for i, ((sid, pid), grp) in enumerate(groups):
    if (i + 1) % 10000 == 0:
        print(f'    ... {i+1:,}/{n_groups:,} serie')

    grp_s = grp.sort_values('day_num')
    sales = np.array(grp_s['hours_sale'].tolist(), dtype=np.float64)  # (N, 24)
    stock = np.array(grp_s['hours_stock_status'].tolist())
    days = grp_s['day_num'].values
    dows = grp_s['dow'].values
    N = len(days)

    # Training data: days 1-83
    train_mask = days <= 83

    if not train_mask.any():
        continue

    # Compute DoW profiles from training data (days 1-83) for train/val
    dow_profiles_train = {}
    global_profile_train = sales[train_mask].mean(axis=0)  # fallback

    for d in range(7):
        dow_train = train_mask & (dows == d)
        if dow_train.any():
            dow_profiles_train[d] = sales[dow_train].mean(axis=0)
        else:
            dow_profiles_train[d] = global_profile_train
            n_missing_dow += 1

    # Compute DoW profiles from train+val data (days 1-90) for test
    trainval_mask = days <= 90
    dow_profiles_test = {}
    global_profile_tv = sales[trainval_mask].mean(axis=0)  # fallback

    for d in range(7):
        dow_tv = trainval_mask & (dows == d)
        if dow_tv.any():
            dow_profiles_test[d] = sales[dow_tv].mean(axis=0)
        else:
            dow_profiles_test[d] = global_profile_tv

    for split_name, d_min, d_max in [('train', 2, 83), ('val', 84, 90), ('test', 91, 97)]:
        mask = (days >= d_min) & (days <= d_max)
        if not mask.any():
            continue

        # Use appropriate profiles
        profiles = dow_profiles_test if split_name == 'test' else dow_profiles_train
        idx = np.where(mask)[0]
        p = np.zeros((len(idx), 24))
        for j, t in enumerate(idx):
            p[j] = profiles[dows[t]]
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

if n_missing_dow > 0:
    print(f'\n  Nota: {n_missing_dow} coppie serie×DoW senza dati training '
          f'(usato profilo globale come fallback)')

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
    out_path = os.path.join(RESULTS_DIR, f'dow_mean_{split_name}_per_series.parquet')
    ps.to_parquet(out_path, index=False)
    print(f'  Salvato: {out_path} ({len(ps):,} serie)')

# ---------------------------------------------------------------------------
# 4. Tabella risultati pooled
# ---------------------------------------------------------------------------
print(format_metrics_table(pooled_results, model_name='Day-of-Week Mean Profile'))

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
# 6. Confronto con tutti i baseline
# ---------------------------------------------------------------------------
print('\n' + '=' * 72)
print('  6. CONFRONTO CON TUTTI I BASELINE')
print('=' * 72)

baseline_files = {
    'Global Mean': 'global_mean',
    'DoW Mean': 'dow_mean',
}

print(f'\n  {"Split":<8} {"Model":<24} {"WAPE_med_ps":>14}')
print('  ' + '-' * 50)

for split_name in ['train', 'val', 'test']:
    for model_label, prefix in baseline_files.items():
        path = os.path.join(RESULTS_DIR, f'{prefix}_{split_name}_per_series.parquet')
        if not os.path.exists(path):
            continue
        ps = pd.read_parquet(path)
        med = ps['wape_overall'].median()
        print(f'  {split_name if model_label == list(baseline_files.keys())[0] else "":<8} '
              f'{model_label:<24} {med:>14.4f}')
    print()

# Pooled table for current model
print(f'  DoW Mean — WAPE pooled:')
for split_name in ['train', 'val', 'test']:
    print(f'    {split_name}: {pooled_results[split_name]["wape_overall"]:.4f} '
          f'(WPE: {pooled_results[split_name]["wpe_overall"]:.4f})')

# ---------------------------------------------------------------------------
# 7. Figure distribuzioni
# ---------------------------------------------------------------------------
print('\n7. Generazione figure distribuzioni...')

# Fig 22: Histograms WAPE_overall e WPE_overall per split
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Day-of-Week Mean Profile — Distribuzione per-serie', fontsize=14)

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
fig.savefig(os.path.join(FIG_DIR, 'fig22_dow_mean_per_series_distributions.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Salvata: fig22_dow_mean_per_series_distributions.png')

# Fig 23: Boxplot comparativi per split
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Day-of-Week Mean Profile — Confronto split', fontsize=14)

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
fig.savefig(os.path.join(FIG_DIR, 'fig23_dow_mean_per_series_boxplots.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Salvata: fig23_dow_mean_per_series_boxplots.png')

# ---------------------------------------------------------------------------
# 8. Boxplot confronto tutti i baseline (test split)
# ---------------------------------------------------------------------------
print('\n8. Generazione boxplot confronto tutti i baseline...')

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Confronto Tutti i Baseline — Distribuzione per-serie', fontsize=15, y=0.98)

all_models = {
    'Global Mean': 'global_mean',
    'DoW Mean': 'dow_mean',
}
colors_all = ['#55A868', '#C44E52']
splits = ['train', 'val', 'test']

for j, split in enumerate(splits):
    # --- WAPE overall ---
    ax = axes[0, j]
    box_data = []
    box_labels = []
    box_colors = []
    medians_raw = []
    for k, (model_label, prefix) in enumerate(all_models.items()):
        path = os.path.join(RESULTS_DIR, f'{prefix}_{split}_per_series.parquet')
        if not os.path.exists(path):
            continue
        ps = pd.read_parquet(path)
        vals = ps['wape_overall'].dropna()
        q99 = vals.quantile(0.99)
        box_data.append(vals.clip(upper=q99))
        box_labels.append(model_label)
        box_colors.append(colors_all[k])
        medians_raw.append(vals.median())

    bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for median_line in bp['medians']:
        median_line.set_color('red')
        median_line.set_linewidth(2)

    for k, med in enumerate(medians_raw):
        ax.text(k + 1, med + 0.01, f'{med:.3f}', ha='center', va='bottom',
                fontsize=8, fontweight='bold', color='red')

    ax.set_title(f'WAPE overall — {split}', fontsize=12)
    ax.set_ylabel('WAPE' if j == 0 else '')
    ax.tick_params(axis='x', rotation=30)

    # --- WPE overall ---
    ax = axes[1, j]
    box_data = []
    box_colors = []
    medians_raw = []
    for k, (model_label, prefix) in enumerate(all_models.items()):
        path = os.path.join(RESULTS_DIR, f'{prefix}_{split}_per_series.parquet')
        if not os.path.exists(path):
            continue
        ps = pd.read_parquet(path)
        vals = ps['wpe_overall'].dropna()
        q01, q99 = vals.quantile(0.01), vals.quantile(0.99)
        box_data.append(vals.clip(lower=q01, upper=q99))
        box_colors.append(colors_all[k])
        medians_raw.append(vals.median())

    bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for median_line in bp['medians']:
        median_line.set_color('red')
        median_line.set_linewidth(2)

    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)

    for k, med in enumerate(medians_raw):
        offset = 0.005 if med >= 0 else -0.005
        va = 'bottom' if med >= 0 else 'top'
        ax.text(k + 1, med + offset, f'{med:.4f}', ha='center', va=va,
                fontsize=8, fontweight='bold', color='red')

    ax.set_title(f'WPE overall — {split}', fontsize=12)
    ax.set_ylabel('WPE' if j == 0 else '')
    ax.tick_params(axis='x', rotation=30)

fig.tight_layout(rect=[0, 0, 1, 0.96])
out_path = os.path.join(FIG_DIR, 'fig24_compare_all_baselines.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  Salvata: {out_path}')

print('\n' + '=' * 72)
print('  DONE')
print('=' * 72)
