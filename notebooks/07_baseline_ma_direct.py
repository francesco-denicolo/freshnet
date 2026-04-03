"""
07_baseline_ma_direct.py — Moving Average Direct Forecast
==========================================================
PINN-Retail: Physics-Informed Neural Networks per Demand Forecasting
di Prodotti Deperibili.

Passo 3f: baseline media mobile in modalità direct forecast.
Predizione: per ogni serie, un unico profilo = media degli ultimi K giorni
prima dell'orizzonte di forecast, applicato a TUTTI i giorni dell'orizzonte.

K selection su val (giorni 84-90) in modalità direct forecast:
  profilo_val = mean(S_obs(giorno 84-K)...S_obs(giorno 83))
  applicato identicamente a tutti i 7 giorni di val.

Test (giorni 91-97) con K*:
  profilo_test = mean(S_obs(giorno 91-K*)...S_obs(giorno 90))
  (include giorni di val — standard: dopo model selection si usa train+val)

Eseguire con: freshnet/bin/python notebooks/07_baseline_ma_direct.py
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

K_CANDIDATES = [1, 3, 5, 7, 14, 21, 28, 42, 63, 83]

# ---------------------------------------------------------------------------
# 1. Caricamento dati
# ---------------------------------------------------------------------------
print('=' * 72)
print('  MOVING AVERAGE — DIRECT FORECAST')
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
print(f'  K candidati: {K_CANDIDATES}')

del df_train, df_eval

# ---------------------------------------------------------------------------
# 2. K selection on validation (direct forecast)
# ---------------------------------------------------------------------------
print('\n2. K selection su val (giorni 84-90) in modalità direct forecast...')
print('   Per ogni K: profilo = media ultimi K giorni di training (fino a g.83)')
print('   Profilo applicato identicamente a tutti i 7 giorni di val.\n')

# Accumulators per K
pooled_val = {
    K: {'sae': 0., 'sao': 0., 'se': 0., 'so': 0.}
    for K in K_CANDIDATES
}
ps_wapes_val = {K: [] for K in K_CANDIDATES}

groups = df_full.groupby(['store_id', 'product_id'], sort=False)
n_groups = len(groups)

for i, ((sid, pid), grp) in enumerate(groups):
    if (i + 1) % 10000 == 0:
        print(f'    ... {i+1:,}/{n_groups:,} serie')

    grp_s = grp.sort_values('day_num')
    sales = np.array(grp_s['hours_sale'].tolist(), dtype=np.float64)  # (N, 24)
    stock = np.array(grp_s['hours_stock_status'].tolist())
    days = grp_s['day_num'].values

    # Val target
    val_mask = (days >= 84) & (days <= 90)
    if not val_mask.any():
        for K in K_CANDIDATES:
            ps_wapes_val[K].append(np.nan)
        continue

    val_obs = sales[val_mask]
    n_val = val_obs.shape[0]

    for K in K_CANDIDATES:
        # Profile: mean of last K days before val (days 84-K ... 83)
        anchor_day = 83
        start_day = anchor_day - K + 1
        if start_day < 1:
            start_day = 1

        hist_mask = (days >= start_day) & (days <= anchor_day)
        if not hist_mask.any():
            ps_wapes_val[K].append(np.nan)
            continue

        profile = sales[hist_mask].mean(axis=0)  # (24,)
        preds = np.tile(profile, (n_val, 1))

        err = preds.ravel() - val_obs.ravel()
        abs_obs = np.abs(val_obs.ravel())

        pooled_val[K]['sae'] += np.abs(err).sum()
        pooled_val[K]['sao'] += abs_obs.sum()
        pooled_val[K]['se'] += err.sum()
        pooled_val[K]['so'] += val_obs.ravel().sum()

        s_abs_obs = np.abs(val_obs).sum()
        if s_abs_obs > 0:
            ps_wapes_val[K].append(np.abs(preds - val_obs).sum() / s_abs_obs)
        else:
            ps_wapes_val[K].append(np.nan)

# K selection table
print(f'\n  {"K":>4} {"WAPE_pooled":>14} {"WAPE_med_ps":>14} {"WPE_pooled":>12}')
print('  ' + '-' * 48)

k_selection = {}
for K in K_CANDIDATES:
    pa = pooled_val[K]
    wape_p = pa['sae'] / pa['sao'] if pa['sao'] > 0 else np.nan
    wpe_p = pa['se'] / pa['so'] if pa['so'] > 0 else np.nan
    wape_m = np.nanmedian(ps_wapes_val[K])

    k_selection[K] = {'wape_pooled': wape_p, 'wape_median': wape_m, 'wpe_pooled': wpe_p}
    print(f'  {K:>4} {wape_p:>14.6f} {wape_m:>14.6f} {wpe_p:>12.6f}')

best_K_pooled = min(K_CANDIDATES, key=lambda k: k_selection[k]['wape_pooled'])
best_K_median = min(K_CANDIDATES, key=lambda k: k_selection[k]['wape_median'])

print(f'\n  Best K (WAPE pooled):           K={best_K_pooled} '
      f'(WAPE={k_selection[best_K_pooled]["wape_pooled"]:.6f})')
print(f'  Best K (WAPE median per-serie): K={best_K_median} '
      f'(WAPE={k_selection[best_K_median]["wape_median"]:.6f})')

if best_K_pooled == best_K_median:
    K_star = best_K_pooled
    print(f'\n  Entrambi i criteri concordano: K*={K_star}')
else:
    K_star = best_K_pooled
    print(f'\n  Criteri discordanti. Uso K*={K_star} (pooled).')
    print(f'  K={best_K_median} (median) riportato per confronto.')

# ---------------------------------------------------------------------------
# 3. K selection plot
# ---------------------------------------------------------------------------
print('\n3. Plot K selection...')

fig, ax = plt.subplots(figsize=(8, 5))
ks = K_CANDIDATES
wape_pooled_vals = [k_selection[k]['wape_pooled'] for k in ks]
wape_median_vals = [k_selection[k]['wape_median'] for k in ks]

ax.plot(ks, wape_pooled_vals, 'o-', color='steelblue', label='WAPE pooled (val)',
        linewidth=2, markersize=8)
ax.plot(ks, wape_median_vals, 's--', color='darkorange', label='WAPE median per-serie (val)',
        linewidth=2, markersize=8)
ax.axvline(best_K_pooled, color='steelblue', linestyle=':', alpha=0.5,
           label=f'Best pooled: K={best_K_pooled}')
if best_K_median != best_K_pooled:
    ax.axvline(best_K_median, color='darkorange', linestyle=':', alpha=0.5,
               label=f'Best median: K={best_K_median}')
ax.set_xlabel('K (window size)')
ax.set_ylabel('WAPE on validation (direct forecast)')
ax.set_title('MA Direct Forecast — K Selection on Validation')
ax.legend()
ax.set_xticks(ks)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig27_ma_direct_k_selection.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Salvata: fig27_ma_direct_k_selection.png')

# ---------------------------------------------------------------------------
# 4. Full evaluation for K* on test (direct forecast)
# ---------------------------------------------------------------------------
print(f'\n4. Valutazione test con K*={K_star} (direct forecast)...')
print(f'   Profilo test = media ultimi {K_star} giorni prima del test (fino a g.90)')

pooled_test = {
    sub: {'sae': 0., 'sao': 0., 'se': 0., 'so': 0., 'n': 0}
    for sub in ['overall', 'instock', 'stockout']
}
per_series_test = []

# Also recompute val with K* for completeness
pooled_val_kstar = {
    sub: {'sae': 0., 'sao': 0., 'se': 0., 'so': 0., 'n': 0}
    for sub in ['overall', 'instock', 'stockout']
}
per_series_val = []

for i, ((sid, pid), grp) in enumerate(groups):
    if (i + 1) % 10000 == 0:
        print(f'    ... {i+1:,}/{n_groups:,} serie')

    grp_s = grp.sort_values('day_num')
    sales = np.array(grp_s['hours_sale'].tolist(), dtype=np.float64)
    stock = np.array(grp_s['hours_stock_status'].tolist())
    days = grp_s['day_num'].values

    for split_name, anchor_day, d_min, d_max, pooled_acc, ps_list in [
        ('val', 83, 84, 90, pooled_val_kstar, per_series_val),
        ('test', 90, 91, 97, pooled_test, per_series_test),
    ]:
        # Profile: mean of last K* days before the horizon
        start_day = max(1, anchor_day - K_star + 1)
        hist_mask = (days >= start_day) & (days <= anchor_day)
        target_mask = (days >= d_min) & (days <= d_max)

        if not hist_mask.any() or not target_mask.any():
            continue

        profile = sales[hist_mask].mean(axis=0)
        n_target = target_mask.sum()
        p = np.tile(profile, (n_target, 1))
        o = sales[target_mask]
        s = stock[target_mask]

        p_flat = p.ravel()
        o_flat = o.ravel()
        s_flat = s.ravel()
        err = p_flat - o_flat

        for sub, smask in [('overall', np.ones(len(p_flat), dtype=bool)),
                           ('instock', s_flat == 0),
                           ('stockout', s_flat == 1)]:
            acc = pooled_acc[sub]
            ef = err[smask]
            of = o_flat[smask]
            acc['sae'] += np.abs(ef).sum()
            acc['sao'] += np.abs(of).sum()
            acc['se'] += ef.sum()
            acc['so'] += of.sum()
            acc['n'] += int(smask.sum())

        m = compute_metrics(p, o, s)
        m['store_id'] = sid
        m['product_id'] = pid
        ps_list.append(m)

# Build results
pooled_results = {}
for split_name, pacc in [('val', pooled_val_kstar), ('test', pooled_test)]:
    r = {}
    for sub in ['overall', 'instock', 'stockout']:
        acc = pacc[sub]
        r[f'wape_{sub}'] = acc['sae'] / acc['sao'] if acc['sao'] > 0 else np.nan
        r[f'wpe_{sub}'] = acc['se'] / acc['so'] if acc['so'] > 0 else np.nan
        r[f'n_{sub}'] = acc['n']
    pooled_results[split_name] = r

per_series_dfs = {}
for split_name, records in [('val', per_series_val), ('test', per_series_test)]:
    ps = pd.DataFrame(records)
    per_series_dfs[split_name] = ps
    out_path = os.path.join(RESULTS_DIR, f'ma_direct_K{K_star}_{split_name}_per_series.parquet')
    ps.to_parquet(out_path, index=False)
    print(f'  Salvato: {out_path} ({len(ps):,} serie)')

# ---------------------------------------------------------------------------
# 5. Tabella risultati
# ---------------------------------------------------------------------------
print(format_metrics_table(pooled_results, model_name=f'MA Direct Forecast K={K_star}'))

# Per-series distribution
METRIC_COLS = ['wape_overall', 'wape_instock', 'wape_stockout',
               'wpe_overall', 'wpe_instock', 'wpe_stockout']
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]

print(f'\n  Distribuzione per-serie (K={K_star}):')
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
print(f'\n6. Generazione figure (K={K_star})...')

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(f'MA Direct Forecast K={K_star} — Distribuzione per-serie', fontsize=14)

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
fig.savefig(os.path.join(FIG_DIR, f'fig28_ma_direct_K{K_star}_distributions.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  Salvata: fig28_ma_direct_K{K_star}_distributions.png')

# ---------------------------------------------------------------------------
# 7. Confronto con tutti i baseline
# ---------------------------------------------------------------------------
print('\n' + '=' * 72)
print('  7. CONFRONTO CON TUTTI I BASELINE (test)')
print('=' * 72)

all_baselines = {
    'Global Mean': 'global_mean',
    'DoW Mean': 'dow_mean',
    'Naive (direct)': 'naive_direct',
    f'MA K={K_star} (direct)': f'ma_direct_K{K_star}',
}

print(f'\n  {"Model":<24} {"WAPE_med_ps":>14} {"WAPE_instock":>14}')
print('  ' + '-' * 56)

for label, prefix in all_baselines.items():
    path = os.path.join(RESULTS_DIR, f'{prefix}_test_per_series.parquet')
    if not os.path.exists(path):
        continue
    ps = pd.read_parquet(path)
    med_all = ps['wape_overall'].median()
    med_in = ps['wape_instock'].median()
    print(f'  {label:<24} {med_all:>14.4f} {med_in:>14.4f}')

print('\n' + '=' * 72)
print('  DONE')
print('=' * 72)
