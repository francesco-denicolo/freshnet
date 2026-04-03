"""
03_baseline_naive.py — Fase A1: Baseline Naive (senza imputation)
=================================================================
Impatto della Qualità dell'Imputation sul Demand Forecasting
di Prodotti Deperibili — CLAUDE_FINAL.md

4 modelli naive su dati sporchi (S_obs con zeri da stockout):
  1. Global Mean: media per (store, product, hour) su tutto il train
  2. DoW Mean: media per (store, product, dow, hour) su tutto il train
  3. Naive Direct: profilo dell'ultimo giorno → applicato a tutti i giorni test
  4. MA (K giorni): media ultimi K giorni, K selezionato su val

Valutazione:
  - Solo ore in-stock, ground truth = S_obs
  - Metriche: WAPE e WPE (orario pooled, orario mediana per-serie,
              giornaliero pooled, giornaliero mediana per-serie)
  - Split: train gg 1-83, val gg 84-90, test eval HF (gg 91-97)
  - Profili calcolati su gg 1-83 → val, ricalcolati su gg 1-90 → test

Direct forecast: profilo fisso, nessun dato del periodo di forecast come input.

Eseguire con: freshnet/bin/python notebooks_final/03_baseline_naive.py
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

MA_K_CANDIDATES = [3, 5, 7, 10, 14, 21, 28, 42, 56, 83]

# ---------------------------------------------------------------------------
# 1. Caricamento dati
# ---------------------------------------------------------------------------
print('=' * 72)
print('  FASE A1 — BASELINE NAIVE (senza imputation)')
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
df_full['dow'] = df_full['dt_parsed'].dt.dayofweek

n_series = df_full.groupby(['store_id', 'product_id']).ngroups
n_days = len(all_dates)
print(f'  Train: {len(df_train):,} righe, giorni 1-90')
print(f'  Eval:  {len(df_eval):,} righe, giorni 91-97')
print(f'  Full:  {len(df_full):,} righe, {n_days} giorni, {n_series:,} serie')

del df_train, df_eval

# Pre-parse hourly arrays
sales_all = np.array(df_full['hours_sale'].tolist(), dtype=np.float64)       # (N, 24)
stock_all = np.array(df_full['hours_stock_status'].tolist(), dtype=np.int8)  # (N, 24)

# ---------------------------------------------------------------------------
# Helper: compute hourly + daily metrics on in-stock only
# ---------------------------------------------------------------------------
def eval_instock(pred_24, obs_24, stock_24):
    """Compute hourly and daily WAPE/WPE on in-stock hours only.

    Args:
        pred_24: (N_days, 24) predicted hourly values
        obs_24:  (N_days, 24) observed hourly values
        stock_24: (N_days, 24) stock status (0=in-stock, 1=stockout)

    Returns:
        dict with hourly_wape, hourly_wpe, daily_wape, daily_wpe, n_hours, n_days
    """
    instock = stock_24 == 0

    # --- Hourly level ---
    p_h = pred_24[instock]
    o_h = obs_24[instock]
    sum_abs_obs_h = np.abs(o_h).sum()
    sum_obs_h = o_h.sum()

    hourly_wape = np.abs(p_h - o_h).sum() / sum_abs_obs_h if sum_abs_obs_h > 0 else np.nan
    hourly_wpe = (p_h - o_h).sum() / sum_obs_h if sum_obs_h != 0 else np.nan

    # --- Daily level ---
    # For each day, sum pred and obs over in-stock hours only
    n_d = pred_24.shape[0]
    pred_daily = np.zeros(n_d)
    obs_daily = np.zeros(n_d)
    valid_days = np.zeros(n_d, dtype=bool)

    for d in range(n_d):
        mask_d = instock[d]
        if mask_d.any():
            pred_daily[d] = pred_24[d, mask_d].sum()
            obs_daily[d] = obs_24[d, mask_d].sum()
            valid_days[d] = True

    pd_v = pred_daily[valid_days]
    od_v = obs_daily[valid_days]
    sum_abs_obs_d = np.abs(od_v).sum()
    sum_obs_d = od_v.sum()

    daily_wape = np.abs(pd_v - od_v).sum() / sum_abs_obs_d if sum_abs_obs_d > 0 else np.nan
    daily_wpe = (pd_v - od_v).sum() / sum_obs_d if sum_obs_d != 0 else np.nan

    return {
        'hourly_wape': hourly_wape,
        'hourly_wpe': hourly_wpe,
        'daily_wape': daily_wape,
        'daily_wpe': daily_wpe,
        'n_hours_instock': int(instock.sum()),
        'n_days_valid': int(valid_days.sum()),
    }


def aggregate_results(per_series_records):
    """Aggregate per-series records into pooled and median metrics.

    Args:
        per_series_records: list of dicts, each with keys from eval_instock
            plus 'store_id', 'product_id'

    Returns:
        (pooled_dict, median_dict)
    """
    df = pd.DataFrame(per_series_records)

    # Pooled: we need to recompute from raw accumulators
    # The per-series records already have WAPE/WPE ratios,
    # but pooled needs to be volume-weighted (sum of numerators / sum of denominators)
    # We stored partial sums in the accumulator, so here we use per-series WAPE
    # as an approximation... But actually we need the raw sums.
    # --> We'll compute pooled separately in the main loop using accumulators.

    # Median per-serie
    median_dict = {}
    for col in ['hourly_wape', 'hourly_wpe', 'daily_wape', 'daily_wpe']:
        vals = df[col].dropna()
        median_dict[col] = vals.median() if len(vals) > 0 else np.nan

    return median_dict


# ---------------------------------------------------------------------------
# 2. Prepara strutture dati per serie
# ---------------------------------------------------------------------------
print('\n2. Preparazione strutture per serie...')

groups = df_full.groupby(['store_id', 'product_id'], sort=False)
n_groups = len(groups)

# Pre-build per-series arrays (indices into df_full)
series_list = []
for (sid, pid), grp in groups:
    grp_s = grp.sort_values('day_num')
    idx = grp_s.index.values
    series_list.append({
        'store_id': sid,
        'product_id': pid,
        'idx': idx,
        'days': df_full.loc[idx, 'day_num'].values,
        'dows': df_full.loc[idx, 'dow'].values,
    })

print(f'  {len(series_list):,} serie preparate')


# ---------------------------------------------------------------------------
# 3. Funzioni per i 4 modelli naive
# ---------------------------------------------------------------------------
def global_mean_profile(sales, days, max_day):
    """Compute mean hourly profile over days <= max_day."""
    mask = days <= max_day
    if not mask.any():
        return np.zeros(24)
    return sales[mask].mean(axis=0)


def dow_mean_profiles(sales, days, dows, max_day):
    """Compute mean hourly profile per DoW over days <= max_day.
    Returns: dict {dow: profile(24,)}
    """
    mask = days <= max_day
    profiles = {}
    for dow in range(7):
        dow_mask = mask & (dows == dow)
        if dow_mask.any():
            profiles[dow] = sales[dow_mask].mean(axis=0)
        else:
            # Fallback: global mean
            profiles[dow] = sales[mask].mean(axis=0) if mask.any() else np.zeros(24)
    return profiles


def naive_direct_profile(sales, days, anchor_day):
    """Profile = S_obs of anchor_day."""
    mask = days == anchor_day
    if mask.any():
        return sales[mask][0]
    # Fallback: last available day
    avail = days <= anchor_day
    if avail.any():
        return sales[avail][-1]
    return np.zeros(24)


def ma_profile(sales, days, anchor_day, K):
    """Mean of last K days up to anchor_day."""
    avail = days <= anchor_day
    if not avail.any():
        return np.zeros(24)
    avail_sales = sales[avail]
    k = min(K, len(avail_sales))
    return avail_sales[-k:].mean(axis=0)


# ---------------------------------------------------------------------------
# 4. Evaluate all naive models
# ---------------------------------------------------------------------------
print('\n3. Valutazione modelli naive...')

# We'll compute for val and test splits
# Val: profiles from gg 1-83, evaluate on gg 84-90
# Test: profiles from gg 1-90, evaluate on gg 91-97

SPLITS = {
    'val': {'profile_max_day': 83, 'eval_min': 84, 'eval_max': 90,
            'naive_anchor': 83},
    'test': {'profile_max_day': 90, 'eval_min': 91, 'eval_max': 97,
             'naive_anchor': 90},
}

# For MA, first select K on val
print('\n  3a. Selezione K per MA su validation...')

ma_val_results = {}
for K in MA_K_CANDIDATES:
    # Pooled accumulators for val
    acc = {'sae_h': 0., 'sao_h': 0., 'se_h': 0., 'so_h': 0.,
           'sae_d': 0., 'sao_d': 0., 'se_d': 0., 'so_d': 0.}

    for si, ser in enumerate(series_list):
        idx = ser['idx']
        days = ser['days']
        dows = ser['dows']
        sales = sales_all[idx]
        stock = stock_all[idx]

        eval_mask = (days >= 84) & (days <= 90)
        if not eval_mask.any():
            continue

        profile = ma_profile(sales, days, 83, K)
        n_eval = eval_mask.sum()
        pred = np.tile(profile, (n_eval, 1))
        obs = sales[eval_mask]
        stk = stock[eval_mask]

        instock = stk == 0

        # Hourly accumulators
        p_h = pred[instock]
        o_h = obs[instock]
        acc['sae_h'] += np.abs(p_h - o_h).sum()
        acc['sao_h'] += np.abs(o_h).sum()
        acc['se_h'] += (p_h - o_h).sum()
        acc['so_h'] += o_h.sum()

        # Daily accumulators
        for d in range(n_eval):
            m_d = instock[d]
            if m_d.any():
                pd_v = pred[d, m_d].sum()
                od_v = obs[d, m_d].sum()
                acc['sae_d'] += abs(pd_v - od_v)
                acc['sao_d'] += abs(od_v)
                acc['se_d'] += pd_v - od_v
                acc['so_d'] += od_v

    wape_h = acc['sae_h'] / acc['sao_h'] if acc['sao_h'] > 0 else np.nan
    wpe_h = acc['se_h'] / acc['so_h'] if acc['so_h'] != 0 else np.nan
    wape_d = acc['sae_d'] / acc['sao_d'] if acc['sao_d'] > 0 else np.nan
    ma_val_results[K] = {'hourly_wape': wape_h, 'hourly_wpe': wpe_h,
                          'daily_wape': wape_d}
    print(f'    K={K:>3}: WAPE_h={wape_h:.4f}, WPE_h={wpe_h:.4f}, WAPE_d={wape_d:.4f}')

# Select K with best hourly WAPE pooled on val
BEST_K = min(ma_val_results, key=lambda k: ma_val_results[k]['hourly_wape'])
print(f'\n  Best K (hourly WAPE pooled): K={BEST_K}')

# ---------------------------------------------------------------------------
# 5. Full evaluation for all 4 models on val and test
# ---------------------------------------------------------------------------
print('\n4. Valutazione completa (val + test)...')

MODELS = ['Global Mean', 'DoW Mean', 'Naive Direct', f'MA (K={BEST_K})']
all_results = {m: {} for m in MODELS}

for split_name, sp in SPLITS.items():
    print(f'\n  --- Split: {split_name} ---')
    max_day = sp['profile_max_day']
    d_min = sp['eval_min']
    d_max = sp['eval_max']
    anchor = sp['naive_anchor']

    # Per-model pooled accumulators
    pooled = {m: {'sae_h': 0., 'sao_h': 0., 'se_h': 0., 'so_h': 0.,
                   'sae_d': 0., 'sao_d': 0., 'se_d': 0., 'so_d': 0.}
              for m in MODELS}

    # Per-series records
    per_series = {m: [] for m in MODELS}

    for si, ser in enumerate(series_list):
        if (si + 1) % 10000 == 0:
            print(f'    ... {si+1:,}/{n_groups:,} serie')

        idx = ser['idx']
        days = ser['days']
        dows = ser['dows']
        sales = sales_all[idx]
        stock = stock_all[idx]

        eval_mask = (days >= d_min) & (days <= d_max)
        if not eval_mask.any():
            continue

        n_eval = eval_mask.sum()
        obs = sales[eval_mask]
        stk = stock[eval_mask]
        eval_dows = dows[eval_mask]

        # Compute profiles for each model
        profiles = {}

        # Global Mean
        profiles['Global Mean'] = np.tile(
            global_mean_profile(sales, days, max_day), (n_eval, 1))

        # DoW Mean
        dow_profs = dow_mean_profiles(sales, days, dows, max_day)
        profiles['DoW Mean'] = np.array([dow_profs[d] for d in eval_dows])

        # Naive Direct
        profiles['Naive Direct'] = np.tile(
            naive_direct_profile(sales, days, anchor), (n_eval, 1))

        # MA
        profiles[f'MA (K={BEST_K})'] = np.tile(
            ma_profile(sales, days, anchor, BEST_K), (n_eval, 1))

        # Evaluate each model
        for model_name, pred in profiles.items():
            instock = stk == 0

            # --- Hourly ---
            p_h = pred[instock]
            o_h = obs[instock]
            sae_h = np.abs(p_h - o_h).sum()
            sao_h = np.abs(o_h).sum()
            se_h = (p_h - o_h).sum()
            so_h = o_h.sum()

            pooled[model_name]['sae_h'] += sae_h
            pooled[model_name]['sao_h'] += sao_h
            pooled[model_name]['se_h'] += se_h
            pooled[model_name]['so_h'] += so_h

            h_wape = sae_h / sao_h if sao_h > 0 else np.nan
            h_wpe = se_h / so_h if so_h != 0 else np.nan

            # --- Daily ---
            sae_d_s = 0.
            sao_d_s = 0.
            se_d_s = 0.
            so_d_s = 0.
            n_valid_d = 0

            for d in range(n_eval):
                m_d = instock[d]
                if m_d.any():
                    pd_v = pred[d, m_d].sum()
                    od_v = obs[d, m_d].sum()
                    sae_d_s += abs(pd_v - od_v)
                    sao_d_s += abs(od_v)
                    se_d_s += pd_v - od_v
                    so_d_s += od_v
                    n_valid_d += 1

            pooled[model_name]['sae_d'] += sae_d_s
            pooled[model_name]['sao_d'] += sao_d_s
            pooled[model_name]['se_d'] += se_d_s
            pooled[model_name]['so_d'] += so_d_s

            d_wape = sae_d_s / sao_d_s if sao_d_s > 0 else np.nan
            d_wpe = se_d_s / so_d_s if so_d_s != 0 else np.nan

            per_series[model_name].append({
                'store_id': ser['store_id'],
                'product_id': ser['product_id'],
                'hourly_wape': h_wape,
                'hourly_wpe': h_wpe,
                'daily_wape': d_wape,
                'daily_wpe': d_wpe,
                'n_hours_instock': int(instock.sum()),
                'n_days_valid': n_valid_d,
            })

    # Finalize pooled and per-series
    for model_name in MODELS:
        acc = pooled[model_name]
        pooled_metrics = {
            'hourly_wape': acc['sae_h'] / acc['sao_h'] if acc['sao_h'] > 0 else np.nan,
            'hourly_wpe': acc['se_h'] / acc['so_h'] if acc['so_h'] != 0 else np.nan,
            'daily_wape': acc['sae_d'] / acc['sao_d'] if acc['sao_d'] > 0 else np.nan,
            'daily_wpe': acc['se_d'] / acc['so_d'] if acc['so_d'] != 0 else np.nan,
        }

        ps_df = pd.DataFrame(per_series[model_name])
        median_metrics = {}
        for col in ['hourly_wape', 'hourly_wpe', 'daily_wape', 'daily_wpe']:
            vals = ps_df[col].dropna()
            median_metrics[col] = vals.median() if len(vals) > 0 else np.nan

        all_results[model_name][split_name] = {
            'pooled': pooled_metrics,
            'median': median_metrics,
            'per_series_df': ps_df,
        }

        # Save per-series parquet
        safe_name = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
        out_path = os.path.join(RESULTS_DIR, f'naive_{safe_name}_{split_name}_per_series.parquet')
        ps_df.to_parquet(out_path, index=False)

# ---------------------------------------------------------------------------
# 6. Print results
# ---------------------------------------------------------------------------
print('\n' + '=' * 72)
print('  5. RISULTATI — BASELINE NAIVE (solo ore in-stock)')
print('=' * 72)

print(f'\n  {"Model":<20} {"Split":<6} '
      f'{"WAPE_h pool":>12} {"WPE_h pool":>11} '
      f'{"WAPE_h med":>11} {"WPE_h med":>10} '
      f'{"WAPE_d pool":>12} {"WPE_d pool":>11} '
      f'{"WAPE_d med":>11} {"WPE_d med":>10}')
print('  ' + '-' * 126)

for model_name in MODELS:
    for split_name in ['val', 'test']:
        r = all_results[model_name][split_name]
        p = r['pooled']
        m = r['median']
        label = model_name if split_name == 'val' else ''
        print(f'  {label:<20} {split_name:<6} '
              f'{p["hourly_wape"]:>12.4f} {p["hourly_wpe"]:>11.4f} '
              f'{m["hourly_wape"]:>11.4f} {m["hourly_wpe"]:>10.4f} '
              f'{p["daily_wape"]:>12.4f} {p["daily_wpe"]:>11.4f} '
              f'{m["daily_wape"]:>11.4f} {m["daily_wpe"]:>10.4f}')
    print()

# ---------------------------------------------------------------------------
# 7. MA K selection table
# ---------------------------------------------------------------------------
print('\n' + '=' * 72)
print('  6. MA — SELEZIONE K (validation)')
print('=' * 72)

print(f'\n  {"K":>4} {"WAPE_h pool":>12} {"WPE_h pool":>11} {"WAPE_d pool":>12} {"selected":>10}')
print('  ' + '-' * 52)
for K in MA_K_CANDIDATES:
    r = ma_val_results[K]
    sel = ' <<<' if K == BEST_K else ''
    print(f'  {K:>4} {r["hourly_wape"]:>12.4f} {r["hourly_wpe"]:>11.4f} '
          f'{r["daily_wape"]:>12.4f} {sel}')

# ---------------------------------------------------------------------------
# 8. Summary table (test only, for the matrix)
# ---------------------------------------------------------------------------
print('\n' + '=' * 72)
print('  7. RIEPILOGO TEST — Prima riga della matrice')
print('=' * 72)

print(f'\n  {"Model":<20} '
      f'{"WAPE_h pool":>12} {"WPE_h pool":>11} '
      f'{"WAPE_h med":>11} {"WPE_h med":>10}')
print('  ' + '-' * 68)

for model_name in MODELS:
    r = all_results[model_name]['test']
    p = r['pooled']
    m = r['median']
    print(f'  {model_name:<20} '
          f'{p["hourly_wape"]:>12.4f} {p["hourly_wpe"]:>11.4f} '
          f'{m["hourly_wape"]:>11.4f} {m["hourly_wpe"]:>10.4f}')

# ---------------------------------------------------------------------------
# 9. Figures
# ---------------------------------------------------------------------------
print('\n8. Generazione figure...')

# Fig 1: Bar chart comparison (test, hourly, pooled + median)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Fase A1 — Baseline Naive (test, in-stock only)', fontsize=14)

x = np.arange(len(MODELS))
w = 0.35

# Pooled
wape_pool = [all_results[m]['test']['pooled']['hourly_wape'] for m in MODELS]
wpe_pool = [all_results[m]['test']['pooled']['hourly_wpe'] for m in MODELS]

ax = axes[0]
bars = ax.bar(x, wape_pool, w, color='steelblue', alpha=0.8, label='WAPE')
ax.set_ylabel('WAPE (pooled)')
ax.set_title('Hourly WAPE — pooled')
ax.set_xticks(x)
ax.set_xticklabels(MODELS, rotation=25, ha='right', fontsize=9)
for i, v in enumerate(wape_pool):
    ax.text(i, v + 0.005, f'{v:.4f}', ha='center', va='bottom', fontsize=8)

# Median per-serie
wape_med = [all_results[m]['test']['median']['hourly_wape'] for m in MODELS]

ax = axes[1]
bars = ax.bar(x, wape_med, w, color='darkorange', alpha=0.8, label='WAPE')
ax.set_ylabel('WAPE (median per-serie)')
ax.set_title('Hourly WAPE — median per-serie')
ax.set_xticks(x)
ax.set_xticklabels(MODELS, rotation=25, ha='right', fontsize=9)
for i, v in enumerate(wape_med):
    ax.text(i, v + 0.005, f'{v:.4f}', ha='center', va='bottom', fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig01_naive_baselines_test.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Salvata: fig01_naive_baselines_test.png')

# Fig 2: Boxplot per-serie WAPE hourly (test)
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Fase A1 — Distribuzione WAPE orario per-serie (test, in-stock)', fontsize=13)

box_data = []
for model_name in MODELS:
    ps = all_results[model_name]['test']['per_series_df']
    vals = ps['hourly_wape'].dropna()
    box_data.append(vals.clip(upper=vals.quantile(0.99)).values)

bp = ax.boxplot(box_data, tick_labels=MODELS, patch_artist=True, widths=0.6)
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
for ml in bp['medians']:
    ml.set_color('red')
    ml.set_linewidth(2)

# Annotate medians
for i, model_name in enumerate(MODELS):
    med = all_results[model_name]['test']['median']['hourly_wape']
    ax.text(i + 1, med + 0.01, f'{med:.4f}', ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='red')

ax.set_ylabel('WAPE (hourly, in-stock)')
ax.tick_params(axis='x', rotation=20)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig02_naive_boxplot_test.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Salvata: fig02_naive_boxplot_test.png')

print('\n' + '=' * 72)
print('  DONE — 03_baseline_naive.py')
print('=' * 72)
