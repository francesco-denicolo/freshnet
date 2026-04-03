"""
08b_forecast_naive_completed.py — Fase B2: Naive forecaster su completed_sales
================================================================================
Per ogni imputer × ogni forecaster naive, calcola i profili dai completed_sales
e valuta sul test set (eval HF).

Imputer: media_cond, media_glob, mediana_cond, lgb
Forecaster naive: Global Mean, DoW Mean, MA (K=21)

Per ogni cella:
  1. Carica completed_sales dell'imputer (gg 1-90)
  2. Calcola profilo dalle completed_sales (media/dow/ma)
  3. Predici il test set (eval HF) con profilo fisso (direct forecast)
  4. Valutazione solo ore in-stock

Eseguire con: freshnet/bin/python notebooks_final/08b_forecast_naive_completed.py
"""

import sys
import os
import functools
import numpy as np
import pandas as pd

print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
COMPLETED_DIR = os.path.join(DATA_DIR, 'completed_sales')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

MA_K = 21

IMPUTERS = {
    'media_cond': 'Media condizionata',
    'media_glob': 'Media globale',
    'mediana_cond': 'Mediana condizionata',
    'lgb': 'LGB imputer',
}

FORECASTERS = ['Global Mean', 'DoW Mean', f'MA (K={MA_K})']

# ===========================================================================
print('=' * 72)
print('  FASE B2 — NAIVE FORECASTER SU COMPLETED_SALES')
print('=' * 72)

# ---------------------------------------------------------------------------
# 1. Caricamento dati
# ---------------------------------------------------------------------------
print('\n1. Caricamento dati base...')
df_train_hf = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df_eval = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_eval.parquet'))

df_train_hf['dt_parsed'] = pd.to_datetime(df_train_hf['dt'])
df_eval['dt_parsed'] = pd.to_datetime(df_eval['dt'])

df_full = pd.concat([df_train_hf, df_eval], ignore_index=True)
df_full = df_full.sort_values(['store_id', 'product_id', 'dt_parsed']).reset_index(drop=True)

all_dates = sorted(df_full['dt_parsed'].unique())
date_to_day = {d: i + 1 for i, d in enumerate(all_dates)}
df_full['day_num'] = df_full['dt_parsed'].map(date_to_day)
df_full['dow'] = df_full['dt_parsed'].dt.dayofweek

sales_orig = np.array(df_full['hours_sale'].tolist(), dtype=np.float64)
stock_orig = np.array(df_full['hours_stock_status'].tolist(), dtype=np.int8)

print(f'  Full: {len(df_full):,} righe, {len(all_dates)} giorni')

del df_train_hf, df_eval

# Build series list
print('  Building series list...')
series_list = []
for (sid, pid), grp in df_full.groupby(['store_id', 'product_id'], sort=False):
    grp_s = grp.sort_values('day_num')
    idx = grp_s.index.values
    series_list.append({
        'store_id': sid,
        'product_id': pid,
        'idx': idx,
        'days': df_full.loc[idx, 'day_num'].values,
        'dows': df_full.loc[idx, 'dow'].values,
    })
print(f'  {len(series_list):,} serie')


# ---------------------------------------------------------------------------
# 2. Evaluation helper
# ---------------------------------------------------------------------------
def eval_instock(pred_24, obs_24, stock_24):
    """Compute in-stock hourly + daily metrics for one series chunk."""
    instock = stock_24 == 0
    p_h, o_h = pred_24[instock], obs_24[instock]
    sae_h = np.abs(p_h - o_h).sum()
    sao_h = np.abs(o_h).sum()
    se_h = (p_h - o_h).sum()
    so_h = o_h.sum()

    n_eval = pred_24.shape[0]
    sae_d, sao_d, se_d, so_d, n_vd = 0., 0., 0., 0., 0
    for d in range(n_eval):
        m = instock[d]
        if m.any():
            pv, ov = pred_24[d, m].sum(), obs_24[d, m].sum()
            sae_d += abs(pv - ov); sao_d += abs(ov)
            se_d += pv - ov; so_d += ov; n_vd += 1

    return {
        'sae_h': sae_h, 'sao_h': sao_h, 'se_h': se_h, 'so_h': so_h,
        'sae_d': sae_d, 'sao_d': sao_d, 'se_d': se_d, 'so_d': so_d,
        'n_in': int(instock.sum()), 'n_vd': n_vd,
    }


# ---------------------------------------------------------------------------
# 3. Profile functions
# ---------------------------------------------------------------------------
def global_mean_profile(sales, days, max_day):
    mask = days <= max_day
    return sales[mask].mean(axis=0) if mask.any() else np.zeros(24)


def dow_mean_profiles(sales, days, dows, max_day):
    mask = days <= max_day
    profiles = {}
    for dow in range(7):
        dm = mask & (dows == dow)
        if dm.any():
            profiles[dow] = sales[dm].mean(axis=0)
        else:
            profiles[dow] = sales[mask].mean(axis=0) if mask.any() else np.zeros(24)
    return profiles


def ma_profile(sales, days, anchor_day, K):
    avail = days <= anchor_day
    if not avail.any():
        return np.zeros(24)
    avail_sales = sales[avail]
    k = min(K, len(avail_sales))
    return avail_sales[-k:].mean(axis=0)


# ===========================================================================
# 4. Main loop
# ===========================================================================
all_results = {}

for imp_key, imp_label in IMPUTERS.items():
    print(f'\n{"="*72}')
    print(f'  IMPUTER: {imp_label} ({imp_key})')
    print(f'{"="*72}')

    # Load completed_sales
    cs_path = os.path.join(COMPLETED_DIR, f'{imp_key}.parquet')
    df_cs = pd.read_parquet(cs_path)
    cs_sales_raw = np.array(df_cs['hours_sale'].tolist(), dtype=np.float64)

    # Align with df_full
    print(f'  Allineamento completed_sales...')
    completed_full = sales_orig.copy()
    df_cs['_key'] = df_cs['store_id'].astype(str) + '_' + df_cs['product_id'].astype(str) + '_' + df_cs['dt']
    df_full_key = df_full['store_id'].astype(str) + '_' + df_full['product_id'].astype(str) + '_' + df_full['dt']
    key_to_idx = dict(zip(df_cs['_key'].values, range(len(df_cs))))

    for i in range(len(df_full)):
        k = df_full_key.values[i]
        if k in key_to_idx:
            completed_full[i] = cs_sales_raw[key_to_idx[k]]

    del df_cs, cs_sales_raw, key_to_idx

    # Evaluate all 3 naive forecasters
    # Profiles computed from completed_sales (gg 1-90), test on gg 91-97
    for fc_name in FORECASTERS:
        print(f'\n  --- {fc_name} con {imp_label} ---')

        pooled = {'sae_h': 0., 'sao_h': 0., 'se_h': 0., 'so_h': 0.,
                  'sae_d': 0., 'sao_d': 0., 'se_d': 0., 'so_d': 0.}
        per_series_records = []

        for si, ser in enumerate(series_list):
            if (si + 1) % 10000 == 0:
                print(f'    ... {si+1:,}/{len(series_list):,}')

            idx = ser['idx']
            days = ser['days']
            dows = ser['dows']
            sales_cs = completed_full[idx]   # completed_sales for profiles
            sales_real = sales_orig[idx]     # original S_obs for ground truth
            stock = stock_orig[idx]

            eval_mask = (days >= 91) & (days <= 97)
            if not eval_mask.any():
                continue

            n_eval = eval_mask.sum()
            obs = sales_real[eval_mask]
            stk = stock[eval_mask]
            eval_dows = dows[eval_mask]

            # Compute profile from completed_sales (gg 1-90)
            if fc_name == 'Global Mean':
                prof = global_mean_profile(sales_cs, days, 90)
                pred = np.tile(prof, (n_eval, 1))
            elif fc_name == 'DoW Mean':
                profs = dow_mean_profiles(sales_cs, days, dows, 90)
                pred = np.array([profs[d] for d in eval_dows])
            else:  # MA
                prof = ma_profile(sales_cs, days, 90, MA_K)
                pred = np.tile(prof, (n_eval, 1))

            m = eval_instock(pred, obs, stk)

            # Pooled accumulation
            for k in ['sae_h', 'sao_h', 'se_h', 'so_h',
                       'sae_d', 'sao_d', 'se_d', 'so_d']:
                pooled[k] += m[k]

            # Per-series
            h_wape = m['sae_h'] / m['sao_h'] if m['sao_h'] > 0 else np.nan
            h_wpe = m['se_h'] / m['so_h'] if m['so_h'] != 0 else np.nan
            d_wape = m['sae_d'] / m['sao_d'] if m['sao_d'] > 0 else np.nan
            d_wpe = m['se_d'] / m['so_d'] if m['so_d'] != 0 else np.nan

            per_series_records.append({
                'store_id': ser['store_id'], 'product_id': ser['product_id'],
                'hourly_wape': h_wape, 'hourly_wpe': h_wpe,
                'daily_wape': d_wape, 'daily_wpe': d_wpe,
                'n_hours_instock': m['n_in'], 'n_days_valid': m['n_vd'],
            })

        # Finalize pooled
        pooled_metrics = {
            'hourly_wape': pooled['sae_h'] / pooled['sao_h'] if pooled['sao_h'] > 0 else np.nan,
            'hourly_wpe': pooled['se_h'] / pooled['so_h'] if pooled['so_h'] != 0 else np.nan,
            'daily_wape': pooled['sae_d'] / pooled['sao_d'] if pooled['sao_d'] > 0 else np.nan,
            'daily_wpe': pooled['se_d'] / pooled['so_d'] if pooled['so_d'] != 0 else np.nan,
        }

        ps_df = pd.DataFrame(per_series_records)
        med_metrics = {c: ps_df[c].dropna().median()
                       for c in ['hourly_wape', 'hourly_wpe', 'daily_wape', 'daily_wpe']}

        # Save
        fc_safe = fc_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
        cell_key = f'{imp_key}__{fc_safe}'
        ps_df.to_parquet(os.path.join(RESULTS_DIR, f'{cell_key}_test_per_series.parquet'), index=False)

        all_results[cell_key] = {
            'imputer': imp_label, 'forecaster': fc_name,
            'pooled': pooled_metrics, 'median': med_metrics,
        }

        print(f'    WAPE_h pool={pooled_metrics["hourly_wape"]:.4f}, '
              f'med={med_metrics["hourly_wape"]:.4f}, '
              f'WPE_h pool={pooled_metrics["hourly_wpe"]:.4f}')

    del completed_full


# ===========================================================================
# SUMMARY
# ===========================================================================
print('\n' + '=' * 72)
print('  RIEPILOGO — Naive forecaster su completed_sales (test, in-stock)')
print('=' * 72)

print(f'\n  {"Imputer":<24} {"Forecaster":<16} {"WAPE_h pool":>12} {"WPE_h pool":>11} '
      f'{"WAPE_h med":>11} {"WPE_h med":>10}')
print('  ' + '-' * 88)

for key, res in all_results.items():
    p, m = res['pooled'], res['median']
    print(f'  {res["imputer"]:<24} {res["forecaster"]:<16} '
          f'{p["hourly_wape"]:>12.4f} {p["hourly_wpe"]:>11.4f} '
          f'{m["hourly_wape"]:>11.4f} {m["hourly_wpe"]:>10.4f}')

# Save
summary_df = pd.DataFrame([
    {'imputer': r['imputer'], 'forecaster': r['forecaster'],
     'wape_h_pool': r['pooled']['hourly_wape'], 'wpe_h_pool': r['pooled']['hourly_wpe'],
     'wape_h_med': r['median']['hourly_wape'], 'wpe_h_med': r['median']['hourly_wpe'],
     'wape_d_pool': r['pooled']['daily_wape'], 'wpe_d_pool': r['pooled']['daily_wpe']}
    for r in all_results.values()
])
summary_df.to_parquet(os.path.join(RESULTS_DIR, 'matrix_b2_naive.parquet'), index=False)
print(f'\n  Salvato: matrix_b2_naive.parquet')

print('\n' + '=' * 72)
print('  DONE — 08b_forecast_naive_completed.py')
print('=' * 72)
