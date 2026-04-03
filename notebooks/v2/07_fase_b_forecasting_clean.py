"""
07_fase_b_forecasting_clean.py — Fase B: Forecasting su dati puliti (completed_sales)
=====================================================================================
Piano: CLAUDE_SEQUENTIAL-2.md, Fase B, punti (4) e (5)

Two-stage forecasting: profili e lag features calcolati da completed_sales
(ore in-stock = S_obs originale, ore stockout = D_hat da LGB Imputer).

Modelli:
  Naive:  Global Mean, DoW Mean, Naive Direct, MA Direct (K=14)
  ML/DL:  LGB F (M5 lags), MLP F (M5 lags)

Differenze da Fase A:
  - Profili naive calcolati da completed_sales (include domanda imputata a stockout)
  - Lag features calcolati da completed_sales (decontaminati da zeri di stockout)
  - Target training LGB/MLP = completed_Y (non S_obs)
  - Evaluation = vs S_obs originale + stock_status (come Fase A)

Workflow:
  1. TUNING:     Train gg 2-83, val gg 84-90 (target = completed_Y, eval vs S_obs)
  2. RETRAINING: Retrain su gg 2-90 con best_iter/best_epoch
  3. TEST:       Eval su gg 91-97 (eval HF) vs S_obs originale

Output:
  notebooks/v2/results/clean_<model>_{val|test}_per_series.parquet

Eseguire con: freshnet/bin/python notebooks/v2/07_fase_b_forecasting_clean.py
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import functools

print = functools.partial(print, flush=True)

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

from src.evaluation.metrics import compute_metrics

import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---- Config ----
DEVICE = 'mps' if torch.backends.mps.is_available() else (
    'cuda' if torch.cuda.is_available() else 'cpu')

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

CONT_COLS = ['discount', 'avg_temperature', 'avg_humidity', 'precpt',
             'avg_wind_level', 'holiday_flag', 'activity_flag']

# LGB
LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.3,
    'bagging_freq': 1,
    'min_child_samples': 500,
    'max_bin': 127,
    'verbose': -1,
    'num_threads': -1,
    'seed': 42,
}
MAX_BOOST_ROUNDS = 500
EARLY_STOPPING_ROUNDS = 30

# MLP
BATCH_SIZE = 4096
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 10
HIDDEN_SIZES = [128, 64]
EMB_DIMS = {'store_id': 32, 'product_id': 32, 'city_id': 8, 'dow': 4}
CARDINALITIES = {'store_id': 898, 'product_id': 865, 'city_id': 18, 'dow': 7}

# MA Direct
K_STAR = 14  # From Fase A selection


def _accumulate_pooled(acc_dict, preds, obs, stock):
    """Accumulate pooled WAPE/WPE statistics."""
    p_flat = preds.ravel()
    o_flat = obs.ravel()
    s_flat = stock.ravel()
    err = p_flat - o_flat
    for sub, smask in [('overall', np.ones(len(p_flat), dtype=bool)),
                       ('instock', s_flat == 0),
                       ('stockout', s_flat == 1)]:
        acc = acc_dict[sub]
        ef = err[smask]
        of = o_flat[smask]
        acc['sae'] += np.abs(ef).sum()
        acc['sao'] += np.abs(of).sum()
        acc['se'] += ef.sum()
        acc['so'] += of.sum()
        acc['n'] += int(smask.sum())


print("=" * 72)
print("  FASE B — FORECASTING SU DATI PULITI (completed_sales)")
print("=" * 72)
print(f"  Device: {DEVICE}")

# =========================================================================
# 1. Caricamento dati
# =========================================================================
t0 = time.time()
print("\n1. Caricamento dati...")

# completed_sales (days 1-90, con D_hat alle ore stockout)
df_completed = pd.read_parquet(os.path.join(DATA_DIR, 'completed_sales.parquet'))
df_completed['dt_parsed'] = pd.to_datetime(df_completed['dt'])

# eval HF (days 91-97, original)
df_eval = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_eval.parquet'))
df_eval['dt_parsed'] = pd.to_datetime(df_eval['dt'])

# Also load original train for stock_status
df_train_orig = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df_train_orig['dt_parsed'] = pd.to_datetime(df_train_orig['dt'])

# Merge to get a full dataset with all needed columns
# For days 1-90: completed_sales + original stock
# For days 91-97: original eval
all_dates = sorted(set(df_completed['dt_parsed'].unique()) |
                   set(df_eval['dt_parsed'].unique()))
date_to_day = {d: i + 1 for i, d in enumerate(all_dates)}

df_completed['day_num'] = df_completed['dt_parsed'].map(date_to_day)
df_completed['dow'] = df_completed['dt_parsed'].dt.dayofweek
df_eval['day_num'] = df_eval['dt_parsed'].map(date_to_day)
df_eval['dow'] = df_eval['dt_parsed'].dt.dayofweek
df_train_orig['day_num'] = df_train_orig['dt_parsed'].map(date_to_day)

n_days = len(all_dates)
print(f"  completed_sales: {len(df_completed):,} righe, giorni 1-90")
print(f"  eval HF: {len(df_eval):,} righe, giorni 91-97")
print(f"  Total: {n_days} giorni")

# Parse arrays
csales_arr = np.array(df_completed['hours_completed_sale'].tolist(), dtype=np.float32)
osales_arr = np.array(df_completed['hours_sale_original'].tolist(), dtype=np.float32)
stock_train_arr = np.array(df_completed['hours_stock_status'].tolist(), dtype=np.float32)

eval_sales_arr = np.array(df_eval['hours_sale'].tolist(), dtype=np.float32)
eval_stock_arr = np.array(df_eval['hours_stock_status'].tolist(), dtype=np.float32)

print(f"  Tempo loading: {time.time()-t0:.1f}s")

# =========================================================================
# 2. Build series cache
# =========================================================================
print("\n2. Building series cache...")
t1 = time.time()

# For each series: completed_sales for days 1-90, original for 91-97
series_data = {}

# Index completed sales by (store_id, product_id)
comp_groups = df_completed.groupby(['store_id', 'product_id'], sort=False)
eval_groups = df_eval.groupby(['store_id', 'product_id'], sort=False)
orig_groups = df_train_orig.groupby(['store_id', 'product_id'], sort=False)

# Get continuous features from original train
orig_cont_cache = {}
for (sid, pid), grp in orig_groups:
    grp_s = grp.sort_values('day_num')
    orig_cont_cache[(sid, pid)] = {
        'city_id': grp_s['city_id'].values[0],
        'conts': grp_s[CONT_COLS].values.astype(np.float32),
        'days': grp_s['day_num'].values,
    }

del df_train_orig

for (sid, pid), grp in comp_groups:
    grp_s = grp.sort_values('day_num')
    idx = grp_s.index.values

    days_comp = grp_s['day_num'].values
    dows_comp = grp_s['dow'].values

    # Get eval data for this series
    eval_grp = eval_groups.get_group((sid, pid)) if (sid, pid) in eval_groups.groups else None
    if eval_grp is not None:
        eval_s = eval_grp.sort_values('day_num')
        eval_idx = eval_s.index.values
        days_eval = eval_s['day_num'].values
        dows_eval = eval_s['dow'].values
        sales_eval = eval_sales_arr[eval_idx]
        stock_eval = eval_stock_arr[eval_idx]
    else:
        days_eval = np.array([], dtype=np.int64)
        dows_eval = np.array([], dtype=np.int64)
        sales_eval = np.zeros((0, 24), dtype=np.float32)
        stock_eval = np.zeros((0, 24), dtype=np.float32)

    # Combine
    all_days = np.concatenate([days_comp, days_eval])
    all_dows = np.concatenate([dows_comp, dows_eval])
    # completed_sales for days 1-90, original for 91-97
    all_completed = np.concatenate([csales_arr[idx], sales_eval])
    # original S_obs for ALL days (for evaluation)
    all_original = np.concatenate([osales_arr[idx], sales_eval])
    # stock status
    all_stock = np.concatenate([stock_train_arr[idx], stock_eval])

    # Continuous features
    oc = orig_cont_cache.get((sid, pid))
    if oc is not None:
        city_id = oc['city_id']
        conts_train = oc['conts']
        # For eval days, use eval features
        if eval_grp is not None:
            conts_eval = eval_s[CONT_COLS].values.astype(np.float32)
            conts_all = np.concatenate([conts_train, conts_eval])
        else:
            conts_all = conts_train
    else:
        city_id = 0
        conts_all = np.zeros((len(all_days), len(CONT_COLS)), dtype=np.float32)

    series_data[(sid, pid)] = {
        'days': all_days,
        'dows': all_dows,
        'completed_sales': all_completed,  # completed Y for training
        'original_sales': all_original,    # S_obs for evaluation
        'stock': all_stock,
        'city_id': city_id,
        'conts': conts_all,
    }

del comp_groups, eval_groups, orig_cont_cache
del csales_arr, osales_arr, stock_train_arr, eval_sales_arr, eval_stock_arr

print(f"  {len(series_data):,} serie in {time.time()-t1:.0f}s")


# =========================================================================
# 3. NAIVE BASELINES su dati puliti
# =========================================================================
print("\n" + "=" * 72)
print("  PARTE 1: NAIVE BASELINES (profili da completed_sales)")
print("=" * 72)

NAIVE_MODELS = ['clean_global_mean', 'clean_dow_mean', 'clean_naive_direct', 'clean_ma_direct']
SPLITS = ['val', 'test']

pooled = {
    model: {
        split: {sub: {'sae': 0., 'sao': 0., 'se': 0., 'so': 0., 'n': 0}
                for sub in ['overall', 'instock', 'stockout']}
        for split in SPLITS
    } for model in NAIVE_MODELS
}
per_series = {model: {split: [] for split in SPLITS} for model in NAIVE_MODELS}

print("\n  Computing naive predictions from completed_sales profiles...")
t2 = time.time()
n_groups = len(series_data)

for i, ((sid, pid), sd) in enumerate(series_data.items()):
    if (i + 1) % 10000 == 0:
        print(f"    ... {i+1:,}/{n_groups:,} serie ({time.time()-t2:.0f}s)")

    days = sd['days']
    dows = sd['dows']
    csales = sd['completed_sales']   # completed Y (for profiles)
    osales = sd['original_sales']    # original S_obs (for evaluation)
    stock = sd['stock']

    train_mask = days <= 83
    trainval_mask = days <= 90

    if not train_mask.any():
        continue

    # Global Mean: profile from completed_sales
    gm_profile_tv = csales[train_mask].mean(axis=0)
    gm_profile_test = csales[trainval_mask].mean(axis=0)

    # DoW Mean: profile from completed_sales
    dow_profs_tv = {}
    for d in range(7):
        dm = train_mask & (dows == d)
        dow_profs_tv[d] = csales[dm].mean(axis=0) if dm.any() else gm_profile_tv

    gm_tv_full = csales[trainval_mask].mean(axis=0)
    dow_profs_test = {}
    for d in range(7):
        dm = trainval_mask & (dows == d)
        dow_profs_test[d] = csales[dm].mean(axis=0) if dm.any() else gm_tv_full

    # Naive Direct: profile from completed_sales (last day)
    anchor_83_idx = np.where(days == 83)[0]
    anchor_90_idx = np.where(days == 90)[0]
    nd_prof_val = csales[anchor_83_idx[0]] if len(anchor_83_idx) > 0 else None
    nd_prof_test = csales[anchor_90_idx[0]] if len(anchor_90_idx) > 0 else None

    # Evaluate each model
    for split_name, d_min, d_max in [('val', 84, 90), ('test', 91, 97)]:
        mask = (days >= d_min) & (days <= d_max)
        if not mask.any():
            continue

        obs = osales[mask]      # evaluate vs ORIGINAL S_obs
        stk = stock[mask]
        split_dows = dows[mask]
        n_split = mask.sum()

        # Global Mean
        gm_prof = gm_profile_test if split_name == 'test' else gm_profile_tv
        gm_pred = np.tile(gm_prof, (n_split, 1))
        _accumulate_pooled(pooled['clean_global_mean'][split_name], gm_pred, obs, stk)
        m = compute_metrics(gm_pred, obs, stk)
        m['store_id'] = sid; m['product_id'] = pid
        per_series['clean_global_mean'][split_name].append(m)

        # DoW Mean
        dprofs = dow_profs_test if split_name == 'test' else dow_profs_tv
        dow_pred = np.array([dprofs[d] for d in split_dows])
        _accumulate_pooled(pooled['clean_dow_mean'][split_name], dow_pred, obs, stk)
        m = compute_metrics(dow_pred, obs, stk)
        m['store_id'] = sid; m['product_id'] = pid
        per_series['clean_dow_mean'][split_name].append(m)

        # Naive Direct
        nd_prof = nd_prof_test if split_name == 'test' else nd_prof_val
        if nd_prof is not None:
            nd_pred = np.tile(nd_prof, (n_split, 1))
            _accumulate_pooled(pooled['clean_naive_direct'][split_name], nd_pred, obs, stk)
            m = compute_metrics(nd_pred, obs, stk)
            m['store_id'] = sid; m['product_id'] = pid
            per_series['clean_naive_direct'][split_name].append(m)

        # MA Direct (K=14)
        if split_name == 'val':
            anc_day = 83
        else:
            anc_day = 90
        start_day = max(1, anc_day - K_STAR + 1)
        hist_m = (days >= start_day) & (days <= anc_day)
        if hist_m.any():
            ma_prof = csales[hist_m].mean(axis=0)
            ma_pred = np.tile(ma_prof, (n_split, 1))
            _accumulate_pooled(pooled['clean_ma_direct'][split_name], ma_pred, obs, stk)
            m = compute_metrics(ma_pred, obs, stk)
            m['store_id'] = sid; m['product_id'] = pid
            per_series['clean_ma_direct'][split_name].append(m)

print(f"\n  Naive loop: {time.time()-t2:.0f}s")

# Save and print naive results
print("\n  --- NAIVE RESULTS ---")
for split in SPLITS:
    print(f"\n  {split.upper()}:")
    print(f"    {'Modello':<25} {'WAPE_in_pool':>14} {'WAPE_in_med':>14} "
          f"{'WPE_in_pool':>14} {'WPE_in_med':>14}")
    print("    " + "-" * 80)

    for model in NAIVE_MODELS:
        if not per_series[model][split]:
            continue
        ps_df = pd.DataFrame(per_series[model][split])
        ps_df.to_parquet(os.path.join(RESULTS_DIR, f'{model}_{split}_per_series.parquet'),
                          index=False)

        acc_in = pooled[model][split]['instock']
        wape_in_p = acc_in['sae'] / acc_in['sao'] if acc_in['sao'] > 0 else np.nan
        wpe_in_p = acc_in['se'] / acc_in['so'] if acc_in['so'] > 0 else np.nan
        wape_in_m = ps_df['wape_instock'].median() if 'wape_instock' in ps_df else np.nan
        wpe_in_m = ps_df['wpe_instock'].median() if 'wpe_instock' in ps_df else np.nan

        label = model.replace('clean_', '').replace('_', ' ').title()
        print(f"    {label:<25} {wape_in_p:>14.4f} {wape_in_m:>14.4f} "
              f"{wpe_in_p:>14.4f} {wpe_in_m:>14.4f}")


# =========================================================================
# 4. LGB F su dati puliti
# =========================================================================
print("\n\n" + "=" * 72)
print("  PARTE 2: LGB VARIANT F (M5 lags da completed_sales)")
print("=" * 72)

# Build hourly dataset with lags from completed_sales
def build_hourly_lgb(series_data, d_min, d_max, anchor_mode='rolling',
                     anchor_day=None, with_lags=True, use_completed_target=True):
    """Build flat per-hour dataset for LGB, with lags from completed_sales."""
    all_rows = []
    for (sid, pid), sd in series_data.items():
        days = sd['days']
        dows = sd['dows']
        csales = sd['completed_sales']
        osales = sd['original_sales']
        stock = sd['stock']
        city = sd['city_id']
        conts = sd['conts']

        for idx in range(len(days)):
            d = days[idx]
            if d < d_min or d > d_max:
                continue

            anc = d - 1 if anchor_mode == 'rolling' else anchor_day

            # Target: completed_Y for training, original S_obs for eval
            target = csales[idx] if use_completed_target else osales[idx]

            # Base features (per hour)
            for h in range(24):
                row = {
                    'store_id': sid, 'product_id': pid, 'city_id': city,
                    'dow': dows[idx], 'hour': h,
                    'target': target[h],
                    'stock_status': stock[idx, h],
                    'original_target': osales[idx, h],
                }
                for j, col in enumerate(CONT_COLS):
                    row[col] = conts[idx, j]

                if with_lags:
                    hist_mask = days <= anc
                    K = int(hist_mask.sum())
                    avail = csales[hist_mask] if K > 0 else None
                    avail_dows = dows[hist_mask] if K > 0 else None

                    row['lag_1d'] = avail[-1, h] if K > 0 else np.nan
                    row['lag_7d'] = avail[-7, h] if K >= 7 else np.nan
                    row['lag_14d'] = avail[-14, h] if K >= 14 else np.nan
                    row['rmean_7d'] = avail[-min(7, K):, h].mean() if K > 0 else np.nan
                    row['rmean_14d'] = avail[-min(14, K):, h].mean() if K >= 14 else np.nan
                    row['rstd_7d'] = avail[-min(7, K):, h].std() if K >= 2 else np.nan

                    same_dow = avail_dows == dows[idx] if K > 0 else np.array([], dtype=bool)
                    if K > 0 and same_dow.any():
                        row['lag_dow'] = avail[same_dow][-1, h]
                        row['rmean_dow'] = avail[same_dow][:, h].mean()
                    else:
                        row['lag_dow'] = np.nan
                        row['rmean_dow'] = np.nan

                    row['daily_total_lag1'] = avail[-1].sum() if K > 0 else np.nan
                    row['daily_total_rmean7'] = avail[-7:].sum(axis=1).mean() if K >= 7 else np.nan

                    rm7 = row['rmean_7d']
                    l1d = row['lag_1d']
                    if not np.isnan(rm7) and not np.isnan(l1d) and rm7 > 0:
                        row['momentum_1d_7d'] = l1d / rm7
                    else:
                        row['momentum_1d_7d'] = np.nan

                all_rows.append(row)

    df = pd.DataFrame(all_rows)
    return df

# This row-by-row approach is too slow for 4M+ rows. Use vectorized approach instead.
# Reuse the Fase A pattern: expand arrays then compute lags per-row in NumPy.

def build_hourly_lgb_fast(series_data, d_min, d_max, anchor_mode='rolling',
                          anchor_day=None, with_lags=True, use_completed_target=True):
    """Build flat per-hour dataset for LGB — vectorized."""
    # Collect day-level arrays first
    sids, pids, cids, dows_l, days_l = [], [], [], [], []
    conts_l, csales_l, osales_l, stock_l = [], [], [], []

    for (sid, pid), sd in series_data.items():
        d = sd['days']
        mask = (d >= d_min) & (d <= d_max)
        n = mask.sum()
        if n == 0:
            continue

        sids.extend([sid] * n)
        pids.extend([pid] * n)
        cids.extend([sd['city_id']] * n)
        dows_l.append(sd['dows'][mask])
        days_l.append(sd['days'][mask])
        conts_l.append(sd['conts'][mask])
        csales_l.append(sd['completed_sales'][mask])
        osales_l.append(sd['original_sales'][mask])
        stock_l.append(sd['stock'][mask])

    N = len(sids)
    sids = np.array(sids, dtype=np.int64)
    pids = np.array(pids, dtype=np.int64)
    cids = np.array(cids, dtype=np.int64)
    dows_arr = np.concatenate(dows_l)
    days_arr = np.concatenate(days_l)
    conts_arr = np.concatenate(conts_l, axis=0)
    csales_arr = np.concatenate(csales_l, axis=0)
    osales_arr = np.concatenate(osales_l, axis=0)
    stock_arr = np.concatenate(stock_l, axis=0)

    N_hourly = N * 24

    # Expand to hourly
    sids_h = np.repeat(sids, 24)
    pids_h = np.repeat(pids, 24)
    cids_h = np.repeat(cids, 24)
    dows_h = np.repeat(dows_arr, 24)
    hour_h = np.tile(np.arange(24, dtype=np.int32), N)
    conts_h = np.repeat(conts_arr, 24, axis=0)

    if use_completed_target:
        y = csales_arr.ravel()
    else:
        y = osales_arr.ravel()

    stock_flat = stock_arr.ravel()
    original_y = osales_arr.ravel()

    # Features
    X = pd.DataFrame({
        'store_id': sids_h, 'product_id': pids_h, 'city_id': cids_h,
        'dow': dows_h, 'hour': hour_h,
    })
    for j, col in enumerate(CONT_COLS):
        X[col] = conts_h[:, j]

    for col in ['store_id', 'product_id', 'city_id', 'dow', 'hour']:
        X[col] = X[col].astype('category')

    if with_lags:
        lag_names = ['lag_1d', 'lag_7d', 'lag_14d', 'rmean_7d', 'rmean_14d',
                     'rstd_7d', 'lag_dow', 'rmean_dow', 'daily_total_lag1',
                     'daily_total_rmean7', 'momentum_1d_7d']
        lag_arrays = {name: np.full(N_hourly, np.nan, dtype=np.float32)
                      for name in lag_names}

        for i in range(N):
            sid, pid = sids[i], pids[i]
            d = days_arr[i]
            dow_i = dows_arr[i]
            hs, he = i * 24, (i + 1) * 24

            anc = d - 1 if anchor_mode == 'rolling' else anchor_day

            sc = series_data.get((sid, pid))
            if sc is None:
                continue

            s_days = sc['days']
            s_csales = sc['completed_sales']  # Use completed_sales for lags!
            s_dows = sc['dows']

            hist_mask = s_days <= anc
            K = int(hist_mask.sum())
            if K == 0:
                continue

            avail = s_csales[hist_mask]
            avail_dows = s_dows[hist_mask]

            lag_arrays['lag_1d'][hs:he] = avail[-1]
            if K >= 7:
                lag_arrays['lag_7d'][hs:he] = avail[-7]
            if K >= 14:
                lag_arrays['lag_14d'][hs:he] = avail[-14]

            n7 = min(7, K)
            lag_arrays['rmean_7d'][hs:he] = avail[-n7:].mean(axis=0)
            if K >= 14:
                lag_arrays['rmean_14d'][hs:he] = avail[-min(14, K):].mean(axis=0)
            if K >= 2:
                lag_arrays['rstd_7d'][hs:he] = avail[-n7:].std(axis=0)

            same_dow = avail_dows == dow_i
            if same_dow.any():
                dow_sales = avail[same_dow]
                lag_arrays['lag_dow'][hs:he] = dow_sales[-1]
                lag_arrays['rmean_dow'][hs:he] = dow_sales.mean(axis=0)

            lag_arrays['daily_total_lag1'][hs:he] = avail[-1].sum()
            if K >= 7:
                lag_arrays['daily_total_rmean7'][hs:he] = avail[-7:].sum(axis=1).mean()

        # Momentum
        rm7 = lag_arrays['rmean_7d']
        l1d = lag_arrays['lag_1d']
        valid_mom = (~np.isnan(rm7)) & (~np.isnan(l1d)) & (rm7 > 0)
        lag_arrays['momentum_1d_7d'][valid_mom] = l1d[valid_mom] / rm7[valid_mom]

        for name in lag_names:
            X[name] = lag_arrays[name]

    return X, y, stock_flat, original_y, sids_h, pids_h


def evaluate_lgb_perseries(preds, y_orig, stock, sids, pids):
    """Evaluate against ORIGINAL S_obs."""
    err = preds - y_orig
    results_pooled = {}
    for sub, smask in [('overall', np.ones(len(preds), dtype=bool)),
                       ('instock', stock == 0),
                       ('stockout', stock == 1)]:
        ef = err[smask]
        of = y_orig[smask]
        sao = np.abs(of).sum()
        so = of.sum()
        results_pooled[f'wape_{sub}'] = np.abs(ef).sum() / sao if sao > 0 else np.nan
        results_pooled[f'wpe_{sub}'] = ef.sum() / so if so > 0 else np.nan
        results_pooled[f'n_{sub}'] = int(smask.sum())

    df_tmp = pd.DataFrame({'sid': sids, 'pid': pids, 'pred': preds,
                           'obs': y_orig, 'stock': stock})
    records = []
    for (sid, pid), grp in df_tmp.groupby(['sid', 'pid'], sort=False):
        m = compute_metrics(grp['pred'].values, grp['obs'].values, grp['stock'].values)
        m['store_id'] = sid
        m['product_id'] = pid
        records.append(m)

    return results_pooled, pd.DataFrame(records)


# 4a. Build LGB datasets
print("\n  4a. Building LGB datasets (variant F, lags from completed_sales)...")
t3 = time.time()

X_train_lgb, y_train_lgb, stock_train_lgb, orig_train_lgb, sids_tr, pids_tr = \
    build_hourly_lgb_fast(series_data, 2, 83, 'rolling', with_lags=True,
                           use_completed_target=True)
X_val_lgb, y_val_lgb, stock_val_lgb, orig_val_lgb, sids_vl, pids_vl = \
    build_hourly_lgb_fast(series_data, 84, 90, 'fixed', anchor_day=83,
                           with_lags=True, use_completed_target=True)

print(f"    Train: {len(X_train_lgb):,}, Val: {len(X_val_lgb):,}")
print(f"    Features: {X_train_lgb.shape[1]}")
nan_pct = X_train_lgb['lag_1d'].isna().mean() * 100
print(f"    lag_1d NaN: {nan_pct:.1f}%")
print(f"    Build time: {time.time()-t3:.0f}s")

# 4b. Train LGB with early stopping
print("\n  4b. Training LGB F (clean lags)...")
t4 = time.time()

lgb_train_ds = lgb.Dataset(X_train_lgb, y_train_lgb, free_raw_data=True)
lgb_val_ds = lgb.Dataset(X_val_lgb, y_val_lgb, reference=lgb_train_ds, free_raw_data=True)

lgb_model = lgb.train(
    LGB_PARAMS, lgb_train_ds,
    num_boost_round=MAX_BOOST_ROUNDS,
    valid_sets=[lgb_val_ds], valid_names=['val'],
    callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(50)],
)

lgb_best_iter = lgb_model.best_iteration
lgb_best_mae = lgb_model.best_score['val']['l1']
print(f"    Best iteration: {lgb_best_iter}, Val MAE: {lgb_best_mae:.6f}")
print(f"    Training time: {time.time()-t4:.0f}s")

# 4c. Val evaluation (vs original S_obs)
print("\n  4c. Val evaluation (vs S_obs)...")
preds_val_lgb = np.clip(lgb_model.predict(X_val_lgb), 0, None)
pooled_val_lgb, ps_val_lgb = evaluate_lgb_perseries(
    preds_val_lgb, orig_val_lgb, stock_val_lgb, sids_vl, pids_vl)
ps_val_lgb.to_parquet(os.path.join(RESULTS_DIR, 'clean_lgb_f_val_per_series.parquet'),
                       index=False)
print(f"    WAPE_in pooled: {pooled_val_lgb['wape_instock']:.4f}")
print(f"    WAPE_in median: {ps_val_lgb['wape_instock'].median():.4f}")
print(f"    WPE_in pooled:  {pooled_val_lgb['wpe_instock']:.4f}")

del X_train_lgb, X_val_lgb, y_train_lgb, y_val_lgb

# 4d. Retrain on days 2-90
print(f"\n  4d. Retraining on days 2-90 ({lgb_best_iter} rounds)...")
t5 = time.time()

X_rt_lgb, y_rt_lgb, _, _, _, _ = build_hourly_lgb_fast(
    series_data, 2, 90, 'rolling', with_lags=True, use_completed_target=True)

lgb_rt_ds = lgb.Dataset(X_rt_lgb, y_rt_lgb, free_raw_data=True)
lgb_rt_model = lgb.train(LGB_PARAMS, lgb_rt_ds, num_boost_round=lgb_best_iter)
print(f"    Retrain time: {time.time()-t5:.0f}s")

lgb_rt_model.save_model(os.path.join(RESULTS_DIR, 'clean_lgb_f_retrained.txt'))
del X_rt_lgb, y_rt_lgb

# 4e. Test evaluation
print("\n  4e. Test evaluation (eval HF, gg 91-97)...")
X_test_lgb, _, stock_test_lgb, orig_test_lgb, sids_te, pids_te = \
    build_hourly_lgb_fast(series_data, 91, 97, 'fixed', anchor_day=90,
                           with_lags=True, use_completed_target=False)

preds_test_lgb = np.clip(lgb_rt_model.predict(X_test_lgb), 0, None)
pooled_test_lgb, ps_test_lgb = evaluate_lgb_perseries(
    preds_test_lgb, orig_test_lgb, stock_test_lgb, sids_te, pids_te)
ps_test_lgb.to_parquet(os.path.join(RESULTS_DIR, 'clean_lgb_f_test_per_series.parquet'),
                        index=False)
print(f"    WAPE_in pooled: {pooled_test_lgb['wape_instock']:.4f}")
print(f"    WAPE_in median: {ps_test_lgb['wape_instock'].median():.4f}")
print(f"    WPE_in pooled:  {pooled_test_lgb['wpe_instock']:.4f}")
print(f"    WPE_in median:  {ps_test_lgb['wpe_instock'].median():.4f}")

# Feature importance
fi = lgb_rt_model.feature_importance(importance_type='gain')
fi_names = lgb_rt_model.feature_name()
fi_sorted = sorted(zip(fi_names, fi), key=lambda x: -x[1])
print(f"\n    Feature importance (top 10):")
for name, imp in fi_sorted[:10]:
    print(f"      {name:<25s} {imp:>12,.0f}")

del X_test_lgb, lgb_model, lgb_rt_model


# =========================================================================
# 5. MLP F su dati puliti
# =========================================================================
print("\n\n" + "=" * 72)
print("  PARTE 3: MLP VARIANT F (M5 lags da completed_sales)")
print("=" * 72)


def _compute_lag_features_f(csales, days, dows, target_day_idx, anchor_day):
    """Compute M5-style lag features from completed_sales."""
    z = np.float32
    feats = [np.zeros(24, dtype=z) for _ in range(11)]
    masks = np.zeros(11, dtype=z)

    avail_mask = days <= anchor_day
    K = int(avail_mask.sum())

    if K > 0:
        avail_sales = csales[avail_mask]
        avail_dows = dows[avail_mask]
        target_dow = dows[target_day_idx]

        feats[0][:] = avail_sales[-1]; masks[0] = 1.0
        if K >= 7: feats[1][:] = avail_sales[-7]; masks[1] = 1.0
        if K >= 14: feats[2][:] = avail_sales[-14]; masks[2] = 1.0
        if K >= 7: feats[3][:] = avail_sales[-7:].mean(axis=0); masks[3] = 1.0
        if K >= 14: feats[4][:] = avail_sales[-14:].mean(axis=0); masks[4] = 1.0
        if K >= 2:
            w = min(7, K)
            feats[5][:] = avail_sales[-w:].std(axis=0)
            masks[5] = 1.0
        same_dow = avail_dows == target_dow
        if same_dow.any():
            feats[6][:] = avail_sales[same_dow][-1]; masks[6] = 1.0
            feats[7][:] = avail_sales[same_dow].mean(axis=0); masks[7] = 1.0
        daily_totals = avail_sales.sum(axis=1)
        feats[8][:] = daily_totals[-1]; masks[8] = 1.0
        if K >= 7: feats[9][:] = daily_totals[-7:].mean(); masks[9] = 1.0
        if masks[3] == 1.0:
            rm7 = feats[3]; valid_h = rm7 > 0
            if valid_h.any():
                feats[10][valid_h] = feats[0][valid_h] / rm7[valid_h]
                masks[10] = 1.0

    return np.concatenate(feats + [masks])


def build_mlp_arrays(series_data, d_min, d_max, anchor_mode='rolling',
                     anchor_day=None, use_completed_target=True,
                     cont_mean=None, cont_std=None, lag_mean=None, lag_std=None):
    """Build arrays for MLP, using completed_sales for lags."""
    cat_list, cont_list, lag_list = [], [], []
    target_list, stock_list, orig_target_list = [], [], []
    sid_list, pid_list = [], []

    for (sid, pid), sd in series_data.items():
        days = sd['days']
        dows = sd['dows']
        csales = sd['completed_sales']
        osales = sd['original_sales']
        stock = sd['stock']
        city = sd['city_id']
        conts = sd['conts']

        for idx in range(len(days)):
            d = days[idx]
            if d < d_min or d > d_max:
                continue

            a_day = d - 1 if anchor_mode == 'rolling' else anchor_day

            cat_list.append([sid, pid, city, dows[idx]])
            cont_list.append(conts[idx])
            lag_list.append(_compute_lag_features_f(csales, days, dows, idx, a_day))

            if use_completed_target:
                target_list.append(csales[idx])
            else:
                target_list.append(osales[idx])
            stock_list.append(stock[idx])
            orig_target_list.append(osales[idx])
            sid_list.append(sid)
            pid_list.append(pid)

    cat_arr = np.array(cat_list, dtype=np.int64)
    cont_arr = np.array(cont_list, dtype=np.float32)
    lag_arr = np.array(lag_list, dtype=np.float32)
    target_arr = np.array(target_list, dtype=np.float32)
    stock_arr = np.array(stock_list, dtype=np.float32)
    orig_arr = np.array(orig_target_list, dtype=np.float32)

    if cont_mean is None:
        cont_mean = cont_arr.mean(axis=0)
        cont_std = cont_arr.std(axis=0)
        cont_std[cont_std < 1e-8] = 1.0
    cont_arr = (cont_arr - cont_mean) / cont_std

    if lag_mean is None:
        lag_mean = lag_arr.mean(axis=0)
        lag_std = lag_arr.std(axis=0)
        lag_std[lag_std < 1e-8] = 1.0
    lag_arr = (lag_arr - lag_mean) / lag_std

    return {
        'cat': cat_arr, 'cont': cont_arr, 'lags': lag_arr,
        'targets': target_arr, 'stock': stock_arr,
        'original_targets': orig_arr,
        'store_ids': np.array(sid_list, dtype=np.int64),
        'product_ids': np.array(pid_list, dtype=np.int64),
        'cont_mean': cont_mean, 'cont_std': cont_std,
        'lag_mean': lag_mean, 'lag_std': lag_std,
    }


class RetailDataset(Dataset):
    def __init__(self, cat, cont, lags, targets):
        self.cat = torch.from_numpy(cat)
        self.cont = torch.from_numpy(cont)
        self.lags = torch.from_numpy(lags)
        self.targets = torch.from_numpy(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.cat[idx], self.cont[idx], self.lags[idx], self.targets[idx]


class RetailMLP(nn.Module):
    def __init__(self, n_cont, n_lags, emb_dims, cardinalities, hidden_sizes):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(cardinalities[name], emb_dims[name])
            for name in emb_dims
        })
        self.emb_names = ['store_id', 'product_id', 'city_id', 'dow']
        total_emb = sum(emb_dims.values())
        input_dim = total_emb + n_cont + n_lags
        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 24))
        layers.append(nn.Softplus())
        self.mlp = nn.Sequential(*layers)

    def forward(self, cat, cont, lags):
        emb_list = [self.embeddings[name](cat[:, i])
                    for i, name in enumerate(self.emb_names)]
        x = torch.cat(emb_list + [cont, lags], dim=1)
        return self.mlp(x)


def mlp_predict(model, data, device, chunk_size=10000):
    model.eval()
    cat_t = torch.from_numpy(data['cat'])
    cont_t = torch.from_numpy(data['cont'])
    lags_t = torch.from_numpy(data['lags'])
    all_preds = []
    with torch.no_grad():
        for s in range(0, len(cat_t), chunk_size):
            e = min(s + chunk_size, len(cat_t))
            p = model(cat_t[s:e].to(device), cont_t[s:e].to(device),
                      lags_t[s:e].to(device))
            all_preds.append(p.cpu().numpy())
    return np.concatenate(all_preds, axis=0)


def mlp_evaluate(model, data, device):
    preds = mlp_predict(model, data, device)
    # Evaluate vs ORIGINAL S_obs
    obs = data['original_targets']
    stock = data['stock']
    sids = data['store_ids']
    pids = data['product_ids']

    p_flat = preds.ravel()
    o_flat = obs.ravel()
    s_flat = stock.ravel()
    err = p_flat - o_flat

    pooled = {}
    for sub, smask in [('overall', np.ones(len(p_flat), dtype=bool)),
                       ('instock', s_flat == 0),
                       ('stockout', s_flat == 1)]:
        ef = err[smask]; of = o_flat[smask]
        sao = np.abs(of).sum(); so = of.sum()
        pooled[f'wape_{sub}'] = np.abs(ef).sum() / sao if sao > 0 else np.nan
        pooled[f'wpe_{sub}'] = ef.sum() / so if so != 0 else np.nan

    df_idx = pd.DataFrame({'sid': sids, 'pid': pids, 'row': np.arange(len(sids))})
    records = []
    for (sid, pid), grp in df_idx.groupby(['sid', 'pid']):
        idx = grp['row'].values
        m = compute_metrics(preds[idx], obs[idx], stock[idx])
        m['store_id'] = sid; m['product_id'] = pid
        records.append(m)

    return pooled, pd.DataFrame(records)


# 5a. Build MLP datasets
print("\n  5a. Building MLP datasets (variant F, clean lags)...")
t6 = time.time()

train_data = build_mlp_arrays(series_data, 2, 83, 'rolling', use_completed_target=True)
val_data = build_mlp_arrays(
    series_data, 84, 90, 'fixed', anchor_day=83, use_completed_target=False,
    cont_mean=train_data['cont_mean'], cont_std=train_data['cont_std'],
    lag_mean=train_data['lag_mean'], lag_std=train_data['lag_std'])

n_cont = train_data['cont'].shape[1]
n_lags = train_data['lags'].shape[1]
print(f"    Train: {len(train_data['targets']):,}, Val: {len(val_data['targets']):,}")
print(f"    Input dim: {sum(EMB_DIMS.values()) + n_cont + n_lags}")
print(f"    Build time: {time.time()-t6:.0f}s")

# 5b. Train MLP
print("\n  5b. Training MLP F (clean lags)...")
t7 = time.time()

torch.manual_seed(SEED)
model = RetailMLP(n_cont, n_lags, EMB_DIMS, CARDINALITIES, HIDDEN_SIZES)
n_params = sum(p.numel() for p in model.parameters())
print(f"    Model params: {n_params:,}")

train_ds = RetailDataset(train_data['cat'], train_data['cont'],
                          train_data['lags'], train_data['targets'])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=0, pin_memory=False)

model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
best_val_wape = float('inf')
best_epoch = 0
best_state = None
epochs_no_improve = 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    train_loss = 0.0
    n_batches = 0
    for cat, cont, lags, targets in train_loader:
        preds = model(cat.to(DEVICE), cont.to(DEVICE), lags.to(DEVICE))
        loss = nn.functional.mse_loss(preds, targets.to(DEVICE))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        n_batches += 1
    avg_loss = train_loss / n_batches

    # Val WAPE (vs completed_Y for early stopping)
    val_preds = mlp_predict(model, val_data, DEVICE)
    sae = np.abs(val_preds - val_data['targets']).sum()
    sao = np.abs(val_data['targets']).sum()
    val_wape = sae / sao if sao > 0 else float('inf')

    if epoch % 5 == 0 or epoch == 1:
        print(f"    Epoch {epoch:3d}: loss={avg_loss:.6f}, val_WAPE={val_wape:.6f}")

    if val_wape < best_val_wape:
        best_val_wape = val_wape
        best_epoch = epoch
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= PATIENCE:
        print(f"    Early stopping at epoch {epoch} (best={best_epoch}, WAPE={best_val_wape:.6f})")
        break

if best_state is not None:
    model.load_state_dict(best_state)
model.to(DEVICE)
print(f"    Best epoch: {best_epoch}, val WAPE: {best_val_wape:.6f}")
print(f"    Training time: {time.time()-t7:.0f}s")

torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'clean_mlp_f.pt'))

# 5c. Val evaluation (vs original S_obs)
print("\n  5c. Val evaluation (vs S_obs)...")
pooled_val_mlp, ps_val_mlp = mlp_evaluate(model, val_data, DEVICE)
ps_val_mlp.to_parquet(os.path.join(RESULTS_DIR, 'clean_mlp_f_val_per_series.parquet'),
                       index=False)
print(f"    WAPE_in pooled: {pooled_val_mlp['wape_instock']:.4f}")
print(f"    WAPE_in median: {ps_val_mlp['wape_instock'].median():.4f}")
print(f"    WPE_in pooled:  {pooled_val_mlp['wpe_instock']:.4f}")

# 5d. Retrain on days 2-90
print(f"\n  5d. Retraining on days 2-90 ({best_epoch} epochs)...")
t8 = time.time()

retrain_data = build_mlp_arrays(series_data, 2, 90, 'rolling', use_completed_target=True)

torch.manual_seed(SEED)
model_rt = RetailMLP(n_cont, n_lags, EMB_DIMS, CARDINALITIES, HIDDEN_SIZES)
rt_ds = RetailDataset(retrain_data['cat'], retrain_data['cont'],
                       retrain_data['lags'], retrain_data['targets'])
rt_loader = DataLoader(rt_ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, pin_memory=False)

model_rt.to(DEVICE)
rt_optimizer = torch.optim.Adam(model_rt.parameters(), lr=LR)
for epoch in range(1, best_epoch + 1):
    model_rt.train()
    train_loss = 0.0
    n_batches = 0
    for cat, cont, lags, targets in rt_loader:
        preds = model_rt(cat.to(DEVICE), cont.to(DEVICE), lags.to(DEVICE))
        loss = nn.functional.mse_loss(preds, targets.to(DEVICE))
        rt_optimizer.zero_grad()
        loss.backward()
        rt_optimizer.step()
        train_loss += loss.item()
        n_batches += 1
    if epoch % 5 == 0 or epoch == 1:
        print(f"    Epoch {epoch:3d}/{best_epoch}: loss={train_loss/n_batches:.6f}")

print(f"    Retrain time: {time.time()-t8:.0f}s")
torch.save(model_rt.state_dict(), os.path.join(RESULTS_DIR, 'clean_mlp_f_retrained.pt'))

# 5e. Test evaluation
print("\n  5e. Test evaluation (eval HF)...")
test_data = build_mlp_arrays(
    series_data, 91, 97, 'fixed', anchor_day=90, use_completed_target=False,
    cont_mean=retrain_data['cont_mean'], cont_std=retrain_data['cont_std'],
    lag_mean=retrain_data['lag_mean'], lag_std=retrain_data['lag_std'])

pooled_test_mlp, ps_test_mlp = mlp_evaluate(model_rt, test_data, DEVICE)
ps_test_mlp.to_parquet(os.path.join(RESULTS_DIR, 'clean_mlp_f_test_per_series.parquet'),
                        index=False)
print(f"    WAPE_in pooled: {pooled_test_mlp['wape_instock']:.4f}")
print(f"    WAPE_in median: {ps_test_mlp['wape_instock'].median():.4f}")
print(f"    WPE_in pooled:  {pooled_test_mlp['wpe_instock']:.4f}")
print(f"    WPE_in median:  {ps_test_mlp['wpe_instock'].median():.4f}")


# =========================================================================
# 6. CONFRONTO FINALE
# =========================================================================
print("\n\n" + "=" * 72)
print("  6. CONFRONTO FINALE (TEST) — Fase A (sporco) vs Fase B (pulito)")
print("=" * 72)

print(f"\n  {'Modello':<25} {'WAPE_in_pool':>14} {'WAPE_in_med':>14} "
      f"{'WPE_in_pool':>14} {'WPE_in_med':>14} {'WAPE_all_med':>14}")
print("  " + "-" * 100)

# Load Fase A results for comparison
fase_a = {
    'Global Mean (A)': 'global_mean',
    'DoW Mean (A)': 'dow_mean',
    'Naive Direct (A)': 'naive_direct',
    'MA Direct (A)': 'ma_direct',
    'LGB F (A)': 'lgb_f',
    'MLP F (A)': 'mlp_f',
}
for label, prefix in fase_a.items():
    path = os.path.join(RESULTS_DIR, f'{prefix}_test_per_series.parquet')
    if os.path.exists(path):
        ps = pd.read_parquet(path)
        wape_in_m = ps['wape_instock'].median() if 'wape_instock' in ps else np.nan
        wpe_in_m = ps['wpe_instock'].median() if 'wpe_instock' in ps else np.nan
        wape_all_m = ps['wape_overall'].median() if 'wape_overall' in ps else np.nan
        print(f"  {label:<25} {'—':>14} {wape_in_m:>14.4f} "
              f"{'—':>14} {wpe_in_m:>14.4f} {wape_all_m:>14.4f}")

# Fase B clean naive
clean_naive = {
    'Global Mean (B)': 'clean_global_mean',
    'DoW Mean (B)': 'clean_dow_mean',
    'Naive Direct (B)': 'clean_naive_direct',
    'MA Direct (B)': 'clean_ma_direct',
}
for label, prefix in clean_naive.items():
    path = os.path.join(RESULTS_DIR, f'{prefix}_test_per_series.parquet')
    if os.path.exists(path):
        ps = pd.read_parquet(path)
        wape_in_m = ps['wape_instock'].median()
        wpe_in_m = ps['wpe_instock'].median()
        wape_all_m = ps['wape_overall'].median()
        print(f"  {label:<25} {'—':>14} {wape_in_m:>14.4f} "
              f"{'—':>14} {wpe_in_m:>14.4f} {wape_all_m:>14.4f}")

# Clean ML
print(f"  {'LGB F (B)':<25} {pooled_test_lgb['wape_instock']:>14.4f} "
      f"{ps_test_lgb['wape_instock'].median():>14.4f} "
      f"{pooled_test_lgb['wpe_instock']:>14.4f} "
      f"{ps_test_lgb['wpe_instock'].median():>14.4f} "
      f"{ps_test_lgb['wape_overall'].median():>14.4f}")

print(f"  {'MLP F (B)':<25} {pooled_test_mlp['wape_instock']:>14.4f} "
      f"{ps_test_mlp['wape_instock'].median():>14.4f} "
      f"{pooled_test_mlp['wpe_instock']:>14.4f} "
      f"{ps_test_mlp['wpe_instock'].median():>14.4f} "
      f"{ps_test_mlp['wape_overall'].median():>14.4f}")

print(f"\n  Tempo totale: {time.time()-t0:.0f}s")
print("=" * 72)
