"""
11_twostage_mlp.py — Two-Stage: Imputation + MLP Forecasting
=============================================================
PINN-Retail: Physics-Informed Neural Networks per Demand Forecasting
di Prodotti Deperibili.

Passo 4b: Two-stage con MLP al posto di LGB nello Stage 2.

Stage 1 — LGB Imputation (riuso approccio da notebook 10):
  Stima D̂ durante ore di stockout con LGB trainato su in-stock hours.

Stage 2 — MLP Forecasting:
  MLP con M5-style lag features calcolati dal dataset completato
  (completed_sales), target = completed Y.
  Valutazione finale contro S_obs originale.

Eseguire con: freshnet/bin/python notebooks/11_twostage_mlp.py
"""

import sys
import os
import gc
import time
import functools
import numpy as np
import pandas as pd

# Force unbuffered output
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import lightgbm as lgb

from src.evaluation.metrics import compute_metrics, format_metrics_table

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

DEVICE = 'mps' if torch.backends.mps.is_available() else (
    'cuda' if torch.cuda.is_available() else 'cpu')

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# MLP Hyperparameters (same as notebook 08)
BATCH_SIZE = 4096
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 10
HIDDEN_SIZES = [128, 64]

EMB_DIMS = {'store_id': 32, 'product_id': 32, 'city_id': 8, 'dow': 4}
CARDINALITIES = {'store_id': 898, 'product_id': 865, 'city_id': 18, 'dow': 7}

CONT_FEATURES = ['discount', 'avg_temperature', 'avg_humidity',
                  'precpt', 'avg_wind_level', 'holiday_flag', 'activity_flag']
CAT_FEATURES_LGB = ['store_id', 'product_id', 'city_id', 'dow', 'hour']

# LGB Stage 1 params (same as notebook 10)
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
    'seed': SEED,
}
MAX_BOOST_ROUNDS_IMP = 500
EARLY_STOPPING_IMP = 30

# ===========================================================================
print('=' * 72)
print('  TWO-STAGE: IMPUTATION + MLP FORECASTING')
print('=' * 72)

# ---------------------------------------------------------------------------
# 1. Caricamento dati
# ---------------------------------------------------------------------------
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
print(f'  Train: {len(df_train):,} righe, giorni 1-90')
print(f'  Eval:  {len(df_eval):,} righe, giorni 91-97')
print(f'  Full:  {len(df_full):,} righe, {len(all_dates)} giorni, {n_series:,} serie')
print(f'  Device: {DEVICE}')

del df_train, df_eval

# Pre-parse hourly arrays
sales_all = np.array(df_full['hours_sale'].tolist(), dtype=np.float32)
stock_all = np.array(df_full['hours_stock_status'].tolist(), dtype=np.float32)

# ---------------------------------------------------------------------------
# 2. Build series_cache
# ---------------------------------------------------------------------------
print('\n2. Costruzione series_cache...')
t0_total = time.time()

series_cache = {}
for (sid, pid), grp in df_full.groupby(['store_id', 'product_id'], sort=False):
    grp_s = grp.sort_values('day_num')
    idx = grp_s.index.values
    series_cache[(sid, pid)] = {
        'days': grp_s['day_num'].values,
        'dows': grp_s['dow'].values,
        'sales': sales_all[idx],
        'stock': stock_all[idx],
        'city_id': grp_s['city_id'].values[0],
        'conts': grp_s[CONT_FEATURES].values.astype(np.float32),
    }

print(f'  {len(series_cache):,} serie')


# ===========================================================================
# STAGE 1: LGB IMPUTATION
# ===========================================================================
print('\n' + '=' * 72)
print('  STAGE 1: LGB IMPUTATION')
print('=' * 72)


def build_hourly_dataset_instock(df, sales_arr, stock_arr, split):
    """Build hourly dataset from IN-STOCK hours only (no lags)."""
    if split == 'train':
        d_min, d_max = 2, 83
    elif split == 'val':
        d_min, d_max = 84, 90
    elif split == 'test':
        d_min, d_max = 91, 97

    mask_split = (df['day_num'] >= d_min) & (df['day_num'] <= d_max)
    df_split = df[mask_split]
    idx_split = np.where(mask_split.values)[0]
    n_days = len(df_split)

    store_ids_day = df_split['store_id'].values
    product_ids_day = df_split['product_id'].values
    city_ids_day = df_split['city_id'].values
    dows_day = df_split['dow'].values
    conts_day = df_split[CONT_FEATURES].values.astype(np.float32)
    sales_day = sales_arr[idx_split]
    stock_day = stock_arr[idx_split]

    hours = np.tile(np.arange(24, dtype=np.int32), n_days)
    store_ids_h = np.repeat(store_ids_day, 24)
    product_ids_h = np.repeat(product_ids_day, 24)
    city_ids_h = np.repeat(city_ids_day, 24)
    dows_h = np.repeat(dows_day, 24)
    conts_h = np.repeat(conts_day, 24, axis=0)
    y = sales_day.ravel().astype(np.float32)
    stock_flat = stock_day.ravel().astype(np.float32)

    instock_mask = stock_flat == 0

    feat_dict = {
        'store_id': store_ids_h[instock_mask],
        'product_id': product_ids_h[instock_mask],
        'city_id': city_ids_h[instock_mask],
        'dow': dows_h[instock_mask],
        'hour': hours[instock_mask],
    }
    for j, c in enumerate(CONT_FEATURES):
        feat_dict[c] = conts_h[instock_mask, j]

    X = pd.DataFrame(feat_dict)
    for c in CAT_FEATURES_LGB:
        X[c] = X[c].astype('category')

    return X, y[instock_mask], int(instock_mask.sum())


def build_hourly_dataset_all_base(df, sales_arr, stock_arr, split):
    """Build hourly dataset for ALL hours, base features only."""
    if split == 'train':
        d_min, d_max = 2, 83
    elif split == 'val':
        d_min, d_max = 84, 90
    elif split == 'test':
        d_min, d_max = 91, 97

    mask_split = (df['day_num'] >= d_min) & (df['day_num'] <= d_max)
    df_split = df[mask_split]
    idx_split = np.where(mask_split.values)[0]
    n_days = len(df_split)

    store_ids_day = df_split['store_id'].values
    product_ids_day = df_split['product_id'].values
    city_ids_day = df_split['city_id'].values
    dows_day = df_split['dow'].values
    conts_day = df_split[CONT_FEATURES].values.astype(np.float32)
    sales_day = sales_arr[idx_split]
    stock_day = stock_arr[idx_split]

    hours = np.tile(np.arange(24, dtype=np.int32), n_days)
    store_ids_h = np.repeat(store_ids_day, 24)
    product_ids_h = np.repeat(product_ids_day, 24)
    city_ids_h = np.repeat(city_ids_day, 24)
    dows_h = np.repeat(dows_day, 24)
    conts_h = np.repeat(conts_day, 24, axis=0)
    y = sales_day.ravel().astype(np.float32)
    stock_flat = stock_day.ravel().astype(np.float32)

    feat_dict = {
        'store_id': store_ids_h,
        'product_id': product_ids_h,
        'city_id': city_ids_h,
        'dow': dows_h,
        'hour': hours,
    }
    for j, c in enumerate(CONT_FEATURES):
        feat_dict[c] = conts_h[:, j]

    X = pd.DataFrame(feat_dict)
    for c in CAT_FEATURES_LGB:
        X[c] = X[c].astype('category')

    return X, y, stock_flat, store_ids_h, product_ids_h


# 3. Train LGB imputer
print('\n3. Training LGB imputer...')
t0 = time.time()

X_imp_train, y_imp_train, n_imp_train = \
    build_hourly_dataset_instock(df_full, sales_all, stock_all, 'train')
print(f'  In-stock train: {n_imp_train:,} righe')

X_imp_val, y_imp_val, n_imp_val = \
    build_hourly_dataset_instock(df_full, sales_all, stock_all, 'val')
print(f'  In-stock val:   {n_imp_val:,} righe')

lgb_imp_train = lgb.Dataset(X_imp_train, y_imp_train, free_raw_data=True)
lgb_imp_val = lgb.Dataset(X_imp_val, y_imp_val,
                           reference=lgb_imp_train, free_raw_data=True)

imp_model = lgb.train(
    LGB_PARAMS, lgb_imp_train,
    num_boost_round=MAX_BOOST_ROUNDS_IMP,
    valid_sets=[lgb_imp_val],
    valid_names=['val'],
    callbacks=[
        lgb.early_stopping(EARLY_STOPPING_IMP),
        lgb.log_evaluation(50),
    ],
)

print(f'  Best iter: {imp_model.best_iteration}, '
      f'MAE: {imp_model.best_score["val"]["l1"]:.6f}')

del X_imp_train, y_imp_train, lgb_imp_train
del X_imp_val, y_imp_val, lgb_imp_val
gc.collect()

# 4. Build completed_sales
print('\n4. Building completed_sales...')

completed_sales = {}
for split_name in ['train', 'val', 'test']:
    print(f'  Predicting {split_name}...')
    X_all, y_all, stock_flat, sids, pids = \
        build_hourly_dataset_all_base(df_full, sales_all, stock_all, split_name)

    preds = np.clip(imp_model.predict(X_all), 0, None)

    if split_name == 'train':
        d_min, d_max = 2, 83
    elif split_name == 'val':
        d_min, d_max = 84, 90
    else:
        d_min, d_max = 91, 97

    mask_split = (df_full['day_num'] >= d_min) & (df_full['day_num'] <= d_max)
    df_split = df_full[mask_split]
    sids_day = df_split['store_id'].values
    pids_day = df_split['product_id'].values
    day_nums = df_split['day_num'].values
    n_days_split = len(df_split)
    preds_2d = preds.reshape(n_days_split, 24)
    stock_2d = stock_flat.reshape(n_days_split, 24)

    for di in range(n_days_split):
        key = (sids_day[di], pids_day[di])
        sc = series_cache[key]
        day_idx = np.searchsorted(sc['days'], day_nums[di])
        if key not in completed_sales:
            completed_sales[key] = sc['sales'].copy()
        so_h = stock_2d[di] == 1
        if so_h.any():
            completed_sales[key][day_idx, so_h] = preds_2d[di, so_h]

    del X_all, y_all, preds, stock_flat, sids, pids
    gc.collect()

del imp_model
gc.collect()

elapsed_s1 = time.time() - t0
print(f'  Stage 1 completato in {elapsed_s1:.1f}s')

# Diagnostica imputation
n_imputed = 0
sum_imputed = 0.0
n_instock = 0
sum_instock = 0.0
for (sid, pid), sc in series_cache.items():
    so_mask = sc['stock'] == 1
    in_mask = sc['stock'] == 0
    n_imputed += so_mask.sum()
    sum_imputed += completed_sales[(sid, pid)][so_mask].sum()
    n_instock += in_mask.sum()
    sum_instock += sc['sales'][in_mask].sum()

mean_imputed = sum_imputed / n_imputed if n_imputed > 0 else 0
mean_instock = sum_instock / n_instock if n_instock > 0 else 0
print(f'  Ore stockout imputate: {n_imputed:,}')
print(f'  Media D̂ imputata:      {mean_imputed:.6f}')
print(f'  Media S_obs in-stock:   {mean_instock:.6f}')
if mean_instock > 0:
    print(f'  Ratio D̂/S_obs_in:      {mean_imputed / mean_instock:.4f}')


# ===========================================================================
# STAGE 2: MLP FORECASTING ON COMPLETED DATA
# ===========================================================================
print('\n' + '=' * 72)
print('  STAGE 2: MLP FORECASTING ON COMPLETED DATA')
print('=' * 72)

# 5. Build completed_series_data
print('\n5. Building completed_series_data...')

completed_series_data = {}
for key, sd in series_cache.items():
    completed_series_data[key] = {
        'days': sd['days'],
        'dows': sd['dows'],
        'sales': completed_sales[key],
        'stock': sd['stock'],
        'city_id': sd['city_id'],
        'conts': sd['conts'],
    }

print(f'  {len(completed_series_data):,} serie con vendite completate')


# ---------------------------------------------------------------------------
# MLP functions — vectorized lag computation
# ---------------------------------------------------------------------------
# Layout dei 275 valori output per riga:
#   [0:24]   lag_1d      [24:48]  lag_7d     [48:72]  lag_14d
#   [72:96]  rmean_7d    [96:120] rmean_14d  [120:144] rstd_7d
#   [144:168] lag_dow    [168:192] rmean_dow
#   [192:216] dt_lag1    [216:240] dt_rmean7 [240:264] momentum
#   [264:275] masks (11 binary flags)

def _compute_series_lags_train(sales, dows, target_indices):
    """Vectorized M5-style lag features for train split (rolling anchor d-1).

    For target at series index ti, available history = sales[0:ti], K = ti.
    Uses cumulative sums for rolling stats → no inner loop over days.

    Args:
        sales: (N, 24) full series sales
        dows: (N,) day of week
        target_indices: (M,) indices of target days in this series

    Returns:
        (M, 275) float32 array
    """
    M = len(target_indices)
    z = np.float32
    out = np.zeros((M, 275), dtype=z)
    if M == 0:
        return out

    ti = target_indices
    N = len(sales)

    # Pre-compute cumulative sums (float64 for numerical stability)
    daily_totals = sales.sum(axis=1)  # (N,)
    cs = np.zeros((N + 1, 24), dtype=np.float64)
    cs[1:] = np.cumsum(sales.astype(np.float64), axis=0)
    cs_sq = np.zeros((N + 1, 24), dtype=np.float64)
    cs_sq[1:] = np.cumsum((sales.astype(np.float64)) ** 2, axis=0)
    cs_dt = np.zeros(N + 1, dtype=np.float64)
    cs_dt[1:] = np.cumsum(daily_totals.astype(np.float64))

    # --- lag_1d: sales[ti-1] (K >= 1) ---
    v1 = ti >= 1
    if v1.any():
        out[v1, 0:24] = sales[ti[v1] - 1]
        out[v1, 264] = 1.0

    # --- lag_7d: sales[ti-7] ---
    v7 = ti >= 7
    if v7.any():
        out[v7, 24:48] = sales[ti[v7] - 7]
        out[v7, 265] = 1.0

    # --- lag_14d: sales[ti-14] ---
    v14 = ti >= 14
    if v14.any():
        out[v14, 48:72] = sales[ti[v14] - 14]
        out[v14, 266] = 1.0

    # --- rmean_7d: mean(sales[ti-7:ti]) ---
    if v7.any():
        t7 = ti[v7]
        out[v7, 72:96] = ((cs[t7] - cs[t7 - 7]) / 7.0).astype(z)
        out[v7, 267] = 1.0

    # --- rmean_14d: mean(sales[ti-14:ti]) ---
    if v14.any():
        t14 = ti[v14]
        out[v14, 96:120] = ((cs[t14] - cs[t14 - 14]) / 14.0).astype(z)
        out[v14, 268] = 1.0

    # --- rstd_7d: std via cumsum for ti >= 7 ---
    if v7.any():
        t7 = ti[v7]
        mean_7 = (cs[t7] - cs[t7 - 7]) / 7.0
        mean_sq_7 = (cs_sq[t7] - cs_sq[t7 - 7]) / 7.0
        var_7 = np.maximum(mean_sq_7 - mean_7 ** 2, 0)
        out[v7, 120:144] = np.sqrt(var_7).astype(z)
        out[v7, 269] = 1.0

    # rstd_7d for 2 <= ti < 7 (few rows, use direct std)
    v2_not7 = (ti >= 2) & (ti < 7)
    if v2_not7.any():
        for row in np.where(v2_not7)[0]:
            out[row, 120:144] = sales[:ti[row]].std(axis=0)
            out[row, 269] = 1.0

    # --- daily_total_lag1: broadcast to 24h ---
    if v1.any():
        out[v1, 192:216] = daily_totals[ti[v1] - 1, np.newaxis]
        out[v1, 272] = 1.0

    # --- daily_total_rmean7 ---
    if v7.any():
        t7 = ti[v7]
        dt_rm7 = ((cs_dt[t7] - cs_dt[t7 - 7]) / 7.0).astype(z)
        out[v7, 216:240] = dt_rm7[:, np.newaxis]
        out[v7, 273] = 1.0

    # --- DoW-specific: vectorized via searchsorted ---
    target_dows = dows[ti]
    for d in range(7):
        d_mask_t = target_dows == d
        if not d_mask_t.any():
            continue
        dow_d_idx = np.where(dows == d)[0]
        if len(dow_d_idx) == 0:
            continue
        dow_d_sales = sales[dow_d_idx]  # (K_d, 24)
        dow_d_cs = np.zeros((len(dow_d_idx) + 1, 24), dtype=np.float64)
        dow_d_cs[1:] = np.cumsum(dow_d_sales.astype(np.float64), axis=0)

        ti_d = ti[d_mask_t]
        count = np.searchsorted(dow_d_idx, ti_d, side='left')  # DoW days before ti
        has_prev = count > 0
        if has_prev.any():
            c = count[has_prev]
            rows = np.where(d_mask_t)[0][has_prev]
            out[rows, 144:168] = dow_d_sales[c - 1]
            out[rows, 168:192] = (dow_d_cs[c] / c[:, np.newaxis]).astype(z)
            out[rows, 270] = 1.0
            out[rows, 271] = 1.0

    # --- momentum: lag_1d / rmean_7d ---
    if v7.any():
        rm7 = out[v7, 72:96]
        l1 = out[v7, 0:24]
        pos = rm7 > 0
        mom = np.zeros_like(rm7)
        mom[pos] = l1[pos] / rm7[pos]
        out[v7, 240:264] = mom
        has_any = pos.any(axis=1)
        rows_mom = np.where(v7)[0][has_any]
        out[rows_mom, 274] = 1.0

    return out


def _compute_series_lags_fixed(sales, dows, target_indices, anchor_day, days):
    """Vectorized lag features for val/test with fixed anchor.

    All target days share the same available history, so non-DoW features
    are computed once and broadcast.
    """
    M = len(target_indices)
    z = np.float32
    out = np.zeros((M, 275), dtype=z)
    if M == 0:
        return out

    avail_mask = days <= anchor_day
    K = int(avail_mask.sum())
    if K == 0:
        return out

    avail = sales[avail_mask]       # (K, 24)
    avail_dows = dows[avail_mask]

    # --- Features constant across all M targets ---
    out[:, 0:24] = avail[-1]       # lag_1d
    out[:, 264] = 1.0

    if K >= 7:
        out[:, 24:48] = avail[-7]
        out[:, 265] = 1.0
        out[:, 72:96] = avail[-7:].mean(axis=0)
        out[:, 267] = 1.0

    if K >= 14:
        out[:, 48:72] = avail[-14]
        out[:, 266] = 1.0
        out[:, 96:120] = avail[-14:].mean(axis=0)
        out[:, 268] = 1.0

    if K >= 2:
        w = min(7, K)
        out[:, 120:144] = avail[-w:].std(axis=0)
        out[:, 269] = 1.0

    dt = avail.sum(axis=1)
    out[:, 192:216] = dt[-1]       # dt_lag1 broadcast
    out[:, 272] = 1.0

    if K >= 7:
        out[:, 216:240] = dt[-7:].mean()
        out[:, 273] = 1.0

    # --- DoW-specific (varies by target_dow) ---
    target_dows = dows[target_indices]
    for d in range(7):
        d_mask = target_dows == d
        if not d_mask.any():
            continue
        same_dow = avail_dows == d
        if same_dow.any():
            dow_sales = avail[same_dow]
            rows = np.where(d_mask)[0]
            out[rows, 144:168] = dow_sales[-1]
            out[rows, 168:192] = dow_sales.mean(axis=0)
            out[rows, 270] = 1.0
            out[rows, 271] = 1.0

    # --- Momentum ---
    if K >= 7:
        rm7 = out[:, 72:96]
        l1 = out[:, 0:24]
        pos = rm7 > 0
        mom = np.zeros((M, 24), dtype=z)
        mom[pos] = l1[pos] / rm7[pos]
        out[:, 240:264] = mom
        has_valid = pos.any(axis=1)
        out[has_valid, 274] = 1.0

    return out


def build_dataset_arrays(sdata, split,
                         cont_mean=None, cont_std=None,
                         lag_mean=None, lag_std=None):
    """Build arrays for MLP variant F — vectorized, pre-allocated."""
    if split == 'train':
        d_min, d_max = 2, 83
    elif split == 'val':
        d_min, d_max = 84, 90
    elif split == 'test':
        d_min, d_max = 91, 97

    # Phase 1: count rows and gather series info
    series_info = []
    total_rows = 0
    for (sid, pid), sd in sdata.items():
        tmask = (sd['days'] >= d_min) & (sd['days'] <= d_max)
        n_t = int(tmask.sum())
        if n_t > 0:
            series_info.append((sid, pid, sd, tmask, n_t))
            total_rows += n_t

    print(f'    Pre-allocating {total_rows:,} rows × 275 lags...')

    # Phase 2: pre-allocate
    cat_arr = np.empty((total_rows, 4), dtype=np.int64)
    cont_arr = np.empty((total_rows, len(CONT_FEATURES)), dtype=np.float32)
    lag_arr = np.empty((total_rows, 275), dtype=np.float32)
    target_arr = np.empty((total_rows, 24), dtype=np.float32)
    stock_arr = np.empty((total_rows, 24), dtype=np.float32)
    sid_arr = np.empty(total_rows, dtype=np.int64)
    pid_arr = np.empty(total_rows, dtype=np.int64)

    # Phase 3: fill — one vectorized call per series, no inner day loop
    cursor = 0
    n_series = len(series_info)
    for i, (sid, pid, sd, tmask, n_t) in enumerate(series_info):
        if (i + 1) % 10000 == 0:
            print(f'    ... {i+1:,}/{n_series:,} serie')

        ti = np.where(tmask)[0]  # target indices within this series
        c = cursor
        cn = c + n_t

        cat_arr[c:cn, 0] = sid
        cat_arr[c:cn, 1] = pid
        cat_arr[c:cn, 2] = sd['city_id']
        cat_arr[c:cn, 3] = sd['dows'][ti]

        cont_arr[c:cn] = sd['conts'][ti]
        target_arr[c:cn] = sd['sales'][ti]
        stock_arr[c:cn] = sd['stock'][ti]
        sid_arr[c:cn] = sid
        pid_arr[c:cn] = pid

        if split == 'train':
            lag_arr[c:cn] = _compute_series_lags_train(
                sd['sales'], sd['dows'], ti)
        else:
            anchor = 83 if split == 'val' else 90
            lag_arr[c:cn] = _compute_series_lags_fixed(
                sd['sales'], sd['dows'], ti, anchor, sd['days'])

        cursor = cn

    # Normalize continuous features
    if cont_mean is None:
        cont_mean = cont_arr.mean(axis=0)
        cont_std = cont_arr.std(axis=0)
        cont_std[cont_std < 1e-8] = 1.0
    cont_arr = (cont_arr - cont_mean) / cont_std

    # Normalize lag features
    if lag_mean is None:
        lag_mean = lag_arr.mean(axis=0)
        lag_std = lag_arr.std(axis=0)
        lag_std[lag_std < 1e-8] = 1.0
    lag_arr = (lag_arr - lag_mean) / lag_std

    return {
        'cat': cat_arr,
        'cont': cont_arr,
        'lags': lag_arr,
        'targets': target_arr,
        'stock': stock_arr,
        'store_ids': sid_arr,
        'product_ids': pid_arr,
        'cont_mean': cont_mean,
        'cont_std': cont_std,
        'lag_mean': lag_mean,
        'lag_std': lag_std,
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
        emb_list = []
        for i, name in enumerate(self.emb_names):
            emb_list.append(self.embeddings[name](cat[:, i]))
        x = torch.cat(emb_list + [cont], dim=1)
        if lags.shape[1] > 0:
            x = torch.cat([x, lags], dim=1)
        return self.mlp(x)


def train_model(model, train_loader, val_data, device, lr, max_epochs, patience):
    """Train model with early stopping on val WAPE."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_wape = float('inf')
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for cat, cont, lags, targets in train_loader:
            cat = cat.to(device)
            cont = cont.to(device)
            lags = lags.to(device)
            targets = targets.to(device)

            preds = model(cat, cont, lags)
            loss = nn.functional.mse_loss(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / n_batches

        model.eval()
        with torch.no_grad():
            val_cat = torch.from_numpy(val_data['cat']).to(device)
            val_cont = torch.from_numpy(val_data['cont']).to(device)
            val_lags = torch.from_numpy(val_data['lags']).to(device)
            chunk_size = 10000
            all_preds = []
            for start in range(0, len(val_cat), chunk_size):
                end = min(start + chunk_size, len(val_cat))
                p = model(val_cat[start:end], val_cont[start:end],
                          val_lags[start:end])
                all_preds.append(p.cpu().numpy())
            val_preds = np.concatenate(all_preds, axis=0)

        val_obs = val_data['targets']
        sae = np.abs(val_preds - val_obs).sum()
        sao = np.abs(val_obs).sum()
        val_wape = sae / sao if sao > 0 else float('inf')

        if epoch % 5 == 0 or epoch == 1:
            print(f'    Epoch {epoch:3d}: train_loss={avg_train_loss:.6f}, '
                  f'val_WAPE={val_wape:.6f}')

        if val_wape < best_val_wape:
            best_val_wape = val_wape
            best_epoch = epoch
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'    Early stopping at epoch {epoch} '
                  f'(best epoch={best_epoch}, val_WAPE={best_val_wape:.6f})')
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return best_val_wape, best_epoch


def predict(model, data, device):
    """Generate predictions."""
    model.eval()
    cat_t = torch.from_numpy(data['cat']).to(device)
    cont_t = torch.from_numpy(data['cont']).to(device)
    lags_t = torch.from_numpy(data['lags']).to(device)
    all_preds = []
    chunk_size = 10000
    with torch.no_grad():
        for start in range(0, len(cat_t), chunk_size):
            end = min(start + chunk_size, len(cat_t))
            p = model(cat_t[start:end], cont_t[start:end], lags_t[start:end])
            all_preds.append(p.cpu().numpy())
    return np.concatenate(all_preds, axis=0)


# ---------------------------------------------------------------------------
# 6. Build datasets
# ---------------------------------------------------------------------------
print('\n6. Costruzione dataset MLP (lag features da completed_sales)...')
t0 = time.time()

print('  Building train data...')
train_data = build_dataset_arrays(completed_series_data, 'train')
n_cont = train_data['cont'].shape[1]
n_lags = train_data['lags'].shape[1]
print(f'  Train: {len(train_data["targets"]):,} samples, '
      f'cont={n_cont}, lags={n_lags}')

print('  Building val data...')
val_data = build_dataset_arrays(completed_series_data, 'val',
                                 cont_mean=train_data['cont_mean'],
                                 cont_std=train_data['cont_std'],
                                 lag_mean=train_data['lag_mean'],
                                 lag_std=train_data['lag_std'])
print(f'  Val:   {len(val_data["targets"]):,} samples')

print('  Building test data...')
test_data = build_dataset_arrays(completed_series_data, 'test',
                                  cont_mean=train_data['cont_mean'],
                                  cont_std=train_data['cont_std'],
                                  lag_mean=train_data['lag_mean'],
                                  lag_std=train_data['lag_std'])
print(f'  Test:  {len(test_data["targets"]):,} samples')

elapsed_ds = time.time() - t0
print(f'  Dataset construction: {elapsed_ds:.1f}s')

# Extract original S_obs for val and test (for evaluation vs real observations)
print('  Extracting original S_obs for evaluation...')


def extract_original_targets(sdata_orig, split):
    """Extract original S_obs targets for a split (same row ordering as
    build_dataset_arrays)."""
    if split == 'val':
        d_min, d_max = 84, 90
    elif split == 'test':
        d_min, d_max = 91, 97
    targets = []
    for (sid, pid), sd in sdata_orig.items():
        days = sd['days']
        for idx in range(len(days)):
            d = days[idx]
            if d_min <= d <= d_max:
                targets.append(sd['sales'][idx])
    return np.array(targets, dtype=np.float32)


val_obs_targets = extract_original_targets(series_cache, 'val')
test_obs_targets = extract_original_targets(series_cache, 'test')
print(f'  Val S_obs: {len(val_obs_targets):,}, '
      f'Test S_obs: {len(test_obs_targets):,}')

# Free completed data (no longer needed after dataset construction)
del completed_series_data, completed_sales
gc.collect()

# ---------------------------------------------------------------------------
# 7. Train MLP
# ---------------------------------------------------------------------------
print('\n7. Training MLP Stage 2...')
t0 = time.time()

torch.manual_seed(SEED)
model = RetailMLP(n_cont, n_lags, EMB_DIMS, CARDINALITIES, HIDDEN_SIZES)
n_params = sum(p.numel() for p in model.parameters())
print(f'  Model params: {n_params:,}')
print(f'  Input dim: {sum(EMB_DIMS.values())} (emb) + {n_cont} (cont) + '
      f'{n_lags} (lags) = {sum(EMB_DIMS.values()) + n_cont + n_lags}')

train_ds = RetailDataset(train_data['cat'], train_data['cont'],
                          train_data['lags'], train_data['targets'])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=0, pin_memory=False)

# Early stopping on val WAPE against completed_Y (consistent with notebook 10)
best_wape, best_epoch = train_model(
    model, train_loader, val_data, DEVICE, LR, MAX_EPOCHS, PATIENCE)

elapsed_train = time.time() - t0
print(f'  Best epoch: {best_epoch}, val WAPE (vs completed_Y): {best_wape:.6f}')
print(f'  Training time: {elapsed_train:.1f}s')

# Save model
torch.save(model.state_dict(),
           os.path.join(RESULTS_DIR, 'twostage_mlp_model.pt'))

del train_ds, train_loader, train_data
gc.collect()

# ---------------------------------------------------------------------------
# 8. Evaluation on val + test (against original S_obs)
# ---------------------------------------------------------------------------
print('\n8. Valutazione su val e test (vs S_obs originale)...')

pooled_results = {}
per_series_dfs = {}

for split_name, data, obs_targets in [('val', val_data, val_obs_targets),
                                        ('test', test_data, test_obs_targets)]:
    print(f'\n  {split_name}...')
    preds = predict(model, data, DEVICE)
    obs = obs_targets
    stock = data['stock']
    sids = data['store_ids']
    pids = data['product_ids']

    # Pooled metrics (against S_obs)
    p_flat = preds.ravel()
    o_flat = obs.ravel()
    s_flat = stock.ravel()

    r = {}
    for sub, smask in [('overall', np.ones(len(p_flat), dtype=bool)),
                       ('instock', s_flat == 0),
                       ('stockout', s_flat == 1)]:
        ef = (p_flat - o_flat)[smask]
        of = o_flat[smask]
        sae = np.abs(ef).sum()
        sao = np.abs(of).sum()
        r[f'wape_{sub}'] = sae / sao if sao > 0 else np.nan
        r[f'wpe_{sub}'] = ef.sum() / of.sum() if of.sum() != 0 else np.nan
        r[f'n_{sub}'] = int(smask.sum())
    pooled_results[split_name] = r

    # Per-series metrics
    print('    Calcolo metriche per-serie...')
    df_idx = pd.DataFrame({'sid': sids, 'pid': pids,
                           'row': np.arange(len(sids))})
    records = []
    for (sid, pid), grp in df_idx.groupby(['sid', 'pid']):
        idx = grp['row'].values
        m = compute_metrics(preds[idx], obs[idx], stock[idx])
        m['store_id'] = sid
        m['product_id'] = pid
        records.append(m)

    ps = pd.DataFrame(records)
    per_series_dfs[split_name] = ps
    out_path = os.path.join(RESULTS_DIR,
                            f'twostage_mlp_{split_name}_per_series.parquet')
    ps.to_parquet(out_path, index=False)
    print(f'    Salvato: {out_path} ({len(ps):,} serie)')

    del preds
    gc.collect()

# ---------------------------------------------------------------------------
# 9. Tabella risultati pooled
# ---------------------------------------------------------------------------
print(format_metrics_table(pooled_results, model_name='Two-Stage MLP'))

# ---------------------------------------------------------------------------
# 10. Distribuzione per-serie
# ---------------------------------------------------------------------------
METRIC_COLS = ['wape_overall', 'wape_instock', 'wape_stockout',
               'wpe_overall', 'wpe_instock', 'wpe_stockout']

print('\n' + '=' * 72)
print('  DISTRIBUZIONE METRICHE PER-SERIE')
print('=' * 72)

print(f'\n  {"Split":<8} {"Metric":<16} {"Mean":>8} {"Median":>8} '
      f'{"Std":>8} {"Q5":>8} {"Q95":>8} {"Valid":>7}')
print('  ' + '-' * 80)

for split_name, ps in per_series_dfs.items():
    for col in METRIC_COLS:
        vals = ps[col].dropna()
        if len(vals) == 0:
            continue
        q5, q95 = np.quantile(vals, [0.05, 0.95])
        print(f'  {split_name:<8} {col:<16} {vals.mean():>8.4f} '
              f'{vals.median():>8.4f} {vals.std():>8.4f} '
              f'{q5:>8.4f} {q95:>8.4f} {len(vals):>7,}')

# ---------------------------------------------------------------------------
# 11. Confronto con tutti i modelli
# ---------------------------------------------------------------------------
print('\n' + '=' * 72)
print('  CONFRONTO CON TUTTI I MODELLI (test)')
print('=' * 72)

all_baselines = {
    'Naive (direct)': 'naive_direct',
    'MA K=14 (direct)': 'ma_direct_K14',
    'Global Mean': 'global_mean',
    'DoW Mean': 'dow_mean',
    'LGB (var A)': 'lgb_a',
    'LGB (var F)': 'lgb_f',
    '2-Stage LGB': 'twostage_lgb',
    'MLP (var A)': 'mlp',
    'MLP (var F)': 'mlp_f',
    '2-Stage MLP': 'twostage_mlp',
}

print(f'\n  {"Model":<24} {"WAPE_in pool":>14} {"WAPE_in med":>14} '
      f'{"WPE_in med":>12} {"WAPE_all med":>14}')
print('  ' + '-' * 82)

for label, prefix in all_baselines.items():
    path = os.path.join(RESULTS_DIR, f'{prefix}_test_per_series.parquet')
    if not os.path.exists(path):
        continue
    ps_bl = pd.read_parquet(path)
    wape_in_med = ps_bl['wape_instock'].median()
    wpe_in_med = ps_bl['wpe_instock'].median()
    wape_all_med = ps_bl['wape_overall'].median()

    if prefix == 'twostage_mlp' and 'test' in pooled_results:
        wape_in_pool = pooled_results['test']['wape_instock']
    else:
        wape_in_pool = np.nan

    if np.isnan(wape_in_pool):
        print(f'  {label:<24} {"—":>14} {wape_in_med:>14.4f} '
              f'{wpe_in_med:>12.4f} {wape_all_med:>14.4f}')
    else:
        print(f'  {label:<24} {wape_in_pool:>14.4f} {wape_in_med:>14.4f} '
              f'{wpe_in_med:>12.4f} {wape_all_med:>14.4f}')

elapsed_total = time.time() - t0_total
print(f'\n  Tempo totale: {elapsed_total:.1f}s')

print('\n' + '=' * 72)
print('  DONE')
print('=' * 72)
