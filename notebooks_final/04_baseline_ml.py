"""
04_baseline_ml.py — Fase A2: Baseline ML/DL (senza imputation)
===============================================================
Impatto della Qualità dell'Imputation sul Demand Forecasting
di Prodotti Deperibili — CLAUDE_FINAL.md

4 modelli ML su dati sporchi (S_obs con zeri da stockout):
  1. LightGBM (no lags): solo feature esogene
  2. LightGBM (M5 lags): feature esogene + 11 lag features M5-style
  3. MLP (no lags): embeddings + feature continue
  4. MLP (M5 lags): embeddings + feature continue + M5-style lag features

Valutazione:
  - Solo ore in-stock, ground truth = S_obs
  - Metriche: WAPE e WPE (orario pooled, orario mediana per-serie,
              giornaliero pooled, giornaliero mediana per-serie)
  - Split: train gg 1-83, val gg 84-90, retrain gg 1-90, test eval HF

Direct forecast: lag calcolati da gg 1-83 (val) o gg 1-90 (test).
Nessun dato del periodo di forecast come input.

Eseguire con: freshnet/bin/python notebooks_final/04_baseline_ml.py
"""

import sys
import os
import gc
import time
import numpy as np
import pandas as pd

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = 'mps' if torch.backends.mps.is_available() else (
    'cuda' if torch.cuda.is_available() else 'cpu')

# --- Feature definitions ---
CONT_FEATURES = ['discount', 'avg_temperature', 'avg_humidity',
                  'precpt', 'avg_wind_level', 'holiday_flag', 'activity_flag']
CAT_FEATURES_LGB = ['store_id', 'product_id', 'city_id', 'dow', 'hour']
LAG_FEATURE_NAMES = [
    'lag_1d', 'lag_7d', 'lag_14d',
    'rmean_7d', 'rmean_14d', 'rstd_7d',
    'lag_dow', 'rmean_dow',
    'daily_total_lag1', 'daily_total_rmean7',
    'momentum_1d_7d',
]

# --- LightGBM hyperparameters ---
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
LGB_MAX_ROUNDS = 500
LGB_EARLY_STOP = 30

# --- MLP hyperparameters ---
MLP_BATCH_SIZE = 4096
MLP_LR = 1e-3
MLP_MAX_EPOCHS = 100
MLP_PATIENCE = 10
MLP_HIDDEN = [128, 64]
EMB_DIMS = {'store_id': 32, 'product_id': 32, 'city_id': 8, 'dow': 4}
CARDINALITIES = {'store_id': 898, 'product_id': 865, 'city_id': 18, 'dow': 7}

# ===========================================================================
print('=' * 72)
print('  FASE A2 — BASELINE ML/DL (senza imputation)')
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
print(f'  Full: {len(df_full):,} righe, {len(all_dates)} giorni, {n_series:,} serie')
print(f'  Device: {DEVICE}')

del df_train, df_eval

# Pre-parse hourly arrays
print('  Parsing hourly arrays...')
sales_all = np.array(df_full['hours_sale'].tolist(), dtype=np.float32)
stock_all = np.array(df_full['hours_stock_status'].tolist(), dtype=np.float32)
print('  Done.')


# ---------------------------------------------------------------------------
# 2. Build series cache for lag computation
# ---------------------------------------------------------------------------
print('\n2. Building series cache...')
series_cache = {}
for (sid, pid), grp in df_full.groupby(['store_id', 'product_id'], sort=False):
    grp_s = grp.sort_values('day_num')
    series_cache[(sid, pid)] = {
        'days': grp_s['day_num'].values,
        'dows': grp_s['dow'].values,
        'sales': sales_all[grp_s.index.values],
        'stock': stock_all[grp_s.index.values],
        'city_id': grp_s['city_id'].values[0],
        'conts': grp_s[CONT_FEATURES].values.astype(np.float32),
    }
print(f'  {len(series_cache):,} serie in cache')


# ---------------------------------------------------------------------------
# 3. Lag computation helper (shared by LGB and MLP)
# ---------------------------------------------------------------------------
def compute_lags_for_day(avail_sales, avail_dows, target_dow, K):
    """Compute 11 M5-style lag features for one day (vectorized across 24h).

    Args:
        avail_sales: (K, 24) sales history up to anchor
        avail_dows: (K,) day-of-week for each available day
        target_dow: int, day-of-week of target day
        K: number of available days

    Returns:
        dict of {feature_name: array(24,)} — NaN where not available
    """
    z = np.float32
    lags = {n: np.full(24, np.nan, dtype=z) for n in LAG_FEATURE_NAMES}

    if K == 0:
        return lags

    lags['lag_1d'] = avail_sales[-1]
    if K >= 7:
        lags['lag_7d'] = avail_sales[-7]
    if K >= 14:
        lags['lag_14d'] = avail_sales[-14]
    if K >= 7:
        lags['rmean_7d'] = avail_sales[-7:].mean(axis=0)
    if K >= 14:
        lags['rmean_14d'] = avail_sales[-14:].mean(axis=0)
    if K >= 2:
        w = min(7, K)
        lags['rstd_7d'] = avail_sales[-w:].std(axis=0)

    same_dow = avail_dows == target_dow
    if same_dow.any():
        dow_sales = avail_sales[same_dow]
        lags['lag_dow'] = dow_sales[-1]
        lags['rmean_dow'] = dow_sales.mean(axis=0)

    daily_totals = avail_sales.sum(axis=1)
    lags['daily_total_lag1'] = np.full(24, daily_totals[-1], dtype=z)
    if K >= 7:
        lags['daily_total_rmean7'] = np.full(24, daily_totals[-7:].mean(), dtype=z)

    # Momentum
    rm7 = lags['rmean_7d']
    l1 = lags['lag_1d']
    if not np.isnan(rm7).all():
        valid_h = (~np.isnan(l1)) & (~np.isnan(rm7)) & (rm7 > 0)
        if valid_h.any():
            mom = np.full(24, np.nan, dtype=z)
            mom[valid_h] = l1[valid_h] / rm7[valid_h]
            lags['momentum_1d_7d'] = mom

    return lags


# ---------------------------------------------------------------------------
# 4. LightGBM dataset builder (vectorized)
# ---------------------------------------------------------------------------
def build_lgb_dataset(split, use_lags):
    """Build flat per-hour dataset for LightGBM."""
    if split == 'train':
        d_min, d_max = 2, 83
    elif split == 'val':
        d_min, d_max = 84, 90
    else:
        d_min, d_max = 91, 97

    mask = (df_full['day_num'] >= d_min) & (df_full['day_num'] <= d_max)
    df_split = df_full[mask]
    idx_split = np.where(mask.values)[0]
    n_days = len(df_split)

    store_ids_day = df_split['store_id'].values
    product_ids_day = df_split['product_id'].values
    city_ids_day = df_split['city_id'].values
    dows_day = df_split['dow'].values
    conts_day = df_split[CONT_FEATURES].values.astype(np.float32)
    day_nums_day = df_split['day_num'].values

    sales_day = sales_all[idx_split]
    stock_day = stock_all[idx_split]

    n_hourly = n_days * 24
    hours = np.tile(np.arange(24, dtype=np.int32), n_days)

    store_ids_h = np.repeat(store_ids_day, 24)
    product_ids_h = np.repeat(product_ids_day, 24)
    city_ids_h = np.repeat(city_ids_day, 24)
    dows_h = np.repeat(dows_day, 24)
    conts_h = np.repeat(conts_day, 24, axis=0)

    y = sales_day.ravel().astype(np.float32)
    stock_flat = stock_day.ravel().astype(np.float32)

    feat_dict = {
        'store_id': store_ids_h, 'product_id': product_ids_h,
        'city_id': city_ids_h, 'dow': dows_h, 'hour': hours,
    }
    for j, c in enumerate(CONT_FEATURES):
        feat_dict[c] = conts_h[:, j]

    if use_lags:
        lag_arrays = {name: np.full(n_hourly, np.nan, dtype=np.float32)
                      for name in LAG_FEATURE_NAMES}

        print(f'      Computing lag features for {n_days:,} days...')
        for row_i in range(n_days):
            if (row_i + 1) % 500000 == 0:
                print(f'        ... {row_i+1:,}/{n_days:,}')

            sid = store_ids_day[row_i]
            pid = product_ids_day[row_i]
            d = day_nums_day[row_i]
            dow_val = dows_day[row_i]

            sc = series_cache[(sid, pid)]
            s_days = sc['days']
            s_dows = sc['dows']
            s_sales = sc['sales']

            if split == 'train':
                a_day = d - 1
            elif split == 'val':
                a_day = 83
            else:
                a_day = 90

            avail_mask = s_days <= a_day
            K = int(avail_mask.sum())
            hs = row_i * 24
            he = hs + 24

            if K > 0:
                lags = compute_lags_for_day(
                    s_sales[avail_mask], s_dows[avail_mask], dow_val, K)
                for name in LAG_FEATURE_NAMES:
                    lag_arrays[name][hs:he] = lags[name]

        for name in LAG_FEATURE_NAMES:
            feat_dict[name] = lag_arrays[name]
        del lag_arrays

    X = pd.DataFrame(feat_dict)
    del feat_dict
    gc.collect()

    for c in CAT_FEATURES_LGB:
        X[c] = X[c].astype('category')

    return X, y, stock_flat, store_ids_h, product_ids_h


# ---------------------------------------------------------------------------
# 5. MLP dataset builder (vectorized, using series_cache)
# ---------------------------------------------------------------------------
def build_mlp_dataset(split, use_lags, cont_mean=None, cont_std=None,
                      lag_mean=None, lag_std=None):
    """Build arrays for MLP (day-level, 24h output). Vectorized via series_cache."""
    if split == 'train':
        d_min, d_max = 2, 83
    elif split == 'val':
        d_min, d_max = 84, 90
    else:
        d_min, d_max = 91, 97

    cat_list, cont_list, lag_list = [], [], []
    target_list, stock_list = [], []
    sid_list, pid_list = [], []

    n_series_done = 0
    for (sid, pid), sd in series_cache.items():
        n_series_done += 1
        if n_series_done % 10000 == 0:
            print(f'      ... {n_series_done:,}/{len(series_cache):,} serie')

        days = sd['days']
        dows = sd['dows']
        sales = sd['sales']
        stock = sd['stock']
        city = sd['city_id']
        conts = sd['conts']

        for idx in range(len(days)):
            d = days[idx]
            if d < d_min or d > d_max:
                continue

            if split == 'train':
                a_day = d - 1
            elif split == 'val':
                a_day = 83
            else:
                a_day = 90

            cat_list.append([sid, pid, city, dows[idx]])
            cont_list.append(conts[idx])

            if use_lags:
                avail_mask = days <= a_day
                K = int(avail_mask.sum())
                lag_dict = compute_lags_for_day(
                    sales[avail_mask], dows[avail_mask], dows[idx], K) if K > 0 \
                    else {n: np.full(24, np.nan, dtype=np.float32) for n in LAG_FEATURE_NAMES}

                # Pack as: 11 features x 24h + 11 binary masks = 275
                feat_arrays = []
                masks = np.zeros(11, dtype=np.float32)
                for fi, name in enumerate(LAG_FEATURE_NAMES):
                    arr = lag_dict[name]
                    has_val = ~np.isnan(arr).all()
                    if has_val:
                        masks[fi] = 1.0
                        arr_clean = np.where(np.isnan(arr), 0.0, arr).astype(np.float32)
                    else:
                        arr_clean = np.zeros(24, dtype=np.float32)
                    feat_arrays.append(arr_clean)
                feat_arrays.append(masks)
                lag_list.append(np.concatenate(feat_arrays))
            else:
                lag_list.append(np.array([], dtype=np.float32))

            target_list.append(sales[idx])
            stock_list.append(stock[idx])
            sid_list.append(sid)
            pid_list.append(pid)

    cat_arr = np.array(cat_list, dtype=np.int64)
    cont_arr = np.array(cont_list, dtype=np.float32)
    target_arr = np.array(target_list, dtype=np.float32)
    stock_arr = np.array(stock_list, dtype=np.float32)

    if len(lag_list) > 0 and len(lag_list[0]) > 0:
        lag_arr = np.array(lag_list, dtype=np.float32)
    else:
        lag_arr = np.zeros((len(cat_list), 0), dtype=np.float32)

    # Normalize continuous
    if cont_mean is None:
        cont_mean = cont_arr.mean(axis=0)
        cont_std = cont_arr.std(axis=0)
        cont_std[cont_std < 1e-8] = 1.0
    cont_arr = (cont_arr - cont_mean) / cont_std

    # Normalize lags (only the 11×24=264 value features, not the 11 masks)
    if lag_arr.shape[1] > 0:
        if lag_mean is None:
            lag_mean = lag_arr.mean(axis=0)
            lag_std = lag_arr.std(axis=0)
            lag_std[lag_std < 1e-8] = 1.0
        lag_arr = (lag_arr - lag_mean) / lag_std

    return {
        'cat': cat_arr, 'cont': cont_arr, 'lags': lag_arr,
        'targets': target_arr, 'stock': stock_arr,
        'store_ids': np.array(sid_list, dtype=np.int64),
        'product_ids': np.array(pid_list, dtype=np.int64),
        'cont_mean': cont_mean, 'cont_std': cont_std,
        'lag_mean': lag_mean, 'lag_std': lag_std,
    }


# ---------------------------------------------------------------------------
# 6. MLP model and training
# ---------------------------------------------------------------------------
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
        x = torch.cat(emb_list + [cont], dim=1)
        if lags.shape[1] > 0:
            x = torch.cat([x, lags], dim=1)
        return self.mlp(x)


def train_mlp(model, train_loader, val_data, device):
    """Train MLP with early stopping on val WAPE (in-stock, pooled)."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LR)

    best_val_wape = float('inf')
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0

    val_stock = val_data['stock']
    val_instock = val_stock == 0

    for epoch in range(1, MLP_MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for cat, cont, lags, targets in train_loader:
            cat, cont, lags, targets = (
                cat.to(device), cont.to(device), lags.to(device), targets.to(device))
            preds = model(cat, cont, lags)
            loss = nn.functional.mse_loss(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        # Validate: WAPE on in-stock hours (pooled)
        model.eval()
        with torch.no_grad():
            val_cat = torch.from_numpy(val_data['cat']).to(device)
            val_cont = torch.from_numpy(val_data['cont']).to(device)
            val_lags = torch.from_numpy(val_data['lags']).to(device)

            all_preds = []
            for start in range(0, len(val_cat), 10000):
                end = min(start + 10000, len(val_cat))
                p = model(val_cat[start:end], val_cont[start:end], val_lags[start:end])
                all_preds.append(p.cpu().numpy())
            val_preds = np.concatenate(all_preds, axis=0)

        val_obs = val_data['targets']
        p_h = val_preds[val_instock]
        o_h = val_obs[val_instock]
        sae = np.abs(p_h - o_h).sum()
        sao = np.abs(o_h).sum()
        val_wape = sae / sao if sao > 0 else float('inf')

        if epoch % 5 == 0 or epoch == 1:
            print(f'    Epoch {epoch:3d}: train_loss={train_loss/n_batches:.6f}, '
                  f'val_WAPE_instock={val_wape:.6f}')

        if val_wape < best_val_wape:
            best_val_wape = val_wape
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= MLP_PATIENCE:
            print(f'    Early stopping at epoch {epoch} '
                  f'(best={best_epoch}, WAPE={best_val_wape:.6f})')
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return best_val_wape, best_epoch


def predict_mlp(model, data, device):
    model.eval()
    cat_t = torch.from_numpy(data['cat']).to(device)
    cont_t = torch.from_numpy(data['cont']).to(device)
    lags_t = torch.from_numpy(data['lags']).to(device)

    all_preds = []
    with torch.no_grad():
        for start in range(0, len(cat_t), 10000):
            end = min(start + 10000, len(cat_t))
            p = model(cat_t[start:end], cont_t[start:end], lags_t[start:end])
            all_preds.append(p.cpu().numpy())
    return np.concatenate(all_preds, axis=0)


# ---------------------------------------------------------------------------
# 7. Evaluation helpers (in-stock only, hourly + daily)
# ---------------------------------------------------------------------------
def eval_instock_flat(preds_flat, obs_flat, stock_flat, store_ids, product_ids):
    """Evaluate from flat hourly arrays (LGB format). Returns (pooled, ps_df)."""
    instock = stock_flat == 0

    # Hourly pooled
    p_h, o_h = preds_flat[instock], obs_flat[instock]
    sae_h, sao_h = np.abs(p_h - o_h).sum(), np.abs(o_h).sum()
    se_h, so_h = (p_h - o_h).sum(), o_h.sum()

    # Daily pooled
    n_days_total = len(preds_flat) // 24
    pred_days = preds_flat.reshape(n_days_total, 24)
    obs_days = obs_flat.reshape(n_days_total, 24)
    stock_days = stock_flat.reshape(n_days_total, 24)
    sid_days = store_ids.reshape(n_days_total, 24)[:, 0]
    pid_days = product_ids.reshape(n_days_total, 24)[:, 0]

    sae_d, sao_d, se_d, so_d = 0., 0., 0., 0.
    for d in range(n_days_total):
        m_d = stock_days[d] == 0
        if m_d.any():
            pd_v, od_v = pred_days[d, m_d].sum(), obs_days[d, m_d].sum()
            sae_d += abs(pd_v - od_v)
            sao_d += abs(od_v)
            se_d += pd_v - od_v
            so_d += od_v

    pooled = {
        'hourly_wape': sae_h / sao_h if sao_h > 0 else np.nan,
        'hourly_wpe': se_h / so_h if so_h != 0 else np.nan,
        'daily_wape': sae_d / sao_d if sao_d > 0 else np.nan,
        'daily_wpe': se_d / so_d if so_d != 0 else np.nan,
    }

    # Per-series
    df_tmp = pd.DataFrame({
        'sid': np.repeat(sid_days, 24), 'pid': np.repeat(pid_days, 24),
        'day_idx': np.repeat(np.arange(n_days_total), 24),
        'pred': preds_flat.astype(np.float64),
        'obs': obs_flat.astype(np.float64),
        'stock': stock_flat,
    })
    # Use day-level for both hourly and daily per-series
    records = _per_series_from_df(df_tmp)
    ps_df = pd.DataFrame(records)
    return pooled, ps_df


def eval_instock_day(preds_24, obs_24, stock_24, store_ids, product_ids):
    """Evaluate from day-level arrays (MLP format). Returns (pooled, ps_df)."""
    instock = stock_24 == 0

    p_h, o_h = preds_24[instock], obs_24[instock]
    sae_h, sao_h = np.abs(p_h - o_h).sum(), np.abs(o_h).sum()
    se_h, so_h = (p_h - o_h).sum(), o_h.sum()

    n_samples = preds_24.shape[0]
    sae_d, sao_d, se_d, so_d = 0., 0., 0., 0.
    for d in range(n_samples):
        m_d = instock[d]
        if m_d.any():
            pd_v, od_v = preds_24[d, m_d].sum(), obs_24[d, m_d].sum()
            sae_d += abs(pd_v - od_v)
            sao_d += abs(od_v)
            se_d += pd_v - od_v
            so_d += od_v

    pooled = {
        'hourly_wape': sae_h / sao_h if sao_h > 0 else np.nan,
        'hourly_wpe': se_h / so_h if so_h != 0 else np.nan,
        'daily_wape': sae_d / sao_d if sao_d > 0 else np.nan,
        'daily_wpe': se_d / so_d if so_d != 0 else np.nan,
    }

    # Per-series
    records = []
    series_map = {}
    for i in range(n_samples):
        key = (store_ids[i], product_ids[i])
        if key not in series_map:
            series_map[key] = []
        series_map[key].append(i)

    for (sid, pid), indices in series_map.items():
        sae_sh, sao_sh, se_sh, so_sh = 0., 0., 0., 0.
        sae_sd, sao_sd, se_sd, so_sd = 0., 0., 0., 0.
        n_in, n_valid_d = 0, 0

        for i in indices:
            m_d = instock[i]
            n_in += int(m_d.sum())
            sae_sh += np.abs(preds_24[i, m_d] - obs_24[i, m_d]).sum()
            sao_sh += np.abs(obs_24[i, m_d]).sum()
            se_sh += (preds_24[i, m_d] - obs_24[i, m_d]).sum()
            so_sh += obs_24[i, m_d].sum()
            if m_d.any():
                pd_v, od_v = preds_24[i, m_d].sum(), obs_24[i, m_d].sum()
                sae_sd += abs(pd_v - od_v)
                sao_sd += abs(od_v)
                se_sd += pd_v - od_v
                so_sd += od_v
                n_valid_d += 1

        records.append({
            'store_id': sid, 'product_id': pid,
            'hourly_wape': sae_sh / sao_sh if sao_sh > 0 else np.nan,
            'hourly_wpe': se_sh / so_sh if so_sh != 0 else np.nan,
            'daily_wape': sae_sd / sao_sd if sao_sd > 0 else np.nan,
            'daily_wpe': se_sd / so_sd if so_sd != 0 else np.nan,
            'n_hours_instock': n_in, 'n_days_valid': n_valid_d,
        })

    ps_df = pd.DataFrame(records)
    return pooled, ps_df


def _per_series_from_df(df_tmp):
    """Helper: compute per-series metrics from flat df with sid/pid/day_idx/pred/obs/stock."""
    records = []
    for (sid, pid), grp in df_tmp.groupby(['sid', 'pid'], sort=False):
        instock_g = grp['stock'].values == 0
        sao_s = np.abs(grp['obs'].values[instock_g]).sum()
        sae_s = np.abs(grp['pred'].values[instock_g] - grp['obs'].values[instock_g]).sum()
        se_s = (grp['pred'].values[instock_g] - grp['obs'].values[instock_g]).sum()
        so_s = grp['obs'].values[instock_g].sum()

        h_wape = sae_s / sao_s if sao_s > 0 else np.nan
        h_wpe = se_s / so_s if so_s != 0 else np.nan

        sae_ds, sao_ds, se_ds, so_ds = 0., 0., 0., 0.
        n_valid_d = 0
        for di, dgrp in grp.groupby('day_idx', sort=False):
            dm = dgrp['stock'].values == 0
            if dm.any():
                pd_v = dgrp['pred'].values[dm].sum()
                od_v = dgrp['obs'].values[dm].sum()
                sae_ds += abs(pd_v - od_v)
                sao_ds += abs(od_v)
                se_ds += pd_v - od_v
                so_ds += od_v
                n_valid_d += 1

        records.append({
            'store_id': sid, 'product_id': pid,
            'hourly_wape': h_wape, 'hourly_wpe': h_wpe,
            'daily_wape': sae_ds / sao_ds if sao_ds > 0 else np.nan,
            'daily_wpe': se_ds / so_ds if so_ds != 0 else np.nan,
            'n_hours_instock': int(instock_g.sum()), 'n_days_valid': n_valid_d,
        })
    return records


# ===========================================================================
# PART 1: LIGHTGBM
# ===========================================================================
print('\n' + '=' * 72)
print('  PARTE 1 — LIGHTGBM')
print('=' * 72)

lgb_all_results = {}

for use_lags, variant_label in [(False, 'LGB (no lags)'), (True, 'LGB (M5 lags)')]:
    print(f'\n  === {variant_label} ===')
    t0 = time.time()

    print('    Building train dataset...')
    X_train, y_train, _, _, _ = build_lgb_dataset('train', use_lags)
    print(f'    Train: {len(X_train):,} rows, {X_train.shape[1]} features')

    print('    Building val dataset...')
    X_val, y_val, stock_val, sids_val, pids_val = build_lgb_dataset('val', use_lags)
    print(f'    Val:   {len(X_val):,} rows')

    print('    Training LightGBM...')
    lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=True)
    lgb_val_ds = lgb.Dataset(X_val, y_val, reference=lgb_train, free_raw_data=True)

    model = lgb.train(
        LGB_PARAMS, lgb_train,
        num_boost_round=LGB_MAX_ROUNDS,
        valid_sets=[lgb_val_ds], valid_names=['val'],
        callbacks=[lgb.early_stopping(LGB_EARLY_STOP), lgb.log_evaluation(100)],
    )

    best_iter = model.best_iteration
    best_mae = model.best_score['val']['l1']
    print(f'    Best iter: {best_iter}, MAE: {best_mae:.6f}')

    # Evaluate val
    preds_val = np.clip(model.predict(X_val), 0, None)
    pooled_val, ps_val = eval_instock_flat(preds_val, y_val, stock_val, sids_val, pids_val)
    med_val = {c: ps_val[c].dropna().median() for c in
               ['hourly_wape', 'hourly_wpe', 'daily_wape', 'daily_wpe']}
    print(f'    Val WAPE_h pool={pooled_val["hourly_wape"]:.4f}, med={med_val["hourly_wape"]:.4f}')

    del X_train, y_train, lgb_train, lgb_val_ds, X_val, preds_val
    gc.collect()

    # Save model, evaluate test
    model_path = os.path.join(RESULTS_DIR, f'lgb_{"nolags" if not use_lags else "m5lags"}.txt')
    model.save_model(model_path)

    print('    Building test dataset...')
    X_test, y_test, stock_test, sids_test, pids_test = build_lgb_dataset('test', use_lags)
    print(f'    Test:  {len(X_test):,} rows')

    preds_test = np.clip(model.predict(X_test), 0, None)
    pooled_test, ps_test = eval_instock_flat(preds_test, y_test, stock_test, sids_test, pids_test)
    med_test = {c: ps_test[c].dropna().median() for c in
                ['hourly_wape', 'hourly_wpe', 'daily_wape', 'daily_wpe']}

    elapsed = time.time() - t0

    # Feature importance
    importance = model.feature_importance(importance_type='gain')
    feat_names = model.feature_name()
    fi = sorted(zip(feat_names, importance), key=lambda x: x[1], reverse=True)
    print(f'    Top 5 features: {", ".join(f"{n}({v:,.0f})" for n, v in fi[:5])}')

    safe_name = 'lgb_nolags' if not use_lags else 'lgb_m5lags'
    ps_val.to_parquet(os.path.join(RESULTS_DIR, f'{safe_name}_val_per_series.parquet'), index=False)
    ps_test.to_parquet(os.path.join(RESULTS_DIR, f'{safe_name}_test_per_series.parquet'), index=False)

    lgb_all_results[variant_label] = {
        'val': {'pooled': pooled_val, 'median': med_val},
        'test': {'pooled': pooled_test, 'median': med_test},
        'best_iter': best_iter, 'best_mae': best_mae,
        'elapsed': elapsed, 'feature_importance': fi[:10],
    }

    print(f'    Test WAPE_h pool={pooled_test["hourly_wape"]:.4f}, '
          f'med={med_test["hourly_wape"]:.4f}, time={elapsed:.0f}s')

    del X_test, preds_test, model
    gc.collect()


# ===========================================================================
# PART 2: MLP
# ===========================================================================
print('\n' + '=' * 72)
print('  PARTE 2 — MLP')
print('=' * 72)

mlp_all_results = {}

for use_lags, variant_label in [(False, 'MLP (no lags)'), (True, 'MLP (M5 lags)')]:
    print(f'\n  === {variant_label} ===')
    t0 = time.time()

    print('    Building train dataset...')
    train_data = build_mlp_dataset('train', use_lags)
    print(f'    Train: {len(train_data["targets"]):,} samples, '
          f'cont={train_data["cont"].shape[1]}, lags={train_data["lags"].shape[1]}')

    print('    Building val dataset...')
    val_data = build_mlp_dataset('val', use_lags,
                                  cont_mean=train_data['cont_mean'],
                                  cont_std=train_data['cont_std'],
                                  lag_mean=train_data['lag_mean'],
                                  lag_std=train_data['lag_std'])
    print(f'    Val:   {len(val_data["targets"]):,} samples')

    n_lags = train_data['lags'].shape[1]
    n_cont = train_data['cont'].shape[1]
    model = RetailMLP(n_cont, n_lags, EMB_DIMS, CARDINALITIES, MLP_HIDDEN)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'    Model: {n_params:,} parameters, input_dim={sum(EMB_DIMS.values())+n_cont+n_lags}')

    train_ds = RetailDataset(train_data['cat'], train_data['cont'],
                              train_data['lags'], train_data['targets'])
    train_loader = DataLoader(train_ds, batch_size=MLP_BATCH_SIZE, shuffle=True,
                               num_workers=0, pin_memory=False)

    best_wape, best_epoch = train_mlp(model, train_loader, val_data, DEVICE)

    # Evaluate val
    val_preds = predict_mlp(model, val_data, DEVICE)
    pooled_val, ps_val = eval_instock_day(
        val_preds, val_data['targets'], val_data['stock'],
        val_data['store_ids'], val_data['product_ids'])
    med_val = {c: ps_val[c].dropna().median() for c in
               ['hourly_wape', 'hourly_wpe', 'daily_wape', 'daily_wpe']}
    print(f'    Val WAPE_h pool={pooled_val["hourly_wape"]:.4f}, med={med_val["hourly_wape"]:.4f}')

    # Build test
    print('    Building test dataset...')
    test_data = build_mlp_dataset('test', use_lags,
                                   cont_mean=train_data['cont_mean'],
                                   cont_std=train_data['cont_std'],
                                   lag_mean=train_data['lag_mean'],
                                   lag_std=train_data['lag_std'])
    print(f'    Test:  {len(test_data["targets"]):,} samples')

    test_preds = predict_mlp(model, test_data, DEVICE)
    pooled_test, ps_test = eval_instock_day(
        test_preds, test_data['targets'], test_data['stock'],
        test_data['store_ids'], test_data['product_ids'])
    med_test = {c: ps_test[c].dropna().median() for c in
                ['hourly_wape', 'hourly_wpe', 'daily_wape', 'daily_wpe']}

    elapsed = time.time() - t0

    safe_name = 'mlp_nolags' if not use_lags else 'mlp_m5lags'
    ps_val.to_parquet(os.path.join(RESULTS_DIR, f'{safe_name}_val_per_series.parquet'), index=False)
    ps_test.to_parquet(os.path.join(RESULTS_DIR, f'{safe_name}_test_per_series.parquet'), index=False)
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f'{safe_name}.pt'))

    mlp_all_results[variant_label] = {
        'val': {'pooled': pooled_val, 'median': med_val},
        'test': {'pooled': pooled_test, 'median': med_test},
        'best_epoch': best_epoch, 'n_params': n_params, 'elapsed': elapsed,
    }

    print(f'    Test WAPE_h pool={pooled_test["hourly_wape"]:.4f}, '
          f'med={med_test["hourly_wape"]:.4f}, time={elapsed:.0f}s')

    del train_data, val_data, test_data, model, train_ds, train_loader
    gc.collect()
    if DEVICE == 'mps':
        torch.mps.empty_cache()


# ===========================================================================
# SUMMARY
# ===========================================================================
print('\n' + '=' * 72)
print('  RIEPILOGO — TUTTI I MODELLI A2 (test, in-stock only)')
print('=' * 72)

all_ml_results = {**lgb_all_results, **mlp_all_results}

print(f'\n  {"Model":<20} '
      f'{"WAPE_h pool":>12} {"WPE_h pool":>11} '
      f'{"WAPE_h med":>11} {"WPE_h med":>10} '
      f'{"WAPE_d pool":>12} {"WPE_d pool":>11}')
print('  ' + '-' * 90)

for label, res in all_ml_results.items():
    p = res['test']['pooled']
    m = res['test']['median']
    print(f'  {label:<20} '
          f'{p["hourly_wape"]:>12.4f} {p["hourly_wpe"]:>11.4f} '
          f'{m["hourly_wape"]:>11.4f} {m["hourly_wpe"]:>10.4f} '
          f'{p["daily_wape"]:>12.4f} {p["daily_wpe"]:>11.4f}')

# Figures
print('\n  Generazione figure...')

labels = list(all_ml_results.keys())
x = np.arange(len(labels))
w = 0.35

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Fase A2 — Baseline ML (test, in-stock only)', fontsize=14)

wape_pool = [all_ml_results[l]['test']['pooled']['hourly_wape'] for l in labels]
wape_med = [all_ml_results[l]['test']['median']['hourly_wape'] for l in labels]

ax = axes[0]
ax.bar(x, wape_pool, w, color='steelblue', alpha=0.8)
ax.set_ylabel('WAPE (pooled)')
ax.set_title('Hourly WAPE — pooled')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
for i, v in enumerate(wape_pool):
    ax.text(i, v + 0.005, f'{v:.4f}', ha='center', va='bottom', fontsize=8)

ax = axes[1]
ax.bar(x, wape_med, w, color='darkorange', alpha=0.8)
ax.set_ylabel('WAPE (median per-serie)')
ax.set_title('Hourly WAPE — median per-serie')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
for i, v in enumerate(wape_med):
    ax.text(i, v + 0.005, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig03_ml_baselines_test.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Salvata: fig03_ml_baselines_test.png')

# Boxplot
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Fase A2 — Distribuzione WAPE orario per-serie (test, in-stock)', fontsize=13)

box_data = []
safe_names = ['lgb_nolags', 'lgb_m5lags', 'mlp_nolags', 'mlp_m5lags']
for sn in safe_names:
    ps = pd.read_parquet(os.path.join(RESULTS_DIR, f'{sn}_test_per_series.parquet'))
    vals = ps['hourly_wape'].dropna()
    box_data.append(vals.clip(upper=vals.quantile(0.99)).values)

bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True, widths=0.6)
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
for ml in bp['medians']:
    ml.set_color('red')
    ml.set_linewidth(2)

for i, l in enumerate(labels):
    med = all_ml_results[l]['test']['median']['hourly_wape']
    ax.text(i + 1, med + 0.01, f'{med:.4f}', ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='red')

ax.set_ylabel('WAPE (hourly, in-stock)')
ax.tick_params(axis='x', rotation=20)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig04_ml_boxplot_test.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Salvata: fig04_ml_boxplot_test.png')

print('\n' + '=' * 72)
print('  DONE — 04_baseline_ml.py')
print('=' * 72)
