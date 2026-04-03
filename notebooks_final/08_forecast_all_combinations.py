"""
08_forecast_all_combinations.py — Fase B2: Forecast su completed_sales
======================================================================
Per ogni imputer × ogni forecaster con lag, allena il forecaster sui
completed_sales e valuta sul test set (eval HF).

Imputer (righe): media_cond, media_glob, mediana_cond, lgb
Forecaster (colonne, solo con M5 lags): LGB, MLP

Le varianti no-lags sono identiche alla Fase A (i lag non usano completed_sales)
→ risultati riusati, non rieseguiti.

Per ogni cella:
  1. Carica completed_sales dell'imputer
  2. Calcola lag M5-style dai completed_sales (decontaminati)
  3. Train forecaster su gg 1-83, val su gg 84-90, retrain non necessario
     (direct forecast: lag dal gg 1-90 via completed_sales)
  4. Test su eval HF, valutazione solo ore in-stock

Eseguire con: freshnet/bin/python notebooks_final/08_forecast_all_combinations.py
"""

import sys
import os
import gc
import time
import functools
import numpy as np
import pandas as pd

print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)

import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
COMPLETED_DIR = os.path.join(DATA_DIR, 'completed_sales')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = 'mps' if torch.backends.mps.is_available() else (
    'cuda' if torch.cuda.is_available() else 'cpu')

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

LGB_PARAMS = {
    'objective': 'regression', 'metric': 'mae',
    'num_leaves': 31, 'learning_rate': 0.1,
    'feature_fraction': 0.8, 'bagging_fraction': 0.3, 'bagging_freq': 1,
    'min_child_samples': 500, 'max_bin': 127,
    'verbose': -1, 'num_threads': -1, 'seed': SEED,
}
LGB_MAX_ROUNDS = 500
LGB_EARLY_STOP = 30

MLP_BATCH_SIZE = 4096
MLP_LR = 1e-3
MLP_MAX_EPOCHS = 100
MLP_PATIENCE = 10
MLP_HIDDEN = [128, 64]
EMB_DIMS = {'store_id': 32, 'product_id': 32, 'city_id': 8, 'dow': 4}
CARDINALITIES = {'store_id': 898, 'product_id': 865, 'city_id': 18, 'dow': 7}

IMPUTERS = {
    'media_cond': 'Media condizionata',
    'media_glob': 'Media globale',
    'mediana_cond': 'Mediana condizionata',
    'lgb': 'LGB imputer',
}

# ===========================================================================
print('=' * 72)
print('  FASE B2 — FORECAST SU COMPLETED_SALES')
print('=' * 72)

# ---------------------------------------------------------------------------
# 1. Caricamento dati base (per eval set e features esogene)
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

# Original sales and stock (for eval ground truth)
sales_orig = np.array(df_full['hours_sale'].tolist(), dtype=np.float32)
stock_orig = np.array(df_full['hours_stock_status'].tolist(), dtype=np.float32)

print(f'  Full: {len(df_full):,} righe, {len(all_dates)} giorni')
print(f'  Device: {DEVICE}')

del df_train_hf, df_eval


# ---------------------------------------------------------------------------
# 2. Lag computation helper
# ---------------------------------------------------------------------------
def compute_lags_for_day(avail_sales, avail_dows, target_dow, K):
    """Compute 11 M5-style lag features for one day (24h)."""
    z = np.float32
    lags = {n: np.full(24, np.nan, dtype=z) for n in LAG_FEATURE_NAMES}
    if K == 0:
        return lags

    lags['lag_1d'] = avail_sales[-1]
    if K >= 7:  lags['lag_7d'] = avail_sales[-7]
    if K >= 14: lags['lag_14d'] = avail_sales[-14]
    if K >= 7:  lags['rmean_7d'] = avail_sales[-7:].mean(axis=0)
    if K >= 14: lags['rmean_14d'] = avail_sales[-14:].mean(axis=0)
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

    rm7, l1 = lags['rmean_7d'], lags['lag_1d']
    if not np.isnan(rm7).all():
        valid_h = (~np.isnan(l1)) & (~np.isnan(rm7)) & (rm7 > 0)
        if valid_h.any():
            mom = np.full(24, np.nan, dtype=z)
            mom[valid_h] = l1[valid_h] / rm7[valid_h]
            lags['momentum_1d_7d'] = mom

    return lags


# ---------------------------------------------------------------------------
# 3. Build series cache from completed_sales
# ---------------------------------------------------------------------------
def build_series_cache(completed_sales_24):
    """Build series cache using completed_sales for lag computation."""
    cache = {}
    for (sid, pid), grp in df_full.groupby(['store_id', 'product_id'], sort=False):
        grp_s = grp.sort_values('day_num')
        idx = grp_s.index.values
        cache[(sid, pid)] = {
            'days': grp_s['day_num'].values,
            'dows': grp_s['dow'].values,
            'sales_completed': completed_sales_24[idx],  # from completed_sales
            'sales_orig': sales_orig[idx],                # original for targets
            'stock': stock_orig[idx],
            'city_id': grp_s['city_id'].values[0],
            'conts': grp_s[CONT_FEATURES].values.astype(np.float32),
        }
    return cache


# ---------------------------------------------------------------------------
# 4. LGB dataset builder with completed_sales lags
# ---------------------------------------------------------------------------
def build_lgb_dataset_completed(series_cache, split):
    """Build flat per-hour dataset with lag from completed_sales."""
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

    # Target = original S_obs (not completed)
    sales_day = sales_orig[idx_split]
    stock_day = stock_orig[idx_split]

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

    # Lag features from completed_sales
    lag_arrays = {name: np.full(n_hourly, np.nan, dtype=np.float32)
                  for name in LAG_FEATURE_NAMES}

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
        s_sales = sc['sales_completed']  # COMPLETED sales for lags

        a_day = d - 1 if split == 'train' else (83 if split == 'val' else 90)
        avail_mask = s_days <= a_day
        K = int(avail_mask.sum())
        hs = row_i * 24
        he = hs + 24

        if K > 0:
            lags = compute_lags_for_day(s_sales[avail_mask], s_dows[avail_mask], dow_val, K)
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
# 5. MLP dataset builder with completed_sales lags
# ---------------------------------------------------------------------------
def build_mlp_dataset_completed(series_cache, split, cont_mean=None, cont_std=None,
                                 lag_mean=None, lag_std=None):
    """Build MLP dataset with lags from completed_sales."""
    if split == 'train':
        d_min, d_max = 2, 83
    elif split == 'val':
        d_min, d_max = 84, 90
    else:
        d_min, d_max = 91, 97

    cat_list, cont_list, lag_list = [], [], []
    target_list, stock_list, sid_list, pid_list = [], [], [], []

    n_done = 0
    for (sid, pid), sd in series_cache.items():
        n_done += 1
        if n_done % 10000 == 0:
            print(f'        ... {n_done:,}/{len(series_cache):,} serie')

        days, dows = sd['days'], sd['dows']
        sales_comp = sd['sales_completed']
        sales_real = sd['sales_orig']
        stock = sd['stock']
        city, conts = sd['city_id'], sd['conts']

        for idx in range(len(days)):
            d = days[idx]
            if d < d_min or d > d_max:
                continue

            a_day = d - 1 if split == 'train' else (83 if split == 'val' else 90)

            cat_list.append([sid, pid, city, dows[idx]])
            cont_list.append(conts[idx])
            target_list.append(sales_real[idx])  # target = original S_obs
            stock_list.append(stock[idx])
            sid_list.append(sid)
            pid_list.append(pid)

            # Lags from completed_sales
            avail_mask = days <= a_day
            K = int(avail_mask.sum())
            if K > 0:
                lag_dict = compute_lags_for_day(
                    sales_comp[avail_mask], dows[avail_mask], dows[idx], K)
            else:
                lag_dict = {n: np.full(24, np.nan, dtype=np.float32)
                            for n in LAG_FEATURE_NAMES}

            feat_arrays, masks = [], np.zeros(11, dtype=np.float32)
            for fi, name in enumerate(LAG_FEATURE_NAMES):
                arr = lag_dict[name]
                if not np.isnan(arr).all():
                    masks[fi] = 1.0
                    feat_arrays.append(np.where(np.isnan(arr), 0.0, arr).astype(np.float32))
                else:
                    feat_arrays.append(np.zeros(24, dtype=np.float32))
            feat_arrays.append(masks)
            lag_list.append(np.concatenate(feat_arrays))

    cat_arr = np.array(cat_list, dtype=np.int64)
    cont_arr = np.array(cont_list, dtype=np.float32)
    target_arr = np.array(target_list, dtype=np.float32)
    stock_arr = np.array(stock_list, dtype=np.float32)
    lag_arr = np.array(lag_list, dtype=np.float32)

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
        'store_ids': np.array(sid_list, dtype=np.int64),
        'product_ids': np.array(pid_list, dtype=np.int64),
        'cont_mean': cont_mean, 'cont_std': cont_std,
        'lag_mean': lag_mean, 'lag_std': lag_std,
    }


# ---------------------------------------------------------------------------
# 6. MLP model + helpers (same as 04b)
# ---------------------------------------------------------------------------
class RetailDataset(Dataset):
    def __init__(self, cat, cont, lags, targets):
        self.cat = torch.from_numpy(cat)
        self.cont = torch.from_numpy(cont)
        self.lags = torch.from_numpy(lags)
        self.targets = torch.from_numpy(targets)
    def __len__(self): return len(self.targets)
    def __getitem__(self, idx):
        return self.cat[idx], self.cont[idx], self.lags[idx], self.targets[idx]


class RetailMLP(nn.Module):
    def __init__(self, n_cont, n_lags):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(CARDINALITIES[name], EMB_DIMS[name])
            for name in EMB_DIMS
        })
        self.emb_names = ['store_id', 'product_id', 'city_id', 'dow']
        total_emb = sum(EMB_DIMS.values())
        input_dim = total_emb + n_cont + n_lags
        layers = []
        prev_dim = input_dim
        for h in MLP_HIDDEN:
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
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LR)
    best_wape, best_epoch, best_state = float('inf'), 0, None
    epochs_no_improve = 0
    val_instock = val_data['stock'] == 0
    val_cat_t = torch.from_numpy(val_data['cat']).to(device)
    val_cont_t = torch.from_numpy(val_data['cont']).to(device)
    val_lags_t = torch.from_numpy(val_data['lags']).to(device)

    for epoch in range(1, MLP_MAX_EPOCHS + 1):
        model.train()
        train_loss, n_b = 0.0, 0
        for cat, cont, lags, targets in train_loader:
            cat, cont, lags, targets = (
                cat.to(device), cont.to(device), lags.to(device), targets.to(device))
            preds = model(cat, cont, lags)
            loss = nn.functional.mse_loss(preds, targets)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss += loss.item(); n_b += 1

        model.eval()
        with torch.no_grad():
            all_p = []
            for s in range(0, len(val_cat_t), 10000):
                e = min(s + 10000, len(val_cat_t))
                all_p.append(model(val_cat_t[s:e], val_cont_t[s:e], val_lags_t[s:e]).cpu().numpy())
            vp = np.concatenate(all_p, axis=0)

        wape = np.abs(vp[val_instock] - val_data['targets'][val_instock]).sum() / \
               np.abs(val_data['targets'][val_instock]).sum()

        print(f'      Epoch {epoch:3d}: loss={train_loss/n_b:.6f}, val_WAPE={wape:.6f}')

        if wape < best_wape:
            best_wape, best_epoch = wape, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= MLP_PATIENCE:
            print(f'      Early stop (best={best_epoch}, WAPE={best_wape:.6f})')
            break

    if best_state: model.load_state_dict(best_state)
    model.to(device)
    return best_wape, best_epoch


def predict_mlp(model, data, device):
    model.eval()
    cat_t = torch.from_numpy(data['cat']).to(device)
    cont_t = torch.from_numpy(data['cont']).to(device)
    lags_t = torch.from_numpy(data['lags']).to(device)
    all_p = []
    with torch.no_grad():
        for s in range(0, len(cat_t), 10000):
            e = min(s + 10000, len(cat_t))
            all_p.append(model(cat_t[s:e], cont_t[s:e], lags_t[s:e]).cpu().numpy())
    return np.concatenate(all_p, axis=0)


# ---------------------------------------------------------------------------
# 7. Evaluation helpers
# ---------------------------------------------------------------------------
def eval_instock_flat(preds_flat, obs_flat, stock_flat, store_ids, product_ids):
    instock = stock_flat == 0
    p_h, o_h = preds_flat[instock], obs_flat[instock]
    sae_h, sao_h = np.abs(p_h - o_h).sum(), np.abs(o_h).sum()
    se_h, so_h = (p_h - o_h).sum(), o_h.sum()

    n_dt = len(preds_flat) // 24
    pd_r, od_r, sk_r = preds_flat.reshape(n_dt, 24), obs_flat.reshape(n_dt, 24), stock_flat.reshape(n_dt, 24)
    sid_d, pid_d = store_ids.reshape(n_dt, 24)[:, 0], product_ids.reshape(n_dt, 24)[:, 0]
    sae_d, sao_d, se_d, so_d = 0., 0., 0., 0.
    for d in range(n_dt):
        m = sk_r[d] == 0
        if m.any():
            pv, ov = pd_r[d, m].sum(), od_r[d, m].sum()
            sae_d += abs(pv - ov); sao_d += abs(ov); se_d += pv - ov; so_d += ov

    pooled = {
        'hourly_wape': sae_h / sao_h if sao_h > 0 else np.nan,
        'hourly_wpe': se_h / so_h if so_h != 0 else np.nan,
        'daily_wape': sae_d / sao_d if sao_d > 0 else np.nan,
        'daily_wpe': se_d / so_d if so_d != 0 else np.nan,
    }

    df_t = pd.DataFrame({'sid': store_ids, 'pid': product_ids, 'day_idx': np.repeat(np.arange(n_dt), 24),
                          'pred': preds_flat.astype(np.float64), 'obs': obs_flat.astype(np.float64), 'stock': stock_flat})
    records = []
    for (sid, pid), grp in df_t.groupby(['sid', 'pid'], sort=False):
        ins = grp['stock'].values == 0
        sao_s = np.abs(grp['obs'].values[ins]).sum()
        sae_s = np.abs(grp['pred'].values[ins] - grp['obs'].values[ins]).sum()
        se_s = (grp['pred'].values[ins] - grp['obs'].values[ins]).sum()
        so_s = grp['obs'].values[ins].sum()
        hw = sae_s / sao_s if sao_s > 0 else np.nan
        hwp = se_s / so_s if so_s != 0 else np.nan
        sad, sod, sed, ssd, nvd = 0., 0., 0., 0., 0
        for di, dg in grp.groupby('day_idx', sort=False):
            dm = dg['stock'].values == 0
            if dm.any():
                pv, ov = dg['pred'].values[dm].sum(), dg['obs'].values[dm].sum()
                sad += abs(pv - ov); sod += abs(ov); sed += pv - ov; ssd += ov; nvd += 1
        records.append({'store_id': sid, 'product_id': pid,
                        'hourly_wape': hw, 'hourly_wpe': hwp,
                        'daily_wape': sad / sod if sod > 0 else np.nan,
                        'daily_wpe': sed / ssd if ssd != 0 else np.nan,
                        'n_hours_instock': int(ins.sum()), 'n_days_valid': nvd})
    return pooled, pd.DataFrame(records)


def eval_instock_day(preds_24, obs_24, stock_24, store_ids, product_ids):
    instock = stock_24 == 0
    p_h, o_h = preds_24[instock], obs_24[instock]
    sae_h, sao_h = np.abs(p_h - o_h).sum(), np.abs(o_h).sum()
    se_h, so_h = (p_h - o_h).sum(), o_h.sum()
    n_s = preds_24.shape[0]
    sae_d, sao_d, se_d, so_d = 0., 0., 0., 0.
    for d in range(n_s):
        m = instock[d]
        if m.any():
            pv, ov = preds_24[d, m].sum(), obs_24[d, m].sum()
            sae_d += abs(pv - ov); sao_d += abs(ov); se_d += pv - ov; so_d += ov
    pooled = {
        'hourly_wape': sae_h / sao_h if sao_h > 0 else np.nan,
        'hourly_wpe': se_h / so_h if so_h != 0 else np.nan,
        'daily_wape': sae_d / sao_d if sao_d > 0 else np.nan,
        'daily_wpe': se_d / so_d if so_d != 0 else np.nan,
    }
    sm = {}
    for i in range(n_s):
        k = (store_ids[i], product_ids[i])
        if k not in sm: sm[k] = []
        sm[k].append(i)
    records = []
    for (sid, pid), idxs in sm.items():
        sash, saoh, seh, soh = 0., 0., 0., 0.
        sasd, saod, sed2, sod2, nvd = 0., 0., 0., 0., 0
        ni = 0
        for i in idxs:
            m = instock[i]; ni += int(m.sum())
            sash += np.abs(preds_24[i, m] - obs_24[i, m]).sum()
            saoh += np.abs(obs_24[i, m]).sum()
            seh += (preds_24[i, m] - obs_24[i, m]).sum()
            soh += obs_24[i, m].sum()
            if m.any():
                pv, ov = preds_24[i, m].sum(), obs_24[i, m].sum()
                sasd += abs(pv - ov); saod += abs(ov); sed2 += pv - ov; sod2 += ov; nvd += 1
        records.append({'store_id': sid, 'product_id': pid,
                        'hourly_wape': sash / saoh if saoh > 0 else np.nan,
                        'hourly_wpe': seh / soh if soh != 0 else np.nan,
                        'daily_wape': sasd / saod if saod > 0 else np.nan,
                        'daily_wpe': sed2 / sod2 if sod2 != 0 else np.nan,
                        'n_hours_instock': ni, 'n_days_valid': nvd})
    return pooled, pd.DataFrame(records)


# ===========================================================================
# 8. Main loop: for each imputer × each forecaster (with lags)
# ===========================================================================
all_results = {}

# Pre-compute _key column once
df_full['_key'] = (df_full['store_id'].astype(str) + '_' +
                   df_full['product_id'].astype(str) + '_' + df_full['dt'])
full_keys = df_full['_key'].values

for imp_key, imp_label in IMPUTERS.items():
    print(f'\n{"="*72}')
    print(f'  IMPUTER: {imp_label} ({imp_key})')
    print(f'{"="*72}')

    # Check if both cells already done
    lgb_done = os.path.exists(os.path.join(RESULTS_DIR, f'{imp_key}__lgb_m5lags_test_per_series.parquet'))
    mlp_done = os.path.exists(os.path.join(RESULTS_DIR, f'{imp_key}__mlp_m5lags_test_per_series.parquet'))

    if lgb_done and mlp_done:
        print(f'  SKIP (both cells already done)')
        for fc, fc_label in [('lgb_m5lags', 'LGB (M5 lags)'), ('mlp_m5lags', 'MLP (M5 lags)')]:
            ck = f'{imp_key}__{fc}'
            ps = pd.read_parquet(os.path.join(RESULTS_DIR, f'{ck}_test_per_series.parquet'))
            med = {c: ps[c].dropna().median() for c in ['hourly_wape', 'hourly_wpe']}
            all_results[ck] = {'pooled': {}, 'median': med,
                                'imputer': imp_label, 'forecaster': fc_label, 'elapsed': 0}
        continue

    # Load completed_sales and build series cache
    cs_path = os.path.join(COMPLETED_DIR, f'{imp_key}.parquet')
    df_cs = pd.read_parquet(cs_path)
    cs_sales = np.array(df_cs['hours_sale'].tolist(), dtype=np.float32)

    completed_full = sales_orig.copy()
    print(f'  Allineamento completed_sales...')
    cs_keys = (df_cs['store_id'].astype(str) + '_' +
               df_cs['product_id'].astype(str) + '_' + df_cs['dt']).values
    key_to_cs_idx = dict(zip(cs_keys, range(len(df_cs))))

    matched = 0
    for i in range(len(df_full)):
        k = full_keys[i]
        if k in key_to_cs_idx:
            completed_full[i] = cs_sales[key_to_cs_idx[k]]
            matched += 1
    print(f'  Matched: {matched:,}/{len(df_full):,}')

    del df_cs, cs_sales, key_to_cs_idx
    gc.collect()

    print(f'  Building series cache...')
    series_cache = build_series_cache(completed_full)
    del completed_full

    # --- LGB (M5 lags) ---
    cell_key_lgb = f'{imp_key}__lgb_m5lags'
    if lgb_done:
        print(f'\n  --- LGB (M5 lags) --- SKIP (already done)')
        ps_lgb = pd.read_parquet(os.path.join(RESULTS_DIR, f'{cell_key_lgb}_test_per_series.parquet'))
        med_lgb = {c: ps_lgb[c].dropna().median() for c in ['hourly_wape', 'hourly_wpe']}
        all_results[cell_key_lgb] = {'pooled': {}, 'median': med_lgb,
                                      'imputer': imp_label, 'forecaster': 'LGB (M5 lags)', 'elapsed': 0}
    else:
        print(f'\n  --- LGB (M5 lags) con {imp_label} ---')
        t0 = time.time()

        print(f'    Building train...')
        X_tr, y_tr, _, _, _ = build_lgb_dataset_completed(series_cache, 'train')
        print(f'    Train: {len(X_tr):,} rows, {X_tr.shape[1]} feat')

        print(f'    Building val...')
        X_va, y_va, stk_va, sid_va, pid_va = build_lgb_dataset_completed(series_cache, 'val')
        print(f'    Val: {len(X_va):,} rows')

        print(f'    Training...')
        lgb_tr = lgb.Dataset(X_tr, y_tr, free_raw_data=True)
        lgb_va = lgb.Dataset(X_va, y_va, reference=lgb_tr, free_raw_data=True)
        model_lgb = lgb.train(LGB_PARAMS, lgb_tr, num_boost_round=LGB_MAX_ROUNDS,
                               valid_sets=[lgb_va], valid_names=['val'],
                               callbacks=[lgb.early_stopping(LGB_EARLY_STOP), lgb.log_evaluation(100)])
        print(f'    Best iter: {model_lgb.best_iteration}')

        del X_tr, y_tr, lgb_tr, lgb_va, X_va
        gc.collect()

        print(f'    Building test...')
        X_te, y_te, stk_te, sid_te, pid_te = build_lgb_dataset_completed(series_cache, 'test')
        preds_lgb = np.clip(model_lgb.predict(X_te), 0, None)
        pooled_lgb, ps_lgb = eval_instock_flat(preds_lgb, y_te, stk_te, sid_te, pid_te)
        med_lgb = {c: ps_lgb[c].dropna().median() for c in ['hourly_wape', 'hourly_wpe']}

        ps_lgb.to_parquet(os.path.join(RESULTS_DIR, f'{cell_key_lgb}_test_per_series.parquet'), index=False)
        all_results[cell_key_lgb] = {'pooled': pooled_lgb, 'median': med_lgb,
                                      'imputer': imp_label, 'forecaster': 'LGB (M5 lags)',
                                      'elapsed': time.time() - t0}

        print(f'    Test: WAPE_h pool={pooled_lgb["hourly_wape"]:.4f}, '
              f'med={med_lgb["hourly_wape"]:.4f}, time={time.time()-t0:.0f}s')

        del X_te, preds_lgb, model_lgb
        gc.collect()

    # --- MLP (M5 lags) ---
    cell_key_mlp = f'{imp_key}__mlp_m5lags'
    if mlp_done:
        print(f'\n  --- MLP (M5 lags) --- SKIP (already done)')
        ps_mlp = pd.read_parquet(os.path.join(RESULTS_DIR, f'{cell_key_mlp}_test_per_series.parquet'))
        med_mlp = {c: ps_mlp[c].dropna().median() for c in ['hourly_wape', 'hourly_wpe']}
        all_results[cell_key_mlp] = {'pooled': {}, 'median': med_mlp,
                                      'imputer': imp_label, 'forecaster': 'MLP (M5 lags)', 'elapsed': 0}
    else:
        print(f'\n  --- MLP (M5 lags) con {imp_label} ---')
        t0 = time.time()

        print(f'    Building train...')
        tr_d = build_mlp_dataset_completed(series_cache, 'train')
        print(f'    Train: {len(tr_d["targets"]):,} samples, lags={tr_d["lags"].shape[1]}')

        print(f'    Building val...')
        va_d = build_mlp_dataset_completed(series_cache, 'val',
                                            cont_mean=tr_d['cont_mean'], cont_std=tr_d['cont_std'],
                                            lag_mean=tr_d['lag_mean'], lag_std=tr_d['lag_std'])

        model_mlp = RetailMLP(tr_d['cont'].shape[1], tr_d['lags'].shape[1]).to(DEVICE)
        tr_ds = RetailDataset(tr_d['cat'], tr_d['cont'], tr_d['lags'], tr_d['targets'])
        tr_loader = DataLoader(tr_ds, batch_size=MLP_BATCH_SIZE, shuffle=True, num_workers=0)

        # Build test BEFORE training (to free series_cache during training)
        print(f'    Building test...')
        te_d = build_mlp_dataset_completed(series_cache, 'test',
                                            cont_mean=tr_d['cont_mean'], cont_std=tr_d['cont_std'],
                                            lag_mean=tr_d['lag_mean'], lag_std=tr_d['lag_std'])
        print(f'    Test:  {len(te_d["targets"]):,} samples')

        # Free series_cache before training to reduce memory pressure
        del series_cache
        gc.collect()

        print(f'    Training...')
        best_wape, best_ep = train_mlp(model_mlp, tr_loader, va_d, DEVICE)

        preds_mlp = predict_mlp(model_mlp, te_d, DEVICE)
        pooled_mlp, ps_mlp = eval_instock_day(
            preds_mlp, te_d['targets'], te_d['stock'], te_d['store_ids'], te_d['product_ids'])
        med_mlp = {c: ps_mlp[c].dropna().median() for c in ['hourly_wape', 'hourly_wpe']}

        ps_mlp.to_parquet(os.path.join(RESULTS_DIR, f'{cell_key_mlp}_test_per_series.parquet'), index=False)
        all_results[cell_key_mlp] = {'pooled': pooled_mlp, 'median': med_mlp,
                                      'imputer': imp_label, 'forecaster': 'MLP (M5 lags)',
                                      'elapsed': time.time() - t0}

        print(f'    Test: WAPE_h pool={pooled_mlp["hourly_wape"]:.4f}, '
              f'med={med_mlp["hourly_wape"]:.4f}, time={time.time()-t0:.0f}s')

        del tr_d, va_d, te_d, model_mlp, tr_ds, tr_loader
        gc.collect()
        if DEVICE == 'mps':
            torch.mps.empty_cache()

    # series_cache may already be deleted (freed before MLP training)
    if 'series_cache' in dir():
        del series_cache
    gc.collect()


# ===========================================================================
# SUMMARY
# ===========================================================================
print('\n' + '=' * 72)
print('  RIEPILOGO — Matrice imputer × forecaster (test, in-stock)')
print('=' * 72)

print(f'\n  {"Imputer":<24} {"Forecaster":<18} {"WAPE_h med":>11} {"WPE_h med":>10}')
print('  ' + '-' * 66)

for key, res in all_results.items():
    m = res['median']
    print(f'  {res["imputer"]:<24} {res["forecaster"]:<18} '
          f'{m["hourly_wape"]:>11.4f} {m["hourly_wpe"]:>10.4f}')

# Save summary
summary_df = pd.DataFrame([
    {'imputer': r['imputer'], 'forecaster': r['forecaster'],
     'wape_h_pool': r['pooled']['hourly_wape'], 'wpe_h_pool': r['pooled']['hourly_wpe'],
     'wape_h_med': r['median']['hourly_wape'], 'wpe_h_med': r['median']['hourly_wpe'],
     'wape_d_pool': r['pooled']['daily_wape'], 'wpe_d_pool': r['pooled']['daily_wpe']}
    for r in all_results.values()
])
summary_df.to_parquet(os.path.join(RESULTS_DIR, 'matrix_b2.parquet'), index=False)
print(f'\n  Salvato: matrix_b2.parquet')

print('\n' + '=' * 72)
print('  DONE — 08_forecast_all_combinations.py')
print('=' * 72)
