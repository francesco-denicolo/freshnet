"""
07b_fase_b_mlp_clean.py — MLP F clean only (retry after hang)
=============================================================
Runs only the MLP F clean part from 07_fase_b_forecasting_clean.py.
Naive and LGB results are already saved from the previous run.

Eseguire con: freshnet/bin/python notebooks/v2/07b_fase_b_mlp_clean.py
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

BATCH_SIZE = 4096
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 10
HIDDEN_SIZES = [128, 64]
EMB_DIMS = {'store_id': 32, 'product_id': 32, 'city_id': 8, 'dow': 4}
CARDINALITIES = {'store_id': 898, 'product_id': 865, 'city_id': 18, 'dow': 7}


print("=" * 72)
print("  MLP F CLEAN — Retry (from 07_fase_b_forecasting_clean.py)")
print("=" * 72)
print(f"  Device: {DEVICE}")

# =========================================================================
# 1. Load data & build series cache (same as 07_fase_b)
# =========================================================================
t0 = time.time()
print("\n1. Caricamento dati...")

df_completed = pd.read_parquet(os.path.join(DATA_DIR, 'completed_sales.parquet'))
df_completed['dt_parsed'] = pd.to_datetime(df_completed['dt'])

df_eval = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_eval.parquet'))
df_eval['dt_parsed'] = pd.to_datetime(df_eval['dt'])

df_train_orig = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df_train_orig['dt_parsed'] = pd.to_datetime(df_train_orig['dt'])

all_dates = sorted(set(df_completed['dt_parsed'].unique()) |
                   set(df_eval['dt_parsed'].unique()))
date_to_day = {d: i + 1 for i, d in enumerate(all_dates)}

df_completed['day_num'] = df_completed['dt_parsed'].map(date_to_day)
df_completed['dow'] = df_completed['dt_parsed'].dt.dayofweek
df_eval['day_num'] = df_eval['dt_parsed'].map(date_to_day)
df_eval['dow'] = df_eval['dt_parsed'].dt.dayofweek
df_train_orig['day_num'] = df_train_orig['dt_parsed'].map(date_to_day)

print(f"  completed_sales: {len(df_completed):,} righe")
print(f"  eval HF: {len(df_eval):,} righe")
print(f"  Tempo loading: {time.time()-t0:.1f}s")

# Parse arrays
csales_arr = np.array(df_completed['hours_completed_sale'].tolist(), dtype=np.float32)
osales_arr = np.array(df_completed['hours_sale_original'].tolist(), dtype=np.float32)
stock_train_arr = np.array(df_completed['hours_stock_status'].tolist(), dtype=np.float32)

eval_sales_arr = np.array(df_eval['hours_sale'].tolist(), dtype=np.float32)
eval_stock_arr = np.array(df_eval['hours_stock_status'].tolist(), dtype=np.float32)

# Build series cache
print("\n2. Building series cache...")
t1 = time.time()

series_data = {}

comp_groups = df_completed.groupby(['store_id', 'product_id'], sort=False)
eval_groups = df_eval.groupby(['store_id', 'product_id'], sort=False)
orig_groups = df_train_orig.groupby(['store_id', 'product_id'], sort=False)

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

    all_days = np.concatenate([days_comp, days_eval])
    all_dows = np.concatenate([dows_comp, dows_eval])
    all_completed = np.concatenate([csales_arr[idx], sales_eval])
    all_original = np.concatenate([osales_arr[idx], sales_eval])
    all_stock = np.concatenate([stock_train_arr[idx], stock_eval])

    oc = orig_cont_cache.get((sid, pid))
    if oc is not None:
        city_id = oc['city_id']
        conts_train = oc['conts']
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
        'completed_sales': all_completed,
        'original_sales': all_original,
        'stock': all_stock,
        'city_id': city_id,
        'conts': conts_all,
    }

del comp_groups, eval_groups, orig_cont_cache
del csales_arr, osales_arr, stock_train_arr, eval_sales_arr, eval_stock_arr
del df_completed, df_eval

print(f"  {len(series_data):,} serie in {time.time()-t1:.0f}s")


# =========================================================================
# 3. MLP F (M5 lags from completed_sales)
# =========================================================================
print("\n" + "=" * 72)
print("  MLP VARIANT F (M5 lags da completed_sales)")
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


# =========================================================================
# 3a. Build MLP datasets
# =========================================================================
print("\n  3a. Building MLP datasets (variant F, clean lags)...")
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

# Free series_data to reduce memory before training
del series_data
import gc; gc.collect()
print("    Freed series_data to reduce memory")

# =========================================================================
# 3b. Train MLP
# =========================================================================
print("\n  3b. Training MLP F (clean lags)...")
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

# =========================================================================
# 3c. Val evaluation (vs original S_obs)
# =========================================================================
print("\n  3c. Val evaluation (vs S_obs)...")
pooled_val_mlp, ps_val_mlp = mlp_evaluate(model, val_data, DEVICE)
ps_val_mlp.to_parquet(os.path.join(RESULTS_DIR, 'clean_mlp_f_val_per_series.parquet'),
                       index=False)
print(f"    WAPE_in pooled: {pooled_val_mlp['wape_instock']:.4f}")
print(f"    WAPE_in median: {ps_val_mlp['wape_instock'].median():.4f}")
print(f"    WPE_in pooled:  {pooled_val_mlp['wpe_instock']:.4f}")

# Free train_data and val_data to reduce memory before retrain
del train_data, train_ds, train_loader
gc.collect()

# =========================================================================
# 3d. Retrain on days 2-90
# =========================================================================
print(f"\n  3d. Retraining on days 2-90 ({best_epoch} epochs)...")
t8 = time.time()

# Need to rebuild series_data for retrain
print("    Rebuilding series_data...")
# Re-load data minimally
df_completed = pd.read_parquet(os.path.join(DATA_DIR, 'completed_sales.parquet'))
df_completed['dt_parsed'] = pd.to_datetime(df_completed['dt'])
df_eval = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_eval.parquet'))
df_eval['dt_parsed'] = pd.to_datetime(df_eval['dt'])
df_train_orig = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df_train_orig['dt_parsed'] = pd.to_datetime(df_train_orig['dt'])

all_dates = sorted(set(df_completed['dt_parsed'].unique()) |
                   set(df_eval['dt_parsed'].unique()))
date_to_day = {d: i + 1 for i, d in enumerate(all_dates)}

df_completed['day_num'] = df_completed['dt_parsed'].map(date_to_day)
df_completed['dow'] = df_completed['dt_parsed'].dt.dayofweek
df_eval['day_num'] = df_eval['dt_parsed'].map(date_to_day)
df_eval['dow'] = df_eval['dt_parsed'].dt.dayofweek
df_train_orig['day_num'] = df_train_orig['dt_parsed'].map(date_to_day)

csales_arr = np.array(df_completed['hours_completed_sale'].tolist(), dtype=np.float32)
osales_arr = np.array(df_completed['hours_sale_original'].tolist(), dtype=np.float32)
stock_train_arr = np.array(df_completed['hours_stock_status'].tolist(), dtype=np.float32)
eval_sales_arr = np.array(df_eval['hours_sale'].tolist(), dtype=np.float32)
eval_stock_arr = np.array(df_eval['hours_stock_status'].tolist(), dtype=np.float32)

series_data = {}
comp_groups = df_completed.groupby(['store_id', 'product_id'], sort=False)
eval_groups = df_eval.groupby(['store_id', 'product_id'], sort=False)
orig_groups = df_train_orig.groupby(['store_id', 'product_id'], sort=False)

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

    eval_grp = eval_groups.get_group((sid, pid)) if (sid, pid) in eval_groups.groups else None
    if eval_grp is not None:
        eval_s = eval_grp.sort_values('day_num')
        eval_idx = eval_s.index.values
        days_eval = eval_s['day_num'].values
        sales_eval = eval_sales_arr[eval_idx]
        stock_eval = eval_stock_arr[eval_idx]
        dows_eval = eval_s['dow'].values
    else:
        days_eval = np.array([], dtype=np.int64)
        dows_eval = np.array([], dtype=np.int64)
        sales_eval = np.zeros((0, 24), dtype=np.float32)
        stock_eval = np.zeros((0, 24), dtype=np.float32)

    all_days = np.concatenate([days_comp, days_eval])
    all_dows = np.concatenate([dows_comp, dows_eval])
    all_completed = np.concatenate([csales_arr[idx], sales_eval])
    all_original = np.concatenate([osales_arr[idx], sales_eval])
    all_stock = np.concatenate([stock_train_arr[idx], stock_eval])

    oc = orig_cont_cache.get((sid, pid))
    if oc is not None:
        city_id = oc['city_id']
        conts_train = oc['conts']
        if eval_grp is not None:
            conts_eval = eval_s[CONT_COLS].values.astype(np.float32)
            conts_all = np.concatenate([conts_train, conts_eval])
        else:
            conts_all = conts_train
    else:
        city_id = 0
        conts_all = np.zeros((len(all_days), len(CONT_COLS)), dtype=np.float32)

    series_data[(sid, pid)] = {
        'days': all_days, 'dows': all_dows,
        'completed_sales': all_completed, 'original_sales': all_original,
        'stock': all_stock, 'city_id': city_id, 'conts': conts_all,
    }

del comp_groups, eval_groups, orig_cont_cache
del csales_arr, osales_arr, stock_train_arr, eval_sales_arr, eval_stock_arr
del df_completed, df_eval
print(f"    {len(series_data):,} serie rebuilt")

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
    print(f"    Epoch {epoch:3d}/{best_epoch}: loss={train_loss/n_batches:.6f}")

print(f"    Retrain time: {time.time()-t8:.0f}s")
torch.save(model_rt.state_dict(), os.path.join(RESULTS_DIR, 'clean_mlp_f_retrained.pt'))

# =========================================================================
# 3e. Test evaluation
# =========================================================================
print("\n  3e. Test evaluation (eval HF)...")
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
# 4. CONFRONTO FINALE
# =========================================================================
print("\n\n" + "=" * 72)
print("  CONFRONTO FINALE (TEST) — Fase A vs Fase B")
print("=" * 72)

print(f"\n  {'Modello':<25} {'WAPE_in_pool':>14} {'WAPE_in_med':>14} "
      f"{'WPE_in_pool':>14} {'WPE_in_med':>14} {'WAPE_all_med':>14}")
print("  " + "-" * 100)

# Load all saved results
all_models = {
    # Fase A
    'Global Mean (A)': 'global_mean',
    'DoW Mean (A)': 'dow_mean',
    'Naive Direct (A)': 'naive_direct',
    'MA Direct (A)': 'ma_direct',
    'LGB F (A)': 'lgb_f',
    'MLP F (A)': 'mlp_f',
    # Fase B naive
    'Global Mean (B)': 'clean_global_mean',
    'DoW Mean (B)': 'clean_dow_mean',
    'Naive Direct (B)': 'clean_naive_direct',
    'MA Direct (B)': 'clean_ma_direct',
    # Fase B ML
    'LGB F (B)': 'clean_lgb_f',
}

for label, prefix in all_models.items():
    path = os.path.join(RESULTS_DIR, f'{prefix}_test_per_series.parquet')
    if os.path.exists(path):
        ps = pd.read_parquet(path)
        wape_in_m = ps['wape_instock'].median() if 'wape_instock' in ps else np.nan
        wpe_in_m = ps['wpe_instock'].median() if 'wpe_instock' in ps else np.nan
        wape_all_m = ps['wape_overall'].median() if 'wape_overall' in ps else np.nan
        print(f"  {label:<25} {'—':>14} {wape_in_m:>14.4f} "
              f"{'—':>14} {wpe_in_m:>14.4f} {wape_all_m:>14.4f}")

# MLP F (B) - from current run
print(f"  {'MLP F (B)':<25} {pooled_test_mlp['wape_instock']:>14.4f} "
      f"{ps_test_mlp['wape_instock'].median():>14.4f} "
      f"{pooled_test_mlp['wpe_instock']:>14.4f} "
      f"{ps_test_mlp['wpe_instock'].median():>14.4f} "
      f"{ps_test_mlp['wape_overall'].median():>14.4f}")

print(f"\n  Tempo totale: {time.time()-t0:.0f}s")
print("\n  DONE!")
