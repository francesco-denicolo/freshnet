"""
08_baseline_mlp.py — MLP Baseline (Direct Forecast)
=====================================================
PINN-Retail: Physics-Informed Neural Networks per Demand Forecasting
di Prodotti Deperibili.

Passo 3g: MLP baseline su vendite osservate (censurate).
Modello globale: un unico MLP per tutte le 50K serie.
Output: 24 vendite orarie predette per (store, product, giorno).
Loss: MSE su tutte le 24 ore (incluse ore di stockout — ignora il censoring).

Lag variants selezionate su validation:
  A: nessuno storico
  B: ultimo giorno disponibile (24 valori)
  C: media ultimi 7gg (24 valori)
  D: ultimo stesso DoW (24 valori)
  E: B + C + D combinati (72 valori)

Eseguire con: freshnet/bin/python notebooks/08_baseline_mlp.py
"""

import sys
import os
import time
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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

# Hyperparameters
BATCH_SIZE = 4096
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 10
HIDDEN_SIZES = [128, 64]

# Embedding dimensions
EMB_DIMS = {
    'store_id': 32,
    'product_id': 32,
    'city_id': 8,
    'dow': 4,
}

# Cardinalities (0-indexed, checked from data)
CARDINALITIES = {
    'store_id': 898,
    'product_id': 865,
    'city_id': 18,
    'dow': 7,
}

CONT_FEATURES = ['discount', 'avg_temperature', 'avg_humidity',
                  'precpt', 'avg_wind_level', 'holiday_flag', 'activity_flag']

LAG_VARIANTS = {
    'F': 'M5-style full (264+11mask=275)',
}

# Variant A results from previous run (reuse, don't recompute)
VARIANT_A_CACHED = {
    'wape_pooled': 0.978500,
    'wape_median': 1.183500,
    'best_epoch': 30,
    'elapsed': 0,
    'n_params': 12120,
}

LAG_FEATURES_F_NAMES = [
    'lag_1d', 'lag_7d', 'lag_14d',
    'rmean_7d', 'rmean_14d', 'rstd_7d',
    'lag_dow', 'rmean_dow',
    'daily_total_lag1', 'daily_total_rmean7',
    'momentum_1d_7d',
]

# ===========================================================================
print('=' * 72)
print('  MLP BASELINE — DIRECT FORECAST')
print('=' * 72)

# ---------------------------------------------------------------------------
# 1. Caricamento e preparazione dati
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

# ---------------------------------------------------------------------------
# 2. Build feature arrays per series
# ---------------------------------------------------------------------------
print('\n2. Preparazione feature arrays per serie...')

# Pre-compute per-series data for lag computation
# Store all series data in a dict keyed by (store_id, product_id)
series_data = {}
groups = df_full.groupby(['store_id', 'product_id'], sort=False)
n_groups = len(groups)

for i, ((sid, pid), grp) in enumerate(groups):
    if (i + 1) % 10000 == 0:
        print(f'    ... {i+1:,}/{n_groups:,} serie')
    grp_s = grp.sort_values('day_num')
    series_data[(sid, pid)] = {
        'days': grp_s['day_num'].values,
        'dows': grp_s['dow'].values,
        'sales': np.array(grp_s['hours_sale'].tolist(), dtype=np.float32),  # (N, 24)
        'stock': np.array(grp_s['hours_stock_status'].tolist(), dtype=np.float32),
        'city_id': grp_s['city_id'].values[0],
        'conts': grp_s[CONT_FEATURES].values.astype(np.float32),  # (N, 7)
    }

print(f'  {len(series_data):,} serie preparate')


def compute_lag_features(sales, days, dows, target_day_idx, anchor_day, variant):
    """Compute lag features for a single row.

    For direct forecast, all days in the forecast horizon use the same anchor.
    - Val: anchor_day=83 (last training day)
    - Test: anchor_day=90 (last training+val day)

    Args:
        sales: (N, 24) array of hourly sales for this series
        days: (N,) day numbers
        dows: (N,) day of week values
        target_day_idx: index of the target day in this series' arrays
        anchor_day: last day available before forecast horizon
        variant: 'A', 'B', 'C', 'D', or 'E'

    Returns:
        np.array of lag features (length depends on variant)
    """
    if variant == 'A':
        return np.array([], dtype=np.float32)

    # Days available for lags: up to anchor_day
    avail_mask = days <= anchor_day
    avail_days = days[avail_mask]
    avail_sales = sales[avail_mask]
    avail_dows = dows[avail_mask]

    target_dow = dows[target_day_idx]

    lag_b = np.zeros(24, dtype=np.float32)  # last day
    lag_c = np.zeros(24, dtype=np.float32)  # mean last 7 days
    lag_d = np.zeros(24, dtype=np.float32)  # last same DoW

    if len(avail_days) > 0:
        # B: last available day
        lag_b = avail_sales[-1].copy()

        # C: mean of last min(7, available) days
        n_c = min(7, len(avail_days))
        lag_c = avail_sales[-n_c:].mean(axis=0)

        # D: last day with same DoW
        same_dow_mask = avail_dows == target_dow
        if same_dow_mask.any():
            same_dow_sales = avail_sales[same_dow_mask]
            lag_d = same_dow_sales[-1].copy()
        else:
            lag_d = lag_c.copy()  # fallback to mean

    if variant == 'B':
        return lag_b
    elif variant == 'C':
        return lag_c
    elif variant == 'D':
        return lag_d
    elif variant == 'E':
        return np.concatenate([lag_b, lag_c, lag_d])
    elif variant == 'F':
        return _compute_lag_features_f(sales, days, dows, target_day_idx, anchor_day)

    return np.array([], dtype=np.float32)


def _compute_lag_features_f(sales, days, dows, target_day_idx, anchor_day):
    """Compute M5-style lag features (11 × 24 values + 11 binary masks = 275).

    Returns np.array of 275 float32 values:
      [lag_1d(24), lag_7d(24), lag_14d(24), rmean_7d(24), rmean_14d(24),
       rstd_7d(24), lag_dow(24), rmean_dow(24), daily_total_lag1(24),
       daily_total_rmean7(24), momentum_1d_7d(24),
       mask_lag1d(1), mask_lag7d(1), mask_lag14d(1), mask_rmean7d(1),
       mask_rmean14d(1), mask_rstd7d(1), mask_lagdow(1), mask_rmeandow(1),
       mask_dtlag1(1), mask_dtrmean7(1), mask_momentum(1)]
    """
    z = np.float32

    # Initialize 11 features with 0 (not NaN — MLP can't handle NaN)
    feat_lag_1d = np.zeros(24, dtype=z)
    feat_lag_7d = np.zeros(24, dtype=z)
    feat_lag_14d = np.zeros(24, dtype=z)
    feat_rmean_7d = np.zeros(24, dtype=z)
    feat_rmean_14d = np.zeros(24, dtype=z)
    feat_rstd_7d = np.zeros(24, dtype=z)
    feat_lag_dow = np.zeros(24, dtype=z)
    feat_rmean_dow = np.zeros(24, dtype=z)
    feat_dt_lag1 = np.zeros(24, dtype=z)
    feat_dt_rmean7 = np.zeros(24, dtype=z)
    feat_momentum = np.zeros(24, dtype=z)

    # 11 binary masks (1=available, 0=missing)
    masks = np.zeros(11, dtype=z)

    # Available history
    avail_mask = days <= anchor_day
    K = int(avail_mask.sum())

    if K > 0:
        avail_sales = sales[avail_mask]   # (K, 24)
        avail_dows = dows[avail_mask]     # (K,)
        target_dow = dows[target_day_idx]

        # --- Raw lags ---
        feat_lag_1d[:] = avail_sales[-1]
        masks[0] = 1.0  # lag_1d available

        if K >= 7:
            feat_lag_7d[:] = avail_sales[-7]
            masks[1] = 1.0

        if K >= 14:
            feat_lag_14d[:] = avail_sales[-14]
            masks[2] = 1.0

        # --- Rolling means ---
        if K >= 7:
            feat_rmean_7d[:] = avail_sales[-7:].mean(axis=0)
            masks[3] = 1.0

        if K >= 14:
            feat_rmean_14d[:] = avail_sales[-14:].mean(axis=0)
            masks[4] = 1.0

        # --- Rolling std ---
        if K >= 2:
            w = min(7, K)
            feat_rstd_7d[:] = avail_sales[-w:].std(axis=0)
            masks[5] = 1.0

        # --- DoW-specific ---
        same_dow = avail_dows == target_dow
        if same_dow.any():
            dow_sales = avail_sales[same_dow]
            feat_lag_dow[:] = dow_sales[-1]
            feat_rmean_dow[:] = dow_sales.mean(axis=0)
            masks[6] = 1.0  # lag_dow
            masks[7] = 1.0  # rmean_dow

        # --- Daily aggregates ---
        daily_totals = avail_sales.sum(axis=1)  # (K,)
        feat_dt_lag1[:] = daily_totals[-1]
        masks[8] = 1.0

        if K >= 7:
            feat_dt_rmean7[:] = daily_totals[-7:].mean()
            masks[9] = 1.0

        # --- Momentum: lag_1d / rmean_7d ---
        if masks[3] == 1.0:  # rmean_7d available
            rm7 = feat_rmean_7d
            valid_h = rm7 > 0
            if valid_h.any():
                feat_momentum[valid_h] = feat_lag_1d[valid_h] / rm7[valid_h]
                masks[10] = 1.0

    return np.concatenate([
        feat_lag_1d, feat_lag_7d, feat_lag_14d,
        feat_rmean_7d, feat_rmean_14d, feat_rstd_7d,
        feat_lag_dow, feat_rmean_dow,
        feat_dt_lag1, feat_dt_rmean7, feat_momentum,
        masks,
    ])


def build_dataset_arrays(series_data, split, variant,
                         cont_mean=None, cont_std=None,
                         lag_mean=None, lag_std=None):
    """Build arrays for a given split and lag variant.

    Args:
        series_data: dict of per-series data
        split: 'train', 'val', or 'test'
        variant: lag variant 'A'-'F'
        cont_mean, cont_std: normalization params (computed from train if None)
        lag_mean, lag_std: lag normalization params (computed from train if None)

    Returns:
        dict with arrays: cat_feats, cont_feats, lag_feats, targets, stock,
                          store_ids, product_ids, cont_mean, cont_std,
                          lag_mean, lag_std
    """
    if split == 'train':
        d_min, d_max = 2, 83
        anchor_day = 83  # for lags, use all available training data
    elif split == 'val':
        d_min, d_max = 84, 90
        anchor_day = 83
    elif split == 'test':
        d_min, d_max = 91, 97
        anchor_day = 90

    cat_list = []    # (store_id, product_id, city_id, dow)
    cont_list = []   # continuous features
    lag_list = []    # lag features
    target_list = [] # hours_sale (24,)
    stock_list = []  # hours_stock_status (24,)
    sid_list = []
    pid_list = []

    for (sid, pid), sd in series_data.items():
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

            # For training split, use a rolling anchor: day before target
            # For val/test, use fixed anchor (last day before horizon)
            if split == 'train':
                a_day = d - 1  # can use data up to yesterday
            else:
                a_day = anchor_day

            cat_list.append([sid, pid, city, dows[idx]])
            cont_list.append(conts[idx])
            lag_list.append(compute_lag_features(sales, days, dows, idx, a_day, variant))
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

    # Normalize continuous features
    if cont_mean is None:
        cont_mean = cont_arr.mean(axis=0)
        cont_std = cont_arr.std(axis=0)
        cont_std[cont_std < 1e-8] = 1.0

    cont_arr = (cont_arr - cont_mean) / cont_std

    # Normalize lag features (important: daily_total ~24x larger, momentum ~ratio)
    if lag_arr.shape[1] > 0:
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
        'store_ids': np.array(sid_list, dtype=np.int64),
        'product_ids': np.array(pid_list, dtype=np.int64),
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


# ---------------------------------------------------------------------------
# 3. MLP Model
# ---------------------------------------------------------------------------
class RetailMLP(nn.Module):
    def __init__(self, n_cont, n_lags, emb_dims, cardinalities, hidden_sizes):
        super().__init__()

        # Embedding layers
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(cardinalities[name], emb_dims[name])
            for name in emb_dims
        })
        self.emb_names = ['store_id', 'product_id', 'city_id', 'dow']

        total_emb = sum(emb_dims.values())
        input_dim = total_emb + n_cont + n_lags

        # MLP layers
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
        # cat columns: [store_id, product_id, city_id, dow]
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
        # Train
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

        # Validate: compute WAPE pooled
        model.eval()
        with torch.no_grad():
            val_cat = torch.from_numpy(val_data['cat']).to(device)
            val_cont = torch.from_numpy(val_data['cont']).to(device)
            val_lags = torch.from_numpy(val_data['lags']).to(device)

            # Process in chunks to avoid OOM
            chunk_size = 10000
            all_preds = []
            for start in range(0, len(val_cat), chunk_size):
                end = min(start + chunk_size, len(val_cat))
                p = model(val_cat[start:end], val_cont[start:end], val_lags[start:end])
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
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
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
    """Generate predictions for a dataset."""
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
# 4. Variant selection loop
# ---------------------------------------------------------------------------
print('\n3. Selezione variante lag su validation...')
print(f'   Varianti: {LAG_VARIANTS}')
print(f'   Device: {DEVICE}\n')

variant_results = {}

for variant, desc in LAG_VARIANTS.items():
    print(f'  --- Variante {variant}: {desc} ---')
    t0 = time.time()

    # Build datasets
    train_data = build_dataset_arrays(series_data, 'train', variant)
    val_data = build_dataset_arrays(series_data, 'val', variant,
                                     cont_mean=train_data['cont_mean'],
                                     cont_std=train_data['cont_std'],
                                     lag_mean=train_data['lag_mean'],
                                     lag_std=train_data['lag_std'])

    n_cont = train_data['cont'].shape[1]
    n_lags = train_data['lags'].shape[1]

    print(f'    Train: {len(train_data["targets"]):,} samples, '
          f'cont={n_cont}, lags={n_lags}')
    print(f'    Val:   {len(val_data["targets"]):,} samples')

    # Build model
    torch.manual_seed(SEED)
    model = RetailMLP(n_cont, n_lags, EMB_DIMS, CARDINALITIES, HIDDEN_SIZES)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'    Model params: {n_params:,}')

    # DataLoader
    train_ds = RetailDataset(train_data['cat'], train_data['cont'],
                              train_data['lags'], train_data['targets'])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=0, pin_memory=False)

    # Train
    best_wape, best_epoch = train_model(
        model, train_loader, val_data, DEVICE, LR, MAX_EPOCHS, PATIENCE)

    elapsed = time.time() - t0

    # Also compute val WAPE median per-series (efficient groupby)
    val_preds = predict(model, val_data, DEVICE)
    sids = val_data['store_ids']
    pids = val_data['product_ids']

    df_tmp = pd.DataFrame({
        'sid': sids, 'pid': pids,
        'abs_err': np.abs(val_preds - val_data['targets']).sum(axis=1),
        'abs_obs': np.abs(val_data['targets']).sum(axis=1),
    })
    grp = df_tmp.groupby(['sid', 'pid'])[['abs_err', 'abs_obs']].sum()
    grp_valid = grp[grp['abs_obs'] > 0]
    ps_wapes = (grp_valid['abs_err'] / grp_valid['abs_obs']).values
    med_wape = np.median(ps_wapes) if len(ps_wapes) > 0 else np.nan

    variant_results[variant] = {
        'wape_pooled': best_wape,
        'wape_median': med_wape,
        'best_epoch': best_epoch,
        'elapsed': elapsed,
        'n_params': n_params,
        'cont_mean': train_data['cont_mean'],
        'cont_std': train_data['cont_std'],
    }

    print(f'    Val WAPE pooled: {best_wape:.6f}, median: {med_wape:.6f}, '
          f'time: {elapsed:.1f}s\n')

    # Save model state for potential reuse
    torch.save(model.state_dict(),
               os.path.join(RESULTS_DIR, f'mlp_variant_{variant}.pt'))

    del model, train_ds, train_loader, train_data, val_data, val_preds
    torch.mps.empty_cache() if DEVICE == 'mps' else None

# ---------------------------------------------------------------------------
# 5. Variant selection table (include cached variant A)
# ---------------------------------------------------------------------------
# Inject cached variant A results
variant_results['A'] = VARIANT_A_CACHED.copy()
ALL_VARIANTS = {'A': 'No history (cached)', **LAG_VARIANTS}

print('\n' + '=' * 72)
print('  4. SELEZIONE VARIANTE LAG')
print('=' * 72)

print(f'\n  {"Var":<4} {"Description":<30} {"WAPE_pool":>10} {"WAPE_med":>10} '
      f'{"Epoch":>6} {"Time":>8} {"Params":>10}')
print('  ' + '-' * 82)

for v in ['A', 'F']:
    r = variant_results[v]
    print(f'  {v:<4} {ALL_VARIANTS[v]:<30} {r["wape_pooled"]:>10.6f} '
          f'{r["wape_median"]:>10.6f} {r["best_epoch"]:>6d} '
          f'{r["elapsed"]:>7.1f}s {r["n_params"]:>10,}')

# Select best variant (only among computed variants, A is cached reference)
best_var_pooled = min(variant_results, key=lambda v: variant_results[v]['wape_pooled'])
best_var_median = min(variant_results, key=lambda v: variant_results[v]['wape_median'])

print(f'\n  Best (WAPE pooled):  variant {best_var_pooled} '
      f'({variant_results[best_var_pooled]["wape_pooled"]:.6f})')
print(f'  Best (WAPE median):  variant {best_var_median} '
      f'({variant_results[best_var_median]["wape_median"]:.6f})')

if best_var_pooled == best_var_median:
    print(f'\n  Entrambi i criteri concordano: variante {best_var_pooled}')
else:
    print(f'\n  Criteri discordanti: pooled→{best_var_pooled}, median→{best_var_median}')

# Always retrain/evaluate F (A is already cached from previous run)
BEST_VAR = 'F'
print(f'  Retraining variante F per valutazione completa.')

# ---------------------------------------------------------------------------
# 6. Retrain best variant and full evaluation
# ---------------------------------------------------------------------------
print(f'\n5. Retraining variante {BEST_VAR} e valutazione completa...')

# Rebuild datasets with best variant
train_data = build_dataset_arrays(series_data, 'train', BEST_VAR)
val_data = build_dataset_arrays(series_data, 'val', BEST_VAR,
                                 cont_mean=train_data['cont_mean'],
                                 cont_std=train_data['cont_std'],
                                 lag_mean=train_data['lag_mean'],
                                 lag_std=train_data['lag_std'])
test_data = build_dataset_arrays(series_data, 'test', BEST_VAR,
                                  cont_mean=train_data['cont_mean'],
                                  cont_std=train_data['cont_std'],
                                  lag_mean=train_data['lag_mean'],
                                  lag_std=train_data['lag_std'])

n_cont = train_data['cont'].shape[1]
n_lags = train_data['lags'].shape[1]

# Retrain (same seed for reproducibility)
torch.manual_seed(SEED)
model = RetailMLP(n_cont, n_lags, EMB_DIMS, CARDINALITIES, HIDDEN_SIZES)

train_ds = RetailDataset(train_data['cat'], train_data['cont'],
                          train_data['lags'], train_data['targets'])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=0, pin_memory=False)

best_wape, best_epoch = train_model(
    model, train_loader, val_data, DEVICE, LR, MAX_EPOCHS, PATIENCE)

print(f'  Best epoch: {best_epoch}, val WAPE: {best_wape:.6f}')

# ---------------------------------------------------------------------------
# 7. Evaluation on all splits
# ---------------------------------------------------------------------------
print(f'\n6. Valutazione su tutti gli split...')

pooled_results = {}
per_series_dfs = {}

for split_name, data in [('val', val_data), ('test', test_data)]:
    preds = predict(model, data, DEVICE)
    obs = data['targets']
    stock = data['stock']
    sids = data['store_ids']
    pids = data['product_ids']

    # Pooled metrics
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

    # Per-series metrics (efficient: build index once, iterate groups)
    df_idx = pd.DataFrame({'sid': sids, 'pid': pids, 'row': np.arange(len(sids))})
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
                            f'mlp_f_{split_name}_per_series.parquet')
    ps.to_parquet(out_path, index=False)
    print(f'  Salvato: {out_path} ({len(ps):,} serie)')

# ---------------------------------------------------------------------------
# 8. Tabella risultati pooled
# ---------------------------------------------------------------------------
print(format_metrics_table(pooled_results,
                            model_name=f'MLP Baseline (variant {BEST_VAR})'))

# ---------------------------------------------------------------------------
# 9. Distribuzione per-serie
# ---------------------------------------------------------------------------
METRIC_COLS = ['wape_overall', 'wape_instock', 'wape_stockout',
               'wpe_overall', 'wpe_instock', 'wpe_stockout']
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]

print('\n' + '=' * 72)
print('  7. DISTRIBUZIONE METRICHE PER-SERIE')
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
# 10. Confronto con tutti i baseline
# ---------------------------------------------------------------------------
print('\n' + '=' * 72)
print('  8. CONFRONTO CON TUTTI I BASELINE (test)')
print('=' * 72)

all_baselines = {
    'Naive (direct)': 'naive_direct',
    'MA K=14 (direct)': 'ma_direct_K14',
    'Global Mean': 'global_mean',
    'DoW Mean': 'dow_mean',
    'LGB (var A)': 'lgb_a',
    'LGB (var F)': 'lgb_f',
    '2-Stage (LGB)': 'twostage_lgb',
    'MLP (var A)': 'mlp',
    f'MLP (var F)': 'mlp_f',
}

print(f'\n  {"Model":<24} {"WAPE_pool":>10} {"WAPE_in_med":>12} '
      f'{"WAPE_all_med":>13} {"WPE_pool":>10}')
print('  ' + '-' * 73)

for label, prefix in all_baselines.items():
    path = os.path.join(RESULTS_DIR, f'{prefix}_test_per_series.parquet')
    if not os.path.exists(path):
        continue
    ps = pd.read_parquet(path)
    med_all = ps['wape_overall'].median()
    med_in = ps['wape_instock'].median() if 'wape_instock' in ps.columns else np.nan

    # For MLP-F, use computed pooled results
    if prefix == 'mlp_f' and 'test' in pooled_results:
        wp = pooled_results['test']['wape_overall']
        wpe = pooled_results['test']['wpe_overall']
    else:
        wp = np.nan
        wpe = np.nan

    print(f'  {label:<24} {wp:>10.4f} {med_in:>12.4f} '
          f'{med_all:>13.4f} {wpe:>10.4f}')

# ---------------------------------------------------------------------------
# 11. Figure
# ---------------------------------------------------------------------------
print('\n9. Generazione figure...')

# Fig 30: Histograms WAPE/WPE per split
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(f'MLP Baseline (variant {BEST_VAR}) — Distribuzione per-serie', fontsize=14)

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
fig.savefig(os.path.join(FIG_DIR, 'fig30_mlp_per_series_distributions.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Salvata: fig30_mlp_per_series_distributions.png')

# Fig 31: Variant selection bar chart (A vs F)
fig, ax = plt.subplots(figsize=(8, 5))
variants = ['A', 'F']
wape_pooled = [variant_results[v]['wape_pooled'] for v in variants]
wape_median = [variant_results[v]['wape_median'] for v in variants]
x = np.arange(len(variants))
w = 0.35

bars1 = ax.bar(x - w/2, wape_pooled, w, label='WAPE pooled (val)', color='steelblue', alpha=0.8)
bars2 = ax.bar(x + w/2, wape_median, w, label='WAPE median (val)', color='darkorange', alpha=0.8)

ax.set_xlabel('Lag variant')
ax.set_ylabel('WAPE on validation')
ax.set_title('MLP — Lag Variant Selection (A vs F)')
ax.set_xticks(x)
ax.set_xticklabels([f'{v}: {ALL_VARIANTS[v]}' for v in variants], rotation=30, ha='right')
ax.legend()

# Annotate best
best_idx = variants.index(BEST_VAR) if BEST_VAR in variants else 0
ax.annotate(f'Best: {BEST_VAR}', xy=(best_idx, wape_pooled[best_idx]),
            xytext=(best_idx + 0.3, wape_pooled[best_idx] + 0.02),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, color='red', fontweight='bold')

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig31_mlp_variant_selection.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Salvata: fig31_mlp_variant_selection.png')

# Save best model
torch.save(model.state_dict(),
           os.path.join(RESULTS_DIR, f'mlp_best_variant_{BEST_VAR}.pt'))

print(f'\n  Variante migliore: {BEST_VAR} ({LAG_VARIANTS[BEST_VAR]})')
print(f'  Best epoch: {best_epoch}')
print(f'  Val WAPE pooled: {best_wape:.6f}')

print('\n' + '=' * 72)
print('  DONE')
print('=' * 72)
