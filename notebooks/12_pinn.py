"""
12_pinn.py — PINN-Retail: Physics-Informed Neural Network
==========================================================
PINN-Retail: Physics-Informed Neural Networks per Demand Forecasting
di Prodotti Deperibili.

Passo 5: PINN end-to-end (nessuna imputation).
Architettura: encoder MLP condiviso → due teste (D* domanda, I* inventario).
Loss: L_data (MSE in-stock) + L_boundary + L_cons (vincoli fisici).
Ottimizzazione: Augmented Lagrangian Method (ALM) con dual updates.

Predizione: output = D* (domanda latente). I* usato solo nel training.
stock_status NON è input del network — usato solo nella loss.

Eseguire con: freshnet/bin/python notebooks/12_pinn.py
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
import torch.nn.functional as F
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

# MLP Hyperparameters
BATCH_SIZE = 4096
LR = 1e-3
HIDDEN_SIZES = [128, 64]

EMB_DIMS = {'store_id': 32, 'product_id': 32, 'city_id': 8, 'dow': 4}
CARDINALITIES = {'store_id': 898, 'product_id': 865, 'city_id': 18, 'dow': 7}

CONT_FEATURES = ['discount', 'avg_temperature', 'avg_humidity',
                  'precpt', 'avg_wind_level', 'holiday_flag', 'activity_flag']

# ALM Hyperparameters
WARMUP_EPOCHS = 3
K_INNER = 3          # epochs per ALM iteration
N_OUTER = 15         # max ALM iterations
ALM_PATIENCE = 5     # early stopping patience (ALM iterations)
RHO_INIT = 1.0
GAMMA = 2.0          # rho multiplier when violation doesn't improve


# ===========================================================================
print('=' * 72)
print('  PINN-RETAIL: PHYSICS-INFORMED NEURAL NETWORK')
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

del df_full, sales_all, stock_all
gc.collect()


# ---------------------------------------------------------------------------
# 3. Vectorized lag computation (M5-style, from S_obs)
# ---------------------------------------------------------------------------
# Layout dei 275 valori per riga:
#   [0:24]   lag_1d      [24:48]  lag_7d     [48:72]  lag_14d
#   [72:96]  rmean_7d    [96:120] rmean_14d  [120:144] rstd_7d
#   [144:168] lag_dow    [168:192] rmean_dow
#   [192:216] dt_lag1    [216:240] dt_rmean7 [240:264] momentum
#   [264:275] masks (11 binary flags)

def _compute_series_lags_train(sales, dows, target_indices):
    """Vectorized M5-style lag features for train split (rolling anchor d-1)."""
    M = len(target_indices)
    z = np.float32
    out = np.zeros((M, 275), dtype=z)
    if M == 0:
        return out

    ti = target_indices
    N = len(sales)

    daily_totals = sales.sum(axis=1)
    cs = np.zeros((N + 1, 24), dtype=np.float64)
    cs[1:] = np.cumsum(sales.astype(np.float64), axis=0)
    cs_sq = np.zeros((N + 1, 24), dtype=np.float64)
    cs_sq[1:] = np.cumsum((sales.astype(np.float64)) ** 2, axis=0)
    cs_dt = np.zeros(N + 1, dtype=np.float64)
    cs_dt[1:] = np.cumsum(daily_totals.astype(np.float64))

    v1 = ti >= 1
    if v1.any():
        out[v1, 0:24] = sales[ti[v1] - 1]
        out[v1, 264] = 1.0

    v7 = ti >= 7
    if v7.any():
        out[v7, 24:48] = sales[ti[v7] - 7]
        out[v7, 265] = 1.0

    v14 = ti >= 14
    if v14.any():
        out[v14, 48:72] = sales[ti[v14] - 14]
        out[v14, 266] = 1.0

    if v7.any():
        t7 = ti[v7]
        out[v7, 72:96] = ((cs[t7] - cs[t7 - 7]) / 7.0).astype(z)
        out[v7, 267] = 1.0

    if v14.any():
        t14 = ti[v14]
        out[v14, 96:120] = ((cs[t14] - cs[t14 - 14]) / 14.0).astype(z)
        out[v14, 268] = 1.0

    if v7.any():
        t7 = ti[v7]
        mean_7 = (cs[t7] - cs[t7 - 7]) / 7.0
        mean_sq_7 = (cs_sq[t7] - cs_sq[t7 - 7]) / 7.0
        var_7 = np.maximum(mean_sq_7 - mean_7 ** 2, 0)
        out[v7, 120:144] = np.sqrt(var_7).astype(z)
        out[v7, 269] = 1.0

    v2_not7 = (ti >= 2) & (ti < 7)
    if v2_not7.any():
        for row in np.where(v2_not7)[0]:
            out[row, 120:144] = sales[:ti[row]].std(axis=0)
            out[row, 269] = 1.0

    if v1.any():
        out[v1, 192:216] = daily_totals[ti[v1] - 1, np.newaxis]
        out[v1, 272] = 1.0

    if v7.any():
        t7 = ti[v7]
        dt_rm7 = ((cs_dt[t7] - cs_dt[t7 - 7]) / 7.0).astype(z)
        out[v7, 216:240] = dt_rm7[:, np.newaxis]
        out[v7, 273] = 1.0

    target_dows = dows[ti]
    for d in range(7):
        d_mask_t = target_dows == d
        if not d_mask_t.any():
            continue
        dow_d_idx = np.where(dows == d)[0]
        if len(dow_d_idx) == 0:
            continue
        dow_d_sales = sales[dow_d_idx]
        dow_d_cs = np.zeros((len(dow_d_idx) + 1, 24), dtype=np.float64)
        dow_d_cs[1:] = np.cumsum(dow_d_sales.astype(np.float64), axis=0)

        ti_d = ti[d_mask_t]
        count = np.searchsorted(dow_d_idx, ti_d, side='left')
        has_prev = count > 0
        if has_prev.any():
            c = count[has_prev]
            rows = np.where(d_mask_t)[0][has_prev]
            out[rows, 144:168] = dow_d_sales[c - 1]
            out[rows, 168:192] = (dow_d_cs[c] / c[:, np.newaxis]).astype(z)
            out[rows, 270] = 1.0
            out[rows, 271] = 1.0

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
    """Vectorized lag features for val/test with fixed anchor."""
    M = len(target_indices)
    z = np.float32
    out = np.zeros((M, 275), dtype=z)
    if M == 0:
        return out

    avail_mask = days <= anchor_day
    K = int(avail_mask.sum())
    if K == 0:
        return out

    avail = sales[avail_mask]
    avail_dows = dows[avail_mask]

    out[:, 0:24] = avail[-1]
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
    out[:, 192:216] = dt[-1]
    out[:, 272] = 1.0

    if K >= 7:
        out[:, 216:240] = dt[-7:].mean()
        out[:, 273] = 1.0

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


# ---------------------------------------------------------------------------
# 4. Build dataset arrays (variant F, lag from S_obs)
# ---------------------------------------------------------------------------
def build_dataset_arrays(sdata, split,
                         cont_mean=None, cont_std=None,
                         lag_mean=None, lag_std=None):
    """Build arrays for PINN — variant F lags from S_obs, includes stock."""
    if split == 'train':
        d_min, d_max = 2, 83
    elif split == 'val':
        d_min, d_max = 84, 90
    elif split == 'test':
        d_min, d_max = 91, 97

    # Phase 1: count rows
    series_info = []
    total_rows = 0
    for (sid, pid), sd in sdata.items():
        tmask = (sd['days'] >= d_min) & (sd['days'] <= d_max)
        n_t = int(tmask.sum())
        if n_t > 0:
            series_info.append((sid, pid, sd, tmask, n_t))
            total_rows += n_t

    print(f'    Pre-allocating {total_rows:,} rows...')

    # Phase 2: pre-allocate
    cat_arr = np.empty((total_rows, 4), dtype=np.int64)
    cont_arr = np.empty((total_rows, len(CONT_FEATURES)), dtype=np.float32)
    lag_arr = np.empty((total_rows, 275), dtype=np.float32)
    target_arr = np.empty((total_rows, 24), dtype=np.float32)
    stock_arr = np.empty((total_rows, 24), dtype=np.float32)
    sid_arr = np.empty(total_rows, dtype=np.int64)
    pid_arr = np.empty(total_rows, dtype=np.int64)

    # Phase 3: fill
    cursor = 0
    n_series = len(series_info)
    for i, (sid, pid, sd, tmask, n_t) in enumerate(series_info):
        if (i + 1) % 10000 == 0:
            print(f'    ... {i+1:,}/{n_series:,} serie')

        ti = np.where(tmask)[0]
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


# ---------------------------------------------------------------------------
# 5. PINNDataset (includes stock for loss computation)
# ---------------------------------------------------------------------------
class PINNDataset(Dataset):
    def __init__(self, cat, cont, lags, targets, stock):
        self.cat = torch.from_numpy(cat)
        self.cont = torch.from_numpy(cont)
        self.lags = torch.from_numpy(lags)
        self.targets = torch.from_numpy(targets)
        self.stock = torch.from_numpy(stock)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (self.cat[idx], self.cont[idx], self.lags[idx],
                self.targets[idx], self.stock[idx])


# ---------------------------------------------------------------------------
# 6. PINNRetail model (encoder + two heads)
# ---------------------------------------------------------------------------
class PINNRetail(nn.Module):
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

        # Shared encoder
        encoder_layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.ReLU())
            prev_dim = h
        self.encoder = nn.Sequential(*encoder_layers)

        # Demand head: D*(t) > 0
        self.head_D = nn.Sequential(
            nn.Linear(prev_dim, 24),
            nn.Softplus(),
        )

        # Inventory head: I*(t) >= 0
        self.head_I = nn.Sequential(
            nn.Linear(prev_dim, 24),
            nn.Softplus(),
        )

    def forward(self, cat, cont, lags):
        # Embeddings
        emb_list = []
        for i, name in enumerate(self.emb_names):
            emb_list.append(self.embeddings[name](cat[:, i]))

        x = torch.cat(emb_list + [cont], dim=1)
        if lags.shape[1] > 0:
            x = torch.cat([x, lags], dim=1)

        h = self.encoder(x)

        D_star = self.head_D(h)   # (B, 24) demand
        I_star = self.head_I(h)   # (B, 24) inventory

        return D_star, I_star


# ---------------------------------------------------------------------------
# 7. PINN Loss function
# ---------------------------------------------------------------------------
def compute_pinn_loss(D_star, I_star, targets, stock,
                      lambda_b, lambda_c, rho_b, rho_c):
    """Compute PINN loss with ALM.

    Args:
        D_star: (B, 24) demand predictions
        I_star: (B, 24) inventory predictions
        targets: (B, 24) observed sales
        stock: (B, 24) stock status (0=in-stock, 1=stockout)
        lambda_b, lambda_c: dual variables (float)
        rho_b, rho_c: penalty coefficients (float)

    Returns:
        L_total, L_data, V_b, V_c (all tensors for backprop except last 3 are .item())
    """
    in_mask = (stock == 0)
    so_mask = (stock == 1)

    # --- L_data: MSE on in-stock hours only ---
    n_in = in_mask.sum()
    if n_in > 0:
        L_data = (D_star[in_mask] - targets[in_mask]).pow(2).mean()
    else:
        L_data = torch.tensor(0.0, device=D_star.device)

    # --- L_boundary ---
    # Sub1: stockout -> I* ≈ 0
    n_so = so_mask.sum()
    if n_so > 0:
        i_so = I_star[so_mask]
        V_b1 = i_so.mean()         # linear violation
        Q_b1 = i_so.pow(2).mean()  # quadratic violation
    else:
        V_b1 = torch.tensor(0.0, device=D_star.device)
        Q_b1 = torch.tensor(0.0, device=D_star.device)

    # Sub2: in-stock -> I* >= D*
    if n_in > 0:
        gap = F.relu(D_star[in_mask] - I_star[in_mask])
        V_b2 = gap.mean()
        Q_b2 = gap.pow(2).mean()
    else:
        V_b2 = torch.tensor(0.0, device=D_star.device)
        Q_b2 = torch.tensor(0.0, device=D_star.device)

    V_b = V_b1 + V_b2
    Q_b = Q_b1 + Q_b2

    # --- L_cons: within-day conservation (hours 0..22) ---
    # R(h) = I*(h+1) - I*(h) + min(D*(h), I*(h)) >= 0
    min_DI = torch.min(D_star[:, :-1], I_star[:, :-1])   # (B, 23)
    delta_I = I_star[:, 1:] - I_star[:, :-1]              # (B, 23)
    implicit_R = delta_I + min_DI                          # (B, 23)
    neg_R = F.relu(-implicit_R)                            # only violations

    V_c = neg_R.mean()
    Q_c = neg_R.pow(2).mean()

    # --- ALM loss ---
    L_total = (L_data
               + lambda_b * V_b + (rho_b / 2.0) * Q_b
               + lambda_c * V_c + (rho_c / 2.0) * Q_c)

    return L_total, L_data.item(), V_b.item(), V_c.item()


# ---------------------------------------------------------------------------
# 8. Prediction function (outputs D* only)
# ---------------------------------------------------------------------------
def predict_pinn(model, data, device):
    """Generate demand predictions D* for a dataset."""
    model.eval()
    cat_t = torch.from_numpy(data['cat']).to(device)
    cont_t = torch.from_numpy(data['cont']).to(device)
    lags_t = torch.from_numpy(data['lags']).to(device)

    all_D = []
    all_I = []
    chunk_size = 10000
    with torch.no_grad():
        for start in range(0, len(cat_t), chunk_size):
            end = min(start + chunk_size, len(cat_t))
            D_star, I_star = model(
                cat_t[start:end], cont_t[start:end], lags_t[start:end])
            all_D.append(D_star.cpu().numpy())
            all_I.append(I_star.cpu().numpy())

    return np.concatenate(all_D, axis=0), np.concatenate(all_I, axis=0)


# ---------------------------------------------------------------------------
# 9. Evaluate constraint violations on a dataset
# ---------------------------------------------------------------------------
def evaluate_constraints(model, data, device, max_samples=50000):
    """Evaluate constraint violations on training data (subset for speed)."""
    model.eval()
    n = len(data['cat'])
    if n > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, max_samples, replace=False)
        cat = data['cat'][idx]
        cont = data['cont'][idx]
        lags = data['lags'][idx]
        targets = data['targets'][idx]
        stock = data['stock'][idx]
    else:
        cat = data['cat']
        cont = data['cont']
        lags = data['lags']
        targets = data['targets']
        stock = data['stock']

    cat_t = torch.from_numpy(cat).to(device)
    cont_t = torch.from_numpy(cont).to(device)
    lags_t = torch.from_numpy(lags).to(device)
    targets_t = torch.from_numpy(targets).to(device)
    stock_t = torch.from_numpy(stock).to(device)

    with torch.no_grad():
        all_D, all_I = [], []
        chunk_size = 10000
        for start in range(0, len(cat_t), chunk_size):
            end = min(start + chunk_size, len(cat_t))
            D_star, I_star = model(
                cat_t[start:end], cont_t[start:end], lags_t[start:end])
            all_D.append(D_star)
            all_I.append(I_star)
        D_star = torch.cat(all_D, dim=0)
        I_star = torch.cat(all_I, dim=0)

        in_mask = (stock_t == 0)
        so_mask = (stock_t == 1)

        # Boundary violations
        V_b1 = I_star[so_mask].mean().item() if so_mask.sum() > 0 else 0.0
        gap = F.relu(D_star[in_mask] - I_star[in_mask])
        V_b2 = gap.mean().item() if in_mask.sum() > 0 else 0.0
        V_b = V_b1 + V_b2

        # Conservation violations
        min_DI = torch.min(D_star[:, :-1], I_star[:, :-1])
        delta_I = I_star[:, 1:] - I_star[:, :-1]
        implicit_R = delta_I + min_DI
        neg_R = F.relu(-implicit_R)
        V_c = neg_R.mean().item()

        # L_data for monitoring
        if in_mask.sum() > 0:
            L_data = (D_star[in_mask] - targets_t[in_mask]).pow(2).mean().item()
        else:
            L_data = 0.0

    return V_b, V_c, L_data


# ---------------------------------------------------------------------------
# 10. Training loop with ALM
# ---------------------------------------------------------------------------
def train_pinn(model, train_data, val_data, device):
    """Train PINN with warmup + ALM."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # DataLoader
    train_ds = PINNDataset(train_data['cat'], train_data['cont'],
                           train_data['lags'], train_data['targets'],
                           train_data['stock'])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)

    # ALM state
    lambda_b = 0.0
    lambda_c = 0.0
    rho_b = RHO_INIT
    rho_c = RHO_INIT

    # Best model tracking
    best_val_wape = float('inf')
    best_state = None
    best_info = {}
    alm_no_improve = 0

    # Compute val WAPE (in-stock pooled)
    def compute_val_wape():
        D_pred, _ = predict_pinn(model, val_data, device)
        obs = val_data['targets']
        stock = val_data['stock']
        in_mask = stock == 0
        sae = np.abs(D_pred[in_mask] - obs[in_mask]).sum()
        sao = np.abs(obs[in_mask]).sum()
        return sae / sao if sao > 0 else float('inf')

    def run_epoch(epoch_num, lam_b, lam_c, r_b, r_c):
        """Run one training epoch. Returns avg loss components."""
        model.train()
        sum_loss = 0.0
        sum_ldata = 0.0
        sum_vb = 0.0
        sum_vc = 0.0
        n_batches = 0

        for cat, cont, lags, targets, stock in train_loader:
            cat = cat.to(device)
            cont = cont.to(device)
            lags = lags.to(device)
            targets = targets.to(device)
            stock = stock.to(device)

            D_star, I_star = model(cat, cont, lags)
            L_total, L_data, V_b, V_c = compute_pinn_loss(
                D_star, I_star, targets, stock,
                lam_b, lam_c, r_b, r_c)

            optimizer.zero_grad()
            L_total.backward()
            optimizer.step()

            sum_loss += L_total.item()
            sum_ldata += L_data
            sum_vb += V_b
            sum_vc += V_c
            n_batches += 1

        return (sum_loss / n_batches, sum_ldata / n_batches,
                sum_vb / n_batches, sum_vc / n_batches)

    # -----------------------------------------------------------------------
    # Phase 1: Warmup (L_data only)
    # -----------------------------------------------------------------------
    print(f'\n  Phase 1: Warmup ({WARMUP_EPOCHS} epochs, L_data only)...')

    for epoch in range(1, WARMUP_EPOCHS + 1):
        avg_loss, avg_ldata, _, _ = run_epoch(epoch, 0.0, 0.0, 0.0, 0.0)
        val_wape = compute_val_wape()
        print(f'    Warmup {epoch}: L_data={avg_ldata:.6f}, '
              f'val_WAPE_in={val_wape:.6f}')

        if val_wape < best_val_wape:
            best_val_wape = val_wape
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            best_info = {'epoch': epoch, 'phase': 'warmup',
                         'val_wape': val_wape}

    total_epochs = WARMUP_EPOCHS

    # -----------------------------------------------------------------------
    # Phase 2: ALM iterations
    # -----------------------------------------------------------------------
    print(f'\n  Phase 2: ALM ({N_OUTER} max iterations × {K_INNER} epochs)...')

    V_b_prev = float('inf')
    V_c_prev = float('inf')

    for alm_iter in range(1, N_OUTER + 1):
        # Primal step: K_INNER epochs with current lambda, rho
        for inner_epoch in range(1, K_INNER + 1):
            total_epochs += 1
            avg_loss, avg_ldata, avg_vb, avg_vc = run_epoch(
                total_epochs, lambda_b, lambda_c, rho_b, rho_c)

        # Evaluate constraints on training set
        V_b_eval, V_c_eval, L_data_eval = evaluate_constraints(
            model, train_data, device)

        # Val WAPE
        val_wape = compute_val_wape()

        print(f'    ALM {alm_iter:2d} (ep {total_epochs:3d}): '
              f'L_data={L_data_eval:.6f}, V_b={V_b_eval:.5f}, '
              f'V_c={V_c_eval:.5f}, '
              f'lam_b={lambda_b:.3f}, lam_c={lambda_c:.3f}, '
              f'rho_b={rho_b:.1f}, rho_c={rho_c:.1f}, '
              f'val_WAPE_in={val_wape:.6f}')

        # Dual step
        lambda_b = max(0.0, lambda_b + rho_b * V_b_eval)
        lambda_c = max(0.0, lambda_c + rho_c * V_c_eval)

        # Adaptation: increase rho if violation not improving enough
        if V_b_eval > 0.25 * V_b_prev and V_b_eval > 1e-6:
            rho_b *= GAMMA
        if V_c_eval > 0.25 * V_c_prev and V_c_eval > 1e-6:
            rho_c *= GAMMA

        V_b_prev = V_b_eval
        V_c_prev = V_c_eval

        # Best model tracking
        if val_wape < best_val_wape:
            best_val_wape = val_wape
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            best_info = {
                'epoch': total_epochs, 'alm_iter': alm_iter,
                'phase': 'alm', 'val_wape': val_wape,
                'V_b': V_b_eval, 'V_c': V_c_eval,
                'lambda_b': lambda_b, 'lambda_c': lambda_c,
                'rho_b': rho_b, 'rho_c': rho_c,
            }
            alm_no_improve = 0
        else:
            alm_no_improve += 1

        if alm_no_improve >= ALM_PATIENCE:
            print(f'    ALM early stopping at iter {alm_iter} '
                  f'(best at iter {best_info.get("alm_iter", "warmup")})')
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    print(f'\n  Best model: {best_info}')
    print(f'  Total epochs: {total_epochs}')

    return best_info


# ===========================================================================
# MAIN
# ===========================================================================

# ---------------------------------------------------------------------------
# 11. Build datasets
# ---------------------------------------------------------------------------
print('\n3. Costruzione dataset (lag features da S_obs)...')
t0 = time.time()

print('  Building train data...')
train_data = build_dataset_arrays(series_cache, 'train')
n_cont = train_data['cont'].shape[1]
n_lags = train_data['lags'].shape[1]
print(f'  Train: {len(train_data["targets"]):,} samples, '
      f'cont={n_cont}, lags={n_lags}')

print('  Building val data...')
val_data = build_dataset_arrays(series_cache, 'val',
                                 cont_mean=train_data['cont_mean'],
                                 cont_std=train_data['cont_std'],
                                 lag_mean=train_data['lag_mean'],
                                 lag_std=train_data['lag_std'])
print(f'  Val:   {len(val_data["targets"]):,} samples')

print('  Building test data...')
test_data = build_dataset_arrays(series_cache, 'test',
                                  cont_mean=train_data['cont_mean'],
                                  cont_std=train_data['cont_std'],
                                  lag_mean=train_data['lag_mean'],
                                  lag_std=train_data['lag_std'])
print(f'  Test:  {len(test_data["targets"]):,} samples')

elapsed_ds = time.time() - t0
print(f'  Dataset construction: {elapsed_ds:.1f}s')

# ---------------------------------------------------------------------------
# 12. Train PINN
# ---------------------------------------------------------------------------
print('\n4. Training PINN-Retail...')
t0 = time.time()

torch.manual_seed(SEED)
model = PINNRetail(n_cont, n_lags, EMB_DIMS, CARDINALITIES, HIDDEN_SIZES)
n_params = sum(p.numel() for p in model.parameters())
print(f'  Model params: {n_params:,}')
print(f'  Input dim: {sum(EMB_DIMS.values())} (emb) + {n_cont} (cont) + '
      f'{n_lags} (lags) = {sum(EMB_DIMS.values()) + n_cont + n_lags}')

best_info = train_pinn(model, train_data, val_data, DEVICE)

elapsed_train = time.time() - t0
print(f'  Training time: {elapsed_train:.1f}s')

# Save model
torch.save(model.state_dict(),
           os.path.join(RESULTS_DIR, 'pinn_model.pt'))

del train_data
gc.collect()

# ---------------------------------------------------------------------------
# 13. Evaluation on val + test
# ---------------------------------------------------------------------------
print('\n5. Valutazione su val e test...')

pooled_results = {}
per_series_dfs = {}

for split_name, data in [('val', val_data), ('test', test_data)]:
    print(f'\n  {split_name}...')
    D_preds, I_preds = predict_pinn(model, data, DEVICE)
    obs = data['targets']
    stock = data['stock']
    sids = data['store_ids']
    pids = data['product_ids']

    # Pooled metrics (D* vs S_obs)
    p_flat = D_preds.ravel()
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

    # Constraint violation metrics
    in_mask = s_flat == 0
    so_mask = s_flat == 1
    i_flat = I_preds.ravel()

    # V_boundary
    v_b1 = i_flat[so_mask].mean() if so_mask.sum() > 0 else 0.0
    gap = np.maximum(0, p_flat[in_mask] - i_flat[in_mask])
    v_b2 = gap.mean() if in_mask.sum() > 0 else 0.0
    r['v_boundary'] = v_b1 + v_b2

    # V_conservation
    min_DI = np.minimum(D_preds[:, :-1], I_preds[:, :-1])
    delta_I = I_preds[:, 1:] - I_preds[:, :-1]
    impl_R = delta_I + min_DI
    neg_R = np.maximum(0, -impl_R)
    r['v_conservation'] = neg_R.mean()

    # Mean I* during stockout and in-stock
    r['mean_I_stockout'] = i_flat[so_mask].mean() if so_mask.sum() > 0 else 0.0
    r['mean_I_instock'] = i_flat[in_mask].mean() if in_mask.sum() > 0 else 0.0
    r['mean_D_stockout'] = p_flat[so_mask].mean() if so_mask.sum() > 0 else 0.0
    r['mean_D_instock'] = p_flat[in_mask].mean() if in_mask.sum() > 0 else 0.0

    # Per-series metrics
    print('    Calcolo metriche per-serie...')
    df_idx = pd.DataFrame({'sid': sids, 'pid': pids,
                           'row': np.arange(len(sids))})
    records = []
    for (sid, pid), grp in df_idx.groupby(['sid', 'pid']):
        idx = grp['row'].values
        m = compute_metrics(D_preds[idx], obs[idx], stock[idx])
        m['store_id'] = sid
        m['product_id'] = pid
        records.append(m)

    ps = pd.DataFrame(records)
    per_series_dfs[split_name] = ps
    out_path = os.path.join(RESULTS_DIR,
                            f'pinn_{split_name}_per_series.parquet')
    ps.to_parquet(out_path, index=False)
    print(f'    Salvato: {out_path} ({len(ps):,} serie)')

    del D_preds, I_preds
    gc.collect()


# ---------------------------------------------------------------------------
# 14. Tabella risultati pooled
# ---------------------------------------------------------------------------
print(format_metrics_table(pooled_results, model_name='PINN-Retail'))

# PINN-specific metrics
print('\n  PINN Constraint Metrics:')
print(f'  {"Split":<8} {"V_bound":>10} {"V_cons":>10} '
      f'{"D_in":>8} {"D_so":>8} {"I_in":>8} {"I_so":>8}')
print('  ' + '-' * 66)
for split_name in ['val', 'test']:
    r = pooled_results[split_name]
    print(f'  {split_name:<8} {r["v_boundary"]:>10.5f} '
          f'{r["v_conservation"]:>10.5f} '
          f'{r["mean_D_instock"]:>8.5f} '
          f'{r["mean_D_stockout"]:>8.5f} '
          f'{r["mean_I_instock"]:>8.5f} '
          f'{r["mean_I_stockout"]:>8.5f}')

# Shadow prices
if 'lambda_b' in best_info:
    print(f'\n  Shadow prices at convergence:')
    print(f'    lambda_boundary = {best_info["lambda_b"]:.4f}')
    print(f'    lambda_conservation = {best_info["lambda_c"]:.4f}')

# ---------------------------------------------------------------------------
# 15. Distribuzione per-serie
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
# 16. Confronto con tutti i modelli
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
    'PINN-Retail': 'pinn',
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

    if prefix == 'pinn' and 'test' in pooled_results:
        wape_in_pool = pooled_results['test']['wape_instock']
    else:
        wape_in_pool = np.nan

    if np.isnan(wape_in_pool):
        print(f'  {label:<24} {"—":>14} {wape_in_med:>14.4f} '
              f'{wpe_in_med:>12.4f} {wape_all_med:>14.4f}')
    else:
        print(f'  {label:<24} {wape_in_pool:>14.4f} {wape_in_med:>14.4f} '
              f'{wpe_in_med:>12.4f} {wape_all_med:>14.4f}')

# ---------------------------------------------------------------------------
# 17. Figure: PINN distributions
# ---------------------------------------------------------------------------
print('\n6. Generazione figure...')

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('PINN-Retail — Distribuzione per-serie', fontsize=14)

for j, (split_name, ps) in enumerate(per_series_dfs.items()):
    ax = axes[0, j]
    vals = ps['wape_instock'].dropna()
    vals_clipped = vals.clip(upper=vals.quantile(0.99))
    ax.hist(vals_clipped, bins=80, color='steelblue', alpha=0.7, edgecolor='none')
    ax.axvline(vals.median(), color='red', linestyle='--', linewidth=1.5,
               label=f'median={vals.median():.3f}')
    ax.set_title(f'WAPE in-stock — {split_name}')
    ax.set_xlabel('WAPE')
    ax.set_ylabel('N serie')
    ax.legend(fontsize=8)

    ax = axes[1, j]
    vals = ps['wpe_instock'].dropna()
    vals_clipped = vals.clip(lower=vals.quantile(0.01), upper=vals.quantile(0.99))
    ax.hist(vals_clipped, bins=80, color='darkorange', alpha=0.7, edgecolor='none')
    ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
    ax.axvline(vals.median(), color='red', linestyle='--', linewidth=1.5,
               label=f'median={vals.median():.3f}')
    ax.set_title(f'WPE in-stock — {split_name}')
    ax.set_xlabel('WPE')
    ax.set_ylabel('N serie')
    ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig40_pinn_per_series_distributions.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Salvata: fig40_pinn_per_series_distributions.png')

elapsed_total = time.time() - t0
print(f'\n  Tempo totale training + eval: {elapsed_total:.1f}s')

print('\n' + '=' * 72)
print('  DONE')
print('=' * 72)
