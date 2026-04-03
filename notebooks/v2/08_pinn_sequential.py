"""
08_pinn_sequential.py — Fase 1: PINN Sequential (Models D & B)
===============================================================
Piano: CLAUDE_SEQUENTIAL-2.md, Fase C punto (6)

Architettura: Transformer Encoder bidirezionale con due teste (D*, I*).
Processa l'intera sequenza di T=168 ore (7 giorni) e produce D*(t) e I*(t)
per ogni timestep. Niente lag features precalcolate: la rete vede S_obs(t)
direttamente e il self-attention impara le dipendenze temporali.

Modelli:
  D (vanilla): solo L_data, lower bound / ablation
  B (PINN):    L_data + L_boundary + L_cons + ALM

stock_status NON e' un input — usato solo nella loss.

Per-timestep features (11 dim):
  S_obs(t), hour_norm, dow_norm, dayofmonth_norm, cont(7)

Per-series embeddings (72 dim, broadcast su T):
  store(32) + product(32) + city(8)

Workflow:
  1. TUNING:     Train gg 1-83, Val gg 84-90
  2. RETRAINING: Retrain su gg 1-90
  3. TRACCIA A:  Recovery su MNAR test masks (seed=123, gg 1-90)
  4. TRACCIA B:  Forecasting su eval HF (gg 91-97)

Eseguire con: freshnet/bin/python notebooks/v2/08_pinn_sequential.py
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import functools

print = functools.partial(print, flush=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.evaluation.metrics import compute_metrics

# =========================================================================
# 1. Config
# =========================================================================
DEVICE = 'mps' if torch.backends.mps.is_available() else (
    'cuda' if torch.cuda.is_available() else 'cpu')

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Architecture
T_WINDOW = 168        # 7 days × 24 hours
D_MODEL = 64
N_LAYERS = 2
N_HEADS = 4
D_FF = 128
DROPOUT = 0.1
EMB_DIMS = {'store_id': 32, 'product_id': 32, 'city_id': 8}
CARDINALITIES = {'store_id': 898, 'product_id': 865, 'city_id': 18}

# Training
BATCH_SIZE = 512
LR = 1e-3

# Model D (vanilla)
MAX_EPOCHS_D = 50
PATIENCE_D = 10

# Model B (PINN)
WARMUP_EPOCHS = 3
ALM_MAX_ITER = 15
ALM_INNER_EPOCHS = 3
ALM_PATIENCE = 5
RHO_INIT = 1.0
RHO_GAMMA = 2.0

CONT_COLS = ['discount', 'avg_temperature', 'avg_humidity', 'precpt',
             'avg_wind_level', 'holiday_flag', 'activity_flag']

# Subsample for faster dev (set to None for full run)
N_SERIES_SUBSAMPLE = None

print("=" * 72)
print("  FASE 1 — PINN SEQUENTIAL (Models D & B)")
print("=" * 72)
print(f"  Device: {DEVICE}")
print(f"  Window: T={T_WINDOW} ({T_WINDOW//24} days)")
print(f"  d_model={D_MODEL}, layers={N_LAYERS}, heads={N_HEADS}, d_ff={D_FF}")
if N_SERIES_SUBSAMPLE:
    print(f"  *** SUBSAMPLE: {N_SERIES_SUBSAMPLE} series ***")


# =========================================================================
# 2. Data loading
# =========================================================================
t0 = time.time()
print("\n1. Caricamento dati...")

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
df_full['dom'] = df_full['dt_parsed'].dt.day

print(f"  Train: {len(df_train):,}, Eval: {len(df_eval):,}")
print(f"  Full: {len(df_full):,}, {len(all_dates)} giorni")
del df_train, df_eval

# =========================================================================
# 3. Build 3D arrays: (n_series, n_days, 24)
# =========================================================================
print("\n2. Building 3D arrays...")
t1 = time.time()

series_keys = []
series_sales = []   # (n_series, 97, 24)
series_stock = []   # (n_series, 97, 24)
series_conts = []   # (n_series, 97, 7)
series_dows = []    # (n_series, 97)
series_doms = []    # (n_series, 97)
series_city = []    # (n_series,)

groups = df_full.groupby(['store_id', 'product_id'], sort=True)
n_days = len(all_dates)

for (sid, pid), grp in groups:
    grp_s = grp.sort_values('day_num')
    if len(grp_s) != n_days:
        continue  # skip incomplete series
    series_keys.append((sid, pid))
    sales = np.array(grp_s['hours_sale'].tolist(), dtype=np.float32)
    stock = np.array(grp_s['hours_stock_status'].tolist(), dtype=np.float32)
    series_sales.append(sales)
    series_stock.append(stock)
    series_conts.append(grp_s[CONT_COLS].values.astype(np.float32))
    series_dows.append(grp_s['dow'].values.astype(np.int32))
    series_doms.append(grp_s['dom'].values.astype(np.int32))
    series_city.append(grp_s['city_id'].values[0])

n_series_full = len(series_keys)
sales_3d = np.stack(series_sales)   # (S, 97, 24)
stock_3d = np.stack(series_stock)
conts_3d = np.stack(series_conts)   # (S, 97, 7)
dows_2d = np.stack(series_dows)     # (S, 97)
doms_2d = np.stack(series_doms)
city_ids = np.array(series_city, dtype=np.int64)
store_ids = np.array([k[0] for k in series_keys], dtype=np.int64)
product_ids = np.array([k[1] for k in series_keys], dtype=np.int64)

del series_sales, series_stock, series_conts, series_dows, series_doms, series_city

print(f"  {n_series_full:,} serie, {n_days} giorni")
print(f"  sales_3d: {sales_3d.shape}, stock_3d: {stock_3d.shape}")
print(f"  conts_3d: {conts_3d.shape}")

# Subsample
if N_SERIES_SUBSAMPLE and N_SERIES_SUBSAMPLE < n_series_full:
    rng = np.random.default_rng(SEED)
    idx = rng.choice(n_series_full, N_SERIES_SUBSAMPLE, replace=False)
    idx.sort()
    sales_3d = sales_3d[idx]
    stock_3d = stock_3d[idx]
    conts_3d = conts_3d[idx]
    dows_2d = dows_2d[idx]
    doms_2d = doms_2d[idx]
    city_ids = city_ids[idx]
    store_ids = store_ids[idx]
    product_ids = product_ids[idx]
    series_keys = [series_keys[i] for i in idx]
    n_series = N_SERIES_SUBSAMPLE
    print(f"  Subsampled to {n_series} series")
else:
    n_series = n_series_full

print(f"  Tempo: {time.time()-t1:.1f}s")

# =========================================================================
# 4. Normalize continuous features
# =========================================================================
print("\n3. Normalizing continuous features...")

# Z-score from training days 1-83
train_conts = conts_3d[:, :83, :]  # (S, 83, 7)
cont_mean = train_conts.reshape(-1, 7).mean(axis=0)
cont_std = train_conts.reshape(-1, 7).std(axis=0)
cont_std[cont_std < 1e-8] = 1.0

conts_3d_norm = (conts_3d - cont_mean) / cont_std
print(f"  cont_mean: {cont_mean}")
print(f"  cont_std:  {cont_std}")

# =========================================================================
# 5. Load MNAR masks
# =========================================================================
print("\n4. Loading MNAR masks...")
masks_val = pd.read_parquet(os.path.join(DATA_DIR, 'mnar_masks_val.parquet'))
masks_test = pd.read_parquet(os.path.join(DATA_DIR, 'mnar_masks_test.parquet'))

masks_val['dt_parsed'] = pd.to_datetime(masks_val['dt'])
masks_val['day_num'] = masks_val['dt_parsed'].map(date_to_day)
masks_test['dt_parsed'] = pd.to_datetime(masks_test['dt'])
masks_test['day_num'] = masks_test['dt_parsed'].map(date_to_day)

print(f"  Val masks: {len(masks_val):,} (gg 84-90, seed=42)")
print(f"  Test masks: {len(masks_test):,} (gg 1-90, seed=123)")

# Build MNAR 3D arrays
def build_mnar_3d(masks_df, n_series, n_days):
    """Build (n_series, n_days, 24) bool array and ground truth array from mask dataframe."""
    key_to_idx = {k: i for i, k in enumerate(series_keys)}
    mnar_mask = np.zeros((n_series, n_days, 24), dtype=bool)
    mnar_gt = np.zeros((n_series, n_days, 24), dtype=np.float32)

    for _, row in masks_df.iterrows():
        k = (row['store_id'], row['product_id'])
        if k not in key_to_idx:
            continue
        s_idx = key_to_idx[k]
        d_idx = int(row['day_num']) - 1  # 0-indexed
        h = int(row['hour'])
        if d_idx < n_days:
            mnar_mask[s_idx, d_idx, h] = True
            mnar_gt[s_idx, d_idx, h] = row['ground_truth']

    return mnar_mask, mnar_gt

t_mnar = time.time()
# Val masks (days 84-90 → indices 83-89)
mnar_val_mask, mnar_val_gt = build_mnar_3d(masks_val, n_series, n_days)
# Test masks (days 1-90 → indices 0-89)
mnar_test_mask, mnar_test_gt = build_mnar_3d(masks_test, n_series, n_days)
print(f"  Val MNAR positions: {mnar_val_mask.sum():,}")
print(f"  Test MNAR positions: {mnar_test_mask.sum():,}")
print(f"  Tempo: {time.time()-t_mnar:.1f}s")


# =========================================================================
# 6. Window construction
# =========================================================================
def build_windows(day_start, day_end, apply_mnar_mask=None, apply_mnar_gt=None):
    """Build sequential windows of T_WINDOW hours.

    Args:
        day_start: first day (1-indexed)
        day_end: last day (1-indexed)
        apply_mnar_mask: optional (n_series, n_days, 24) bool array
        apply_mnar_gt: optional (n_series, n_days, 24) ground truth

    Returns dict with:
        x_seq:    (N_windows, T, 11) per-timestep features
        cat_ids:  (N_windows, 3) store_id, product_id, city_id
        stock:    (N_windows, T) stock status
        mnar:     (N_windows, T) MNAR mask
        gt:       (N_windows, T) ground truth at MNAR positions
        valid:    (N_windows, T) valid timestep mask
        meta:     (N_windows, 2) [series_idx, start_day_idx]
    """
    d0 = day_start - 1  # 0-indexed
    d1 = day_end - 1
    n_days_range = d1 - d0 + 1
    n_windows_per_series = n_days_range // (T_WINDOW // 24)

    if n_windows_per_series == 0:
        # Partial window: pad
        n_windows_per_series = 1

    N = n_series * n_windows_per_series
    T = T_WINDOW

    x_seq = np.zeros((N, T, 11), dtype=np.float32)
    cat_ids = np.zeros((N, 3), dtype=np.int64)
    stock_arr = np.zeros((N, T), dtype=np.float32)
    mnar_arr = np.zeros((N, T), dtype=bool)
    gt_arr = np.zeros((N, T), dtype=np.float32)
    valid_arr = np.zeros((N, T), dtype=np.float32)
    meta_arr = np.zeros((N, 2), dtype=np.int64)

    win_idx = 0
    for s in range(n_series):
        for w in range(n_windows_per_series):
            start_d = d0 + w * (T_WINDOW // 24)
            end_d = start_d + (T_WINDOW // 24) - 1

            # Handle overflow
            if end_d > d1:
                start_d = d1 - (T_WINDOW // 24) + 1
                end_d = d1
            if start_d < d0:
                start_d = d0

            actual_days = end_d - start_d + 1

            for dd in range(actual_days):
                d_abs = start_d + dd
                if d_abs < 0 or d_abs >= n_days:
                    continue

                for h in range(24):
                    t_idx = dd * 24 + h

                    s_obs = sales_3d[s, d_abs, h]
                    stk = stock_3d[s, d_abs, h]

                    # Apply MNAR mask if provided
                    is_mnar = False
                    if apply_mnar_mask is not None and apply_mnar_mask[s, d_abs, h]:
                        is_mnar = True
                        s_obs = 0.0  # Censor the observation
                        stk = 1.0   # Mark as stockout

                    x_seq[win_idx, t_idx, 0] = s_obs
                    x_seq[win_idx, t_idx, 1] = h / 23.0           # hour_norm
                    x_seq[win_idx, t_idx, 2] = dows_2d[s, d_abs] / 6.0  # dow_norm
                    x_seq[win_idx, t_idx, 3] = (doms_2d[s, d_abs] - 1) / 30.0  # dom_norm
                    x_seq[win_idx, t_idx, 4:11] = conts_3d_norm[s, d_abs]

                    stock_arr[win_idx, t_idx] = stk
                    valid_arr[win_idx, t_idx] = 1.0

                    if is_mnar:
                        mnar_arr[win_idx, t_idx] = True
                        gt_arr[win_idx, t_idx] = apply_mnar_gt[s, d_abs, h]

            cat_ids[win_idx] = [store_ids[s], product_ids[s], city_ids[s]]
            meta_arr[win_idx] = [s, start_d]
            win_idx += 1

    # Trim if fewer windows than allocated
    if win_idx < N:
        x_seq = x_seq[:win_idx]
        cat_ids = cat_ids[:win_idx]
        stock_arr = stock_arr[:win_idx]
        mnar_arr = mnar_arr[:win_idx]
        gt_arr = gt_arr[:win_idx]
        valid_arr = valid_arr[:win_idx]
        meta_arr = meta_arr[:win_idx]

    return {
        'x_seq': x_seq, 'cat_ids': cat_ids, 'stock': stock_arr,
        'mnar': mnar_arr, 'gt': gt_arr, 'valid': valid_arr, 'meta': meta_arr,
    }


print("\n5. Building windows...")
t2 = time.time()

# Train: days 1-84 (12 non-overlapping 7-day windows per series)
train_windows = build_windows(1, 84)
print(f"  Train: {len(train_windows['x_seq']):,} windows (days 1-84)")

# Val: days 84-90 (1 window per series, without MNAR for early stopping)
val_windows = build_windows(84, 90)
print(f"  Val: {len(val_windows['x_seq']):,} windows (days 84-90)")

# Val with MNAR masks applied (for recovery evaluation)
val_mnar_windows = build_windows(84, 90,
                                  apply_mnar_mask=mnar_val_mask,
                                  apply_mnar_gt=mnar_val_gt)
n_mnar_val = val_mnar_windows['mnar'].sum()
print(f"  Val MNAR: {len(val_mnar_windows['x_seq']):,} windows, "
      f"{n_mnar_val:,} masked positions")

print(f"  Tempo: {time.time()-t2:.1f}s")


# =========================================================================
# 7. Dataset & Model
# =========================================================================
class SeqDataset(Dataset):
    def __init__(self, data):
        self.x_seq = torch.from_numpy(data['x_seq'])
        self.cat_ids = torch.from_numpy(data['cat_ids'])
        self.stock = torch.from_numpy(data['stock'])
        self.valid = torch.from_numpy(data['valid'])

    def __len__(self):
        return len(self.x_seq)

    def __getitem__(self, idx):
        return (self.x_seq[idx], self.cat_ids[idx],
                self.stock[idx], self.valid[idx])


class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                        (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class PINNSequential(nn.Module):
    def __init__(self, n_cont_feats, emb_dims, cardinalities,
                 d_model, n_layers, n_heads, d_ff, dropout):
        super().__init__()

        # Embeddings (per-series, broadcast to all timesteps)
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(cardinalities[name], emb_dims[name])
            for name in emb_dims
        })
        self.emb_names = list(emb_dims.keys())
        total_emb = sum(emb_dims.values())

        # Input projection
        input_dim = n_cont_feats + total_emb
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pe = SinusoidalPE(d_model, max_len=T_WINDOW + 16)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='relu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Head D (demand)
        self.head_D = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1), nn.Softplus())

        # Head I (inventory)
        self.head_I = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1), nn.Softplus())

    def forward(self, x_seq, cat_ids):
        """
        x_seq: (B, T, n_cont_feats) per-timestep features
        cat_ids: (B, 3) store_id, product_id, city_id

        Returns:
            D_star: (B, T) demand prediction
            I_star: (B, T) inventory prediction
        """
        B, T, _ = x_seq.shape

        # Embeddings → broadcast to all timesteps
        emb_list = [self.embeddings[name](cat_ids[:, i])
                    for i, name in enumerate(self.emb_names)]
        emb_cat = torch.cat(emb_list, dim=1)   # (B, total_emb)
        emb_broadcast = emb_cat.unsqueeze(1).expand(-1, T, -1)  # (B, T, total_emb)

        # Concat per-timestep + embeddings
        x = torch.cat([x_seq, emb_broadcast], dim=2)  # (B, T, input_dim)

        # Project + PE + Transformer
        h = self.input_proj(x)      # (B, T, d_model)
        h = self.pe(h)
        h = self.encoder(h)         # (B, T, d_model)

        # Two heads
        D_star = self.head_D(h).squeeze(-1)  # (B, T)
        I_star = self.head_I(h).squeeze(-1)  # (B, T)

        return D_star, I_star


# =========================================================================
# 8. Loss function
# =========================================================================
def compute_loss(D_star, I_star, x_seq, stock, valid, mode='vanilla',
                 lam_b=0.0, rho_b=0.0, lam_c=0.0, rho_c=0.0):
    """Compute loss for PINN Sequential.

    Args:
        D_star: (B, T) predicted demand
        I_star: (B, T) predicted inventory
        x_seq: (B, T, 11) input features (S_obs is feature 0)
        stock: (B, T) stock status (0=in-stock, 1=stockout)
        valid: (B, T) valid timestep mask
        mode: 'vanilla' (L_data only) or 'pinn' (all constraints)
    """
    targets = x_seq[:, :, 0]  # S_obs

    in_mask = (stock == 0) & (valid > 0)   # in-stock and valid
    so_mask = (stock == 1) & (valid > 0)   # stockout/MNAR and valid

    # L_data: MSE on in-stock hours only
    if in_mask.sum() > 0:
        L_data = ((D_star[in_mask] - targets[in_mask]) ** 2).mean()
    else:
        L_data = torch.tensor(0.0, device=D_star.device)

    if mode == 'vanilla':
        return L_data, {'L_data': L_data.item(), 'V_b': 0.0, 'V_c': 0.0}

    # ---- PINN constraints ----

    # L_boundary: I* ≈ 0 at stockout, I* ≥ D* at in-stock
    if so_mask.sum() > 0:
        V_b_so = I_star[so_mask].mean()
    else:
        V_b_so = torch.tensor(0.0, device=D_star.device)

    if in_mask.sum() > 0:
        gap = torch.relu(D_star[in_mask] - I_star[in_mask])
        V_b_in = gap.mean()
    else:
        V_b_in = torch.tensor(0.0, device=D_star.device)

    V_b = V_b_so + V_b_in

    # Quadratic penalty
    if so_mask.sum() > 0:
        Q_b_so = I_star[so_mask].pow(2).mean()
    else:
        Q_b_so = torch.tensor(0.0, device=D_star.device)

    if in_mask.sum() > 0:
        Q_b_in = torch.relu(D_star[in_mask] - I_star[in_mask]).pow(2).mean()
    else:
        Q_b_in = torch.tensor(0.0, device=D_star.device)

    Q_b = Q_b_so + Q_b_in

    # L_cons: conservation inequality R(t) >= 0
    # I(t+1) - I(t) + min(D(t), I(t)) = R(t) >= 0
    min_DI = torch.min(D_star[:, :-1], I_star[:, :-1])
    R_implied = I_star[:, 1:] - I_star[:, :-1] + min_DI
    neg_R = torch.relu(-R_implied)

    # Valid pairs: both t and t+1 must be valid
    valid_pairs = (valid[:, :-1] > 0) & (valid[:, 1:] > 0)

    if valid_pairs.sum() > 0:
        V_c = neg_R[valid_pairs].mean()
        Q_c = neg_R[valid_pairs].pow(2).mean()
    else:
        V_c = torch.tensor(0.0, device=D_star.device)
        Q_c = torch.tensor(0.0, device=D_star.device)

    # ALM loss
    L_total = (L_data
               + lam_b * V_b + (rho_b / 2) * Q_b
               + lam_c * V_c + (rho_c / 2) * Q_c)

    return L_total, {
        'L_data': L_data.item(),
        'V_b': V_b.item(), 'Q_b': Q_b.item(),
        'V_c': V_c.item(), 'Q_c': Q_c.item(),
    }


# =========================================================================
# 9. Prediction and evaluation utilities
# =========================================================================
def predict_windows(model, data, device, batch_size=256):
    """Predict D*, I* for all windows."""
    model.eval()
    x_seq_t = torch.from_numpy(data['x_seq'])
    cat_ids_t = torch.from_numpy(data['cat_ids'])

    all_D, all_I = [], []
    with torch.no_grad():
        for s in range(0, len(x_seq_t), batch_size):
            e = min(s + batch_size, len(x_seq_t))
            D, I = model(x_seq_t[s:e].to(device), cat_ids_t[s:e].to(device))
            all_D.append(D.cpu().numpy())
            all_I.append(I.cpu().numpy())

    return np.concatenate(all_D), np.concatenate(all_I)


def eval_wape_instock(model, data, device):
    """Compute WAPE on in-stock hours (for early stopping)."""
    D_star, _ = predict_windows(model, data, device)
    stock = data['stock']
    valid = data['valid']
    targets = data['x_seq'][:, :, 0]

    in_mask = (stock == 0) & (valid > 0)
    preds = D_star[in_mask]
    obs = targets[in_mask]

    sae = np.abs(preds - obs).sum()
    sao = np.abs(obs).sum()
    return sae / sao if sao > 0 else float('inf')


def eval_recovery_mnar(model, data, device):
    """Evaluate recovery on MNAR-masked positions."""
    D_star, _ = predict_windows(model, data, device)
    mnar = data['mnar']
    gt = data['gt']

    if mnar.sum() == 0:
        return {'wape_pool': np.nan, 'wpe_pool': np.nan}

    preds = D_star[mnar]
    gts = gt[mnar]

    sae = np.abs(preds - gts).sum()
    se = (preds - gts).sum()
    s_gt = np.abs(gts).sum()

    return {
        'wape_pool': sae / s_gt if s_gt > 0 else np.nan,
        'wpe_pool': se / gts.sum() if gts.sum() != 0 else np.nan,
        'n_positions': int(mnar.sum()),
        'pred_mean': float(preds.mean()),
        'gt_mean': float(gts.mean()),
    }


def eval_recovery_per_series(model, windows_data, device):
    """Evaluate recovery per-series for MNAR positions."""
    D_star, _ = predict_windows(model, windows_data, device)
    mnar = windows_data['mnar']
    gt = windows_data['gt']
    meta = windows_data['meta']

    # Group by series
    records = []
    for s_idx in range(n_series):
        # Find windows for this series
        win_mask = meta[:, 0] == s_idx
        if not win_mask.any():
            continue

        s_mnar = mnar[win_mask]
        s_gt = gt[win_mask]
        s_D = D_star[win_mask]

        if s_mnar.sum() == 0:
            continue

        preds = s_D[s_mnar]
        gts = s_gt[s_mnar]

        sae = np.abs(preds - gts).sum()
        se = (preds - gts).sum()
        s_gt_sum = np.abs(gts).sum()

        wape = sae / s_gt_sum if s_gt_sum > 0 else np.nan
        wpe = se / gts.sum() if gts.sum() != 0 else np.nan

        sid, pid = series_keys[s_idx]
        records.append({
            'store_id': sid, 'product_id': pid,
            'n_masked': int(s_mnar.sum()),
            'wape_recovery': wape, 'wpe_recovery': wpe,
            'gt_sum': float(gts.sum()), 'pred_sum': float(preds.sum()),
        })

    return pd.DataFrame(records)


def eval_forecasting(model, windows_data, device):
    """Evaluate forecasting (vs original S_obs)."""
    D_star, _ = predict_windows(model, windows_data, device)
    stock = windows_data['stock']
    valid = windows_data['valid']
    targets = windows_data['x_seq'][:, :, 0]  # Original S_obs
    meta = windows_data['meta']

    # Pooled
    in_mask = (stock == 0) & (valid > 0)
    preds_in = D_star[in_mask]
    obs_in = targets[in_mask]
    sae = np.abs(preds_in - obs_in).sum()
    sao = np.abs(obs_in).sum()
    pooled = {
        'wape_instock': sae / sao if sao > 0 else np.nan,
        'wpe_instock': (preds_in - obs_in).sum() / obs_in.sum() if obs_in.sum() != 0 else np.nan,
    }

    # Per-series
    records = []
    for s_idx in range(n_series):
        win_mask = meta[:, 0] == s_idx
        if not win_mask.any():
            continue

        s_D = D_star[win_mask]
        s_stock = stock[win_mask]
        s_valid = valid[win_mask]
        s_targets = targets[win_mask]

        s_in = (s_stock == 0) & (s_valid > 0)
        if s_in.sum() == 0:
            continue

        p = s_D.ravel()[s_in.ravel()]
        o = s_targets.ravel()[s_in.ravel()]
        stk = s_stock.ravel()
        vld = s_valid.ravel()

        # Use compute_metrics on flattened data
        p_all = s_D.ravel()[s_valid.ravel() > 0]
        o_all = s_targets.ravel()[s_valid.ravel() > 0]
        stk_all = s_stock.ravel()[s_valid.ravel() > 0]

        m = compute_metrics(p_all, o_all, stk_all)

        sid, pid = series_keys[s_idx]
        m['store_id'] = sid
        m['product_id'] = pid
        records.append(m)

    return pooled, pd.DataFrame(records)


# =========================================================================
# 10. Training functions
# =========================================================================
def train_vanilla(model, train_data, val_data, device, max_epochs, patience, lr):
    """Train Model D (vanilla, L_data only)."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = SeqDataset(train_data)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=0, pin_memory=False)

    best_wape = float('inf')
    best_epoch = 0
    best_state = None
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for x_seq, cat_ids, stock, valid in train_loader:
            x_seq = x_seq.to(device)
            cat_ids = cat_ids.to(device)
            stock = stock.to(device)
            valid = valid.to(device)

            D_star, I_star = model(x_seq, cat_ids)
            loss, _ = compute_loss(D_star, I_star, x_seq, stock, valid, mode='vanilla')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        val_wape = eval_wape_instock(model, val_data, device)

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}: loss={avg_loss:.6f}, val_WAPE_in={val_wape:.6f}")

        if val_wape < best_wape:
            best_wape = val_wape
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"    Early stopping epoch {epoch} (best={best_epoch}, WAPE={best_wape:.6f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return best_wape, best_epoch


def train_pinn(model, train_data, val_data, device, warmup_epochs, alm_max_iter,
               alm_inner_epochs, alm_patience, lr, rho_init, rho_gamma):
    """Train Model B (PINN with ALM)."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = SeqDataset(train_data)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=0, pin_memory=False)

    # Warmup: L_data only
    print(f"    Warmup: {warmup_epochs} epochs (L_data only)...")
    for epoch in range(1, warmup_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for x_seq, cat_ids, stock, valid in train_loader:
            x_seq = x_seq.to(device)
            cat_ids = cat_ids.to(device)
            D_star, I_star = model(x_seq, cat_ids)
            loss, _ = compute_loss(D_star, I_star, x_seq, stock.to(device),
                                   valid.to(device), mode='vanilla')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"      Epoch {epoch}: loss={total_loss/n_batches:.6f}")

    # ALM iterations
    lam_b, lam_c = 0.0, 0.0
    rho_b, rho_c = rho_init, rho_init
    best_wape = float('inf')
    best_alm_iter = 0
    best_state = None
    no_improve = 0
    prev_V_b, prev_V_c = float('inf'), float('inf')

    total_epoch = warmup_epochs

    for alm_iter in range(1, alm_max_iter + 1):
        print(f"\n    ALM iter {alm_iter}/{alm_max_iter} "
              f"(lam_b={lam_b:.4f}, rho_b={rho_b:.2f}, "
              f"lam_c={lam_c:.4f}, rho_c={rho_c:.2f})")

        # Inner epochs
        avg_metrics = None
        for inner_ep in range(1, alm_inner_epochs + 1):
            model.train()
            total_loss = 0.0
            n_batches = 0
            ep_metrics = {'L_data': 0., 'V_b': 0., 'V_c': 0.}

            for x_seq, cat_ids, stock, valid in train_loader:
                x_seq = x_seq.to(device)
                cat_ids = cat_ids.to(device)
                stock = stock.to(device)
                valid = valid.to(device)

                D_star, I_star = model(x_seq, cat_ids)
                loss, metrics = compute_loss(
                    D_star, I_star, x_seq, stock, valid, mode='pinn',
                    lam_b=lam_b, rho_b=rho_b, lam_c=lam_c, rho_c=rho_c)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
                for k in ep_metrics:
                    ep_metrics[k] += metrics[k]

            total_epoch += 1
            avg_metrics = {k: v / n_batches for k, v in ep_metrics.items()}

        # Evaluate
        val_wape = eval_wape_instock(model, val_data, device)
        print(f"      L_data={avg_metrics['L_data']:.6f}, "
              f"V_b={avg_metrics['V_b']:.6f}, V_c={avg_metrics['V_c']:.6f}, "
              f"val_WAPE={val_wape:.6f}")

        # Dual update
        lam_b = max(0, lam_b + rho_b * avg_metrics['V_b'])
        lam_c = max(0, lam_c + rho_c * avg_metrics['V_c'])

        # Rho adaptation
        if avg_metrics['V_b'] > 0.75 * prev_V_b:
            rho_b *= rho_gamma
        if avg_metrics['V_c'] > 0.75 * prev_V_c:
            rho_c *= rho_gamma

        prev_V_b = avg_metrics['V_b']
        prev_V_c = avg_metrics['V_c']

        # Early stopping on val WAPE
        if val_wape < best_wape:
            best_wape = val_wape
            best_alm_iter = alm_iter
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= alm_patience:
            print(f"    ALM early stopping at iter {alm_iter} "
                  f"(best={best_alm_iter}, WAPE={best_wape:.6f})")
            break

    total_epochs = warmup_epochs + best_alm_iter * alm_inner_epochs
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return best_wape, total_epochs, best_alm_iter


def retrain_fixed(model, train_data, device, lr, n_epochs):
    """Retrain for fixed number of epochs (no validation)."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = SeqDataset(train_data)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=0, pin_memory=False)

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for x_seq, cat_ids, stock, valid in train_loader:
            x_seq = x_seq.to(device)
            cat_ids = cat_ids.to(device)
            D_star, I_star = model(x_seq, cat_ids)
            loss, _ = compute_loss(D_star, I_star, x_seq, stock.to(device),
                                   valid.to(device), mode='vanilla')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if epoch % 5 == 0 or epoch == 1:
            print(f"      Epoch {epoch:3d}/{n_epochs}: loss={total_loss/n_batches:.6f}")


def retrain_pinn_fixed(model, train_data, device, lr, warmup_epochs, alm_iters,
                       inner_epochs, rho_init, rho_gamma):
    """Retrain PINN for fixed ALM iterations."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = SeqDataset(train_data)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=0, pin_memory=False)

    # Warmup
    for epoch in range(1, warmup_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for x_seq, cat_ids, stock, valid in train_loader:
            x_seq = x_seq.to(device)
            cat_ids = cat_ids.to(device)
            D_star, I_star = model(x_seq, cat_ids)
            loss, _ = compute_loss(D_star, I_star, x_seq, stock.to(device),
                                   valid.to(device), mode='vanilla')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if epoch % 3 == 0 or epoch == 1:
            print(f"      Warmup {epoch}: loss={total_loss/n_batches:.6f}")

    # ALM
    lam_b, lam_c = 0.0, 0.0
    rho_b, rho_c = rho_init, rho_init
    prev_V_b, prev_V_c = float('inf'), float('inf')

    for alm_iter in range(1, alm_iters + 1):
        avg_metrics = None
        for inner_ep in range(1, inner_epochs + 1):
            model.train()
            total_loss = 0.0
            n_batches = 0
            ep_metrics = {'L_data': 0., 'V_b': 0., 'V_c': 0.}
            for x_seq, cat_ids, stock, valid in train_loader:
                x_seq = x_seq.to(device)
                cat_ids = cat_ids.to(device)
                stock = stock.to(device)
                valid = valid.to(device)
                D_star, I_star = model(x_seq, cat_ids)
                loss, metrics = compute_loss(
                    D_star, I_star, x_seq, stock, valid, mode='pinn',
                    lam_b=lam_b, rho_b=rho_b, lam_c=lam_c, rho_c=rho_c)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
                for k in ep_metrics:
                    ep_metrics[k] += metrics[k]
            avg_metrics = {k: v / n_batches for k, v in ep_metrics.items()}

        print(f"      ALM {alm_iter}: L_data={avg_metrics['L_data']:.6f}, "
              f"V_b={avg_metrics['V_b']:.6f}, V_c={avg_metrics['V_c']:.6f}")

        lam_b = max(0, lam_b + rho_b * avg_metrics['V_b'])
        lam_c = max(0, lam_c + rho_c * avg_metrics['V_c'])
        if avg_metrics['V_b'] > 0.75 * prev_V_b:
            rho_b *= rho_gamma
        if avg_metrics['V_c'] > 0.75 * prev_V_c:
            rho_c *= rho_gamma
        prev_V_b = avg_metrics['V_b']
        prev_V_c = avg_metrics['V_c']


# =========================================================================
# 11. Run Model D (Vanilla)
# =========================================================================
N_CONT_FEATS = 11  # S_obs + hour + dow + dom + 7 cont

print("\n\n" + "=" * 72)
print("  MODEL D — TRANSFORMER VANILLA (L_data only)")
print("=" * 72)
t_d = time.time()

torch.manual_seed(SEED)
model_D = PINNSequential(N_CONT_FEATS, EMB_DIMS, CARDINALITIES,
                          D_MODEL, N_LAYERS, N_HEADS, D_FF, DROPOUT)
n_params = sum(p.numel() for p in model_D.parameters())
print(f"  Params: {n_params:,}")

# Train
print("\n  Training Model D...")
best_wape_D, best_epoch_D = train_vanilla(
    model_D, train_windows, val_windows, DEVICE, MAX_EPOCHS_D, PATIENCE_D, LR)
print(f"  Best: epoch={best_epoch_D}, val WAPE_in={best_wape_D:.6f}")
print(f"  Training time: {time.time()-t_d:.0f}s")

torch.save(model_D.state_dict(), os.path.join(RESULTS_DIR, 'pinn_seq_D.pt'))

# Val recovery evaluation
print("\n  Val recovery (MNAR val, seed=42):")
rec_val_D = eval_recovery_mnar(model_D, val_mnar_windows, DEVICE)
print(f"    WAPE_recovery: {rec_val_D['wape_pool']:.4f}")
print(f"    WPE_recovery:  {rec_val_D['wpe_pool']:.4f}")

# Retrain on days 1-90
print(f"\n  Retraining Model D on days 1-90 ({best_epoch_D} epochs)...")
t_rt = time.time()
retrain_windows_D = build_windows(1, 90)
print(f"  Retrain windows: {len(retrain_windows_D['x_seq']):,}")

torch.manual_seed(SEED)
model_D_rt = PINNSequential(N_CONT_FEATS, EMB_DIMS, CARDINALITIES,
                              D_MODEL, N_LAYERS, N_HEADS, D_FF, DROPOUT)
retrain_fixed(model_D_rt, retrain_windows_D, DEVICE, LR, best_epoch_D)
print(f"  Retrain time: {time.time()-t_rt:.0f}s")

torch.save(model_D_rt.state_dict(), os.path.join(RESULTS_DIR, 'pinn_seq_D_retrained.pt'))

# Test recovery (MNAR test, seed=123)
print("\n  Test recovery (MNAR test, seed=123):")
test_mnar_windows = build_windows(1, 90,
                                   apply_mnar_mask=mnar_test_mask,
                                   apply_mnar_gt=mnar_test_gt)
rec_test_D = eval_recovery_mnar(model_D_rt, test_mnar_windows, DEVICE)
print(f"    WAPE_recovery: {rec_test_D['wape_pool']:.4f}")
print(f"    WPE_recovery:  {rec_test_D['wpe_pool']:.4f}")
print(f"    N positions:   {rec_test_D['n_positions']:,}")

ps_rec_D = eval_recovery_per_series(model_D_rt, test_mnar_windows, DEVICE)
ps_rec_D.to_parquet(os.path.join(RESULTS_DIR, 'pinn_seq_D_recovery_per_series.parquet'),
                     index=False)
print(f"    WAPE_recovery median: {ps_rec_D['wape_recovery'].median():.4f}")
print(f"    WPE_recovery median:  {ps_rec_D['wpe_recovery'].median():.4f}")

# Test forecasting (eval HF)
print("\n  Test forecasting (eval HF, gg 91-97):")
test_fc_windows = build_windows(91, 97)
pooled_fc_D, ps_fc_D = eval_forecasting(model_D_rt, test_fc_windows, DEVICE)
ps_fc_D.to_parquet(os.path.join(RESULTS_DIR, 'pinn_seq_D_test_per_series.parquet'),
                    index=False)
print(f"    WAPE_in pooled: {pooled_fc_D['wape_instock']:.4f}")
print(f"    WAPE_in median: {ps_fc_D['wape_instock'].median():.4f}")
print(f"    WPE_in pooled:  {pooled_fc_D['wpe_instock']:.4f}")
print(f"    WPE_in median:  {ps_fc_D['wpe_instock'].median():.4f}")

print(f"\n  Total Model D time: {time.time()-t_d:.0f}s")

del model_D, model_D_rt, retrain_windows_D
if DEVICE == 'mps':
    torch.mps.empty_cache()


# =========================================================================
# 12. Run Model B (PINN)
# =========================================================================
print("\n\n" + "=" * 72)
print("  MODEL B — PINN SEQUENTIAL (L_data + L_boundary + L_cons + ALM)")
print("=" * 72)
t_b = time.time()

torch.manual_seed(SEED)
model_B = PINNSequential(N_CONT_FEATS, EMB_DIMS, CARDINALITIES,
                          D_MODEL, N_LAYERS, N_HEADS, D_FF, DROPOUT)

# Train
print("\n  Training Model B...")
best_wape_B, total_epochs_B, best_alm_B = train_pinn(
    model_B, train_windows, val_windows, DEVICE,
    WARMUP_EPOCHS, ALM_MAX_ITER, ALM_INNER_EPOCHS, ALM_PATIENCE,
    LR, RHO_INIT, RHO_GAMMA)
print(f"  Best: ALM iter={best_alm_B}, total_epochs={total_epochs_B}, "
      f"val WAPE_in={best_wape_B:.6f}")
print(f"  Training time: {time.time()-t_b:.0f}s")

torch.save(model_B.state_dict(), os.path.join(RESULTS_DIR, 'pinn_seq_B.pt'))

# Val recovery
print("\n  Val recovery (MNAR val, seed=42):")
rec_val_B = eval_recovery_mnar(model_B, val_mnar_windows, DEVICE)
print(f"    WAPE_recovery: {rec_val_B['wape_pool']:.4f}")
print(f"    WPE_recovery:  {rec_val_B['wpe_pool']:.4f}")

# Retrain
print(f"\n  Retraining Model B on days 1-90...")
t_rt2 = time.time()
retrain_windows_B = build_windows(1, 90)

torch.manual_seed(SEED)
model_B_rt = PINNSequential(N_CONT_FEATS, EMB_DIMS, CARDINALITIES,
                              D_MODEL, N_LAYERS, N_HEADS, D_FF, DROPOUT)
retrain_pinn_fixed(model_B_rt, retrain_windows_B, DEVICE, LR,
                    WARMUP_EPOCHS, best_alm_B, ALM_INNER_EPOCHS,
                    RHO_INIT, RHO_GAMMA)
print(f"  Retrain time: {time.time()-t_rt2:.0f}s")

torch.save(model_B_rt.state_dict(), os.path.join(RESULTS_DIR, 'pinn_seq_B_retrained.pt'))

# Test recovery
print("\n  Test recovery (MNAR test, seed=123):")
rec_test_B = eval_recovery_mnar(model_B_rt, test_mnar_windows, DEVICE)
print(f"    WAPE_recovery: {rec_test_B['wape_pool']:.4f}")
print(f"    WPE_recovery:  {rec_test_B['wpe_pool']:.4f}")
print(f"    N positions:   {rec_test_B['n_positions']:,}")

ps_rec_B = eval_recovery_per_series(model_B_rt, test_mnar_windows, DEVICE)
ps_rec_B.to_parquet(os.path.join(RESULTS_DIR, 'pinn_seq_B_recovery_per_series.parquet'),
                     index=False)
print(f"    WAPE_recovery median: {ps_rec_B['wape_recovery'].median():.4f}")
print(f"    WPE_recovery median:  {ps_rec_B['wpe_recovery'].median():.4f}")

# Test forecasting
print("\n  Test forecasting (eval HF, gg 91-97):")
pooled_fc_B, ps_fc_B = eval_forecasting(model_B_rt, test_fc_windows, DEVICE)
ps_fc_B.to_parquet(os.path.join(RESULTS_DIR, 'pinn_seq_B_test_per_series.parquet'),
                    index=False)
print(f"    WAPE_in pooled: {pooled_fc_B['wape_instock']:.4f}")
print(f"    WAPE_in median: {ps_fc_B['wape_instock'].median():.4f}")
print(f"    WPE_in pooled:  {pooled_fc_B['wpe_instock']:.4f}")
print(f"    WPE_in median:  {ps_fc_B['wpe_instock'].median():.4f}")

print(f"\n  Total Model B time: {time.time()-t_b:.0f}s")


# =========================================================================
# 13. Confronto finale
# =========================================================================
print("\n\n" + "=" * 72)
print("  CONFRONTO FINALE")
print("=" * 72)

# Recovery
print(f"\n  --- TRACCIA A: Recovery (MNAR test, seed=123) ---")
print(f"  {'Modello':<20} {'WAPE_pool':>12} {'WAPE_med':>12} "
      f"{'WPE_pool':>12} {'WPE_med':>12}")
print("  " + "-" * 60)

# Load imputation results for comparison
for label, prefix in [('LGB Imputer', 'imputation_lgb'),
                       ('Cond Mean', 'imputation_cond_mean'),
                       ('Global Mean', 'imputation_global_mean')]:
    path = os.path.join(RESULTS_DIR, f'{prefix}_test_per_series.parquet')
    if os.path.exists(path):
        ps = pd.read_parquet(path)
        print(f"  {label:<20} {'—':>12} {ps['wape_recovery'].median():>12.4f} "
              f"{'—':>12} {ps['wpe_recovery'].median():>12.4f}")

print(f"  {'Model D (vanilla)':<20} {rec_test_D['wape_pool']:>12.4f} "
      f"{ps_rec_D['wape_recovery'].median():>12.4f} "
      f"{rec_test_D['wpe_pool']:>12.4f} "
      f"{ps_rec_D['wpe_recovery'].median():>12.4f}")
print(f"  {'Model B (PINN)':<20} {rec_test_B['wape_pool']:>12.4f} "
      f"{ps_rec_B['wape_recovery'].median():>12.4f} "
      f"{rec_test_B['wpe_pool']:>12.4f} "
      f"{ps_rec_B['wpe_recovery'].median():>12.4f}")

# Forecasting
print(f"\n  --- TRACCIA B: Forecasting (eval HF, gg 91-97) ---")
print(f"  {'Modello':<20} {'WAPE_in_pool':>14} {'WAPE_in_med':>14} "
      f"{'WPE_in_pool':>14} {'WPE_in_med':>14}")
print("  " + "-" * 76)

# Load Fase A results
for label, prefix in [('MLP F (dirty)', 'mlp_f'), ('LGB F (dirty)', 'lgb_f'),
                       ('DoW Mean (dirty)', 'dow_mean')]:
    path = os.path.join(RESULTS_DIR, f'{prefix}_test_per_series.parquet')
    if os.path.exists(path):
        ps = pd.read_parquet(path)
        print(f"  {label:<20} {'—':>14} {ps['wape_instock'].median():>14.4f} "
              f"{'—':>14} {ps['wpe_instock'].median():>14.4f}")

# Clean
for label, prefix in [('MLP F (clean)', 'clean_mlp_f'), ('LGB F (clean)', 'clean_lgb_f')]:
    path = os.path.join(RESULTS_DIR, f'{prefix}_test_per_series.parquet')
    if os.path.exists(path):
        ps = pd.read_parquet(path)
        print(f"  {label:<20} {'—':>14} {ps['wape_instock'].median():>14.4f} "
              f"{'—':>14} {ps['wpe_instock'].median():>14.4f}")

print(f"  {'Model D (vanilla)':<20} {pooled_fc_D['wape_instock']:>14.4f} "
      f"{ps_fc_D['wape_instock'].median():>14.4f} "
      f"{pooled_fc_D['wpe_instock']:>14.4f} "
      f"{ps_fc_D['wpe_instock'].median():>14.4f}")
print(f"  {'Model B (PINN)':<20} {pooled_fc_B['wape_instock']:>14.4f} "
      f"{ps_fc_B['wape_instock'].median():>14.4f} "
      f"{pooled_fc_B['wpe_instock']:>14.4f} "
      f"{ps_fc_B['wpe_instock'].median():>14.4f}")

print(f"\n  Tempo totale: {time.time()-t0:.0f}s")
print("=" * 72)
