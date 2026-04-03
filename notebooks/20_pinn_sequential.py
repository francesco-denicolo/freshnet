"""
20_pinn_sequential.py — PINN Sequenziale: Transformer Encoder per Demand Recovery
==================================================================================
Fase 1 di CLAUDE_SEQUENTIAL.md.

Architettura: Transformer encoder bidirezionale con due teste (D*, I*).
Il Transformer processa l'intera sequenza di T=168 ore e produce D*(t) e I*(t)
per ogni timestep. stock_status NON è un input (usato solo nella loss).

Due modalità:
- Model D (MODE='vanilla'): solo L_data, lower bound / ablation
- Model B (MODE='pinn'):    L_data + L_boundary + L_cons + ALM

Valutazione: Traccia A — Recovery su posizioni MNAR (maschere seed=42).

Eseguire con: freshnet/bin/python notebooks/20_pinn_sequential.py
"""

import os
import sys
import gc
import math
import time
import functools
import numpy as np
import pandas as pd

print = functools.partial(print, flush=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =========================================================================
# 1. CONFIG
# =========================================================================
MODE = 'pinn'  # 'vanilla' (Model D) or 'pinn' (Model B)

SEED = 42
MAX_DAY = 83
TRAIN_DAYS = 76
WINDOW_DAYS = 7
WINDOW_HOURS = WINDOW_DAYS * 24  # 168

# Transformer hyperparameters
D_MODEL = 64
N_LAYERS = 2
N_HEADS = 4
D_FF = 128
DROPOUT = 0.1

# Training
BATCH_SIZE = 512
LR = 1e-3

# Embeddings (per-series, broadcast over T)
EMB_DIMS = {'store_id': 32, 'product_id': 32, 'city_id': 8}
CARDINALITIES = {'store_id': 898, 'product_id': 865, 'city_id': 18}

# Continuous covariates (daily, repeated hourly)
CONT_COLS = ['discount', 'avg_temperature', 'avg_humidity',
             'precpt', 'avg_wind_level', 'holiday_flag', 'activity_flag']

# ALM hyperparameters (only used in 'pinn' mode)
WARMUP_EPOCHS = 3
K_INNER = 3          # epochs per ALM iteration
N_OUTER = 15         # max ALM iterations
ALM_PATIENCE = 5
RHO_INIT = 1.0
GAMMA = 2.0

# Vanilla training
MAX_EPOCHS_VANILLA = 50
PATIENCE_VANILLA = 10

# Subsample for debugging (None = all 50K)
N_SERIES_SUBSAMPLE = 5000

# Prediction
PREDICT_CHUNK = 2000

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Device
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

torch.manual_seed(SEED)
np.random.seed(SEED)

# Per-timestep feature count: S_obs + hour + dow + 7 cont = 10
# stock_status NOT an input (only used in loss) — see CLAUDE.md design decision
N_PER_STEP = 1 + 2 + len(CONT_COLS)  # 10

print("=" * 70)
print(f"FASE 1 — PINN Sequenziale: Model {'D (Vanilla)' if MODE == 'vanilla' else 'B (PINN)'}")
print("=" * 70)
print(f"  Device:      {DEVICE}")
print(f"  Mode:        {MODE}")
print(f"  Window:      {WINDOW_DAYS} giorni ({WINDOW_HOURS} ore)")
print(f"  d_model:     {D_MODEL}, layers: {N_LAYERS}, heads: {N_HEADS}")
print(f"  Batch size:  {BATCH_SIZE}")
print(f"  Subsample:   {N_SERIES_SUBSAMPLE or 'all'}")

# =========================================================================
# 2. DATA LOADING
# =========================================================================
t0 = time.time()
print("\n1. Caricamento dati...")

df = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df['dt_parsed'] = pd.to_datetime(df['dt'])
min_date = df['dt_parsed'].min()
df['day_num'] = (df['dt_parsed'] - min_date).dt.days + 1
df['dow'] = df['dt_parsed'].dt.dayofweek

df = df[df['day_num'] <= MAX_DAY].copy()
df = df.sort_values(['store_id', 'product_id', 'dt_parsed']).reset_index(drop=True)
N = len(df)
print(f"  Righe: {N:,}")

# Parse hourly arrays
sales_all = np.array(df['hours_sale'].tolist(), dtype=np.float32)
stock_all = np.array(df['hours_stock_status'].tolist(), dtype=np.float32)

# Series info
series_keys = df.groupby(['store_id', 'product_id'], sort=False).first().reset_index()[
    ['store_id', 'product_id', 'city_id']]
n_series_total = len(series_keys)
n_days = MAX_DAY

print(f"  Serie: {n_series_total:,}, Giorni: {n_days}")
assert N == n_series_total * n_days

# Reshape to 3D
sales_3d = sales_all.reshape(n_series_total, n_days, 24)
stock_3d = stock_all.reshape(n_series_total, n_days, 24)

# Covariates (daily) — 7 features
cov_daily = df[CONT_COLS].values.astype(np.float32).reshape(n_series_total, n_days, len(CONT_COLS))
dows_2d = df['dow'].values.reshape(n_series_total, n_days)

# Extract per-series categorical IDs for embeddings
store_ids_arr = series_keys['store_id'].values.astype(np.int64)
product_ids_arr = series_keys['product_id'].values.astype(np.int64)
city_ids_arr = series_keys['city_id'].values.astype(np.int64)

# Normalize covariates from train days only
cont_norms = {}
for ci, col in enumerate(CONT_COLS):
    train_vals = cov_daily[:, :TRAIN_DAYS, ci]
    mu, sig = train_vals.mean(), train_vals.std()
    if sig > 0:
        cov_daily[:, :, ci] = (cov_daily[:, :, ci] - mu) / sig
    cont_norms[col] = (mu, sig)
    print(f"  Normalizzato {col}: mean={mu:.4f}, std={sig:.4f}")

print(f"  Tempo loading: {time.time()-t0:.1f}s")

# =========================================================================
# 2b. LOAD MNAR MASKS
# =========================================================================
print("\n2. Caricamento maschere MNAR...")
mask_df = pd.read_parquet(os.path.join(DATA_DIR, 'mnar_masks.parquet'))
print(f"  Maschere totali: {len(mask_df):,}")

series_keys['series_idx'] = np.arange(n_series_total)
mask_df = mask_df.merge(
    series_keys[['store_id', 'product_id', 'series_idx']],
    on=['store_id', 'product_id'], how='inner')

mask_df['dt_parsed'] = pd.to_datetime(mask_df['dt'])
mask_df['day_idx'] = (mask_df['dt_parsed'] - min_date).dt.days
mask_df = mask_df[(mask_df['day_idx'] >= 0) & (mask_df['day_idx'] < MAX_DAY)].copy()

mnar_3d = np.zeros((n_series_total, n_days, 24), dtype=bool)
gt_3d = np.zeros((n_series_total, n_days, 24), dtype=np.float32)
si = mask_df['series_idx'].values
di = mask_df['day_idx'].values
hi = mask_df['hour'].values.astype(np.int32)
mnar_3d[si, di, hi] = True
gt_3d[si, di, hi] = mask_df['ground_truth'].values.astype(np.float32)

print(f"  MNAR slots: {mnar_3d.sum():,}")
print(f"  MNAR train: {mnar_3d[:, :TRAIN_DAYS, :].sum():,}")
print(f"  MNAR val:   {mnar_3d[:, TRAIN_DAYS:, :].sum():,}")
del mask_df

# =========================================================================
# 3. SUBSAMPLE (optional)
# =========================================================================
if N_SERIES_SUBSAMPLE is not None and N_SERIES_SUBSAMPLE < n_series_total:
    print(f"\n3. Subsampling a {N_SERIES_SUBSAMPLE} serie...")
    rng = np.random.default_rng(SEED)
    sel_idx = rng.choice(n_series_total, N_SERIES_SUBSAMPLE, replace=False)
    sel_idx.sort()

    sales_3d = sales_3d[sel_idx]
    stock_3d = stock_3d[sel_idx]
    mnar_3d = mnar_3d[sel_idx]
    gt_3d = gt_3d[sel_idx]
    cov_daily = cov_daily[sel_idx]
    dows_2d = dows_2d[sel_idx]
    store_ids_arr = store_ids_arr[sel_idx]
    product_ids_arr = product_ids_arr[sel_idx]
    city_ids_arr = city_ids_arr[sel_idx]
    series_keys = series_keys.iloc[sel_idx].reset_index(drop=True)
    n_series = N_SERIES_SUBSAMPLE
else:
    n_series = n_series_total
    print(f"\n3. Usando tutte le {n_series} serie")

# =========================================================================
# 4. WINDOW CONSTRUCTION
# =========================================================================
print("\n4. Creazione finestre sequenziali...")

hour_pattern = np.tile(np.arange(24, dtype=np.float32) / 23.0, WINDOW_DAYS)


def build_sequential_windows(day_start, day_end, label=""):
    """Build windowed dataset for sequential Transformer.

    Key difference from nb16: S_obs has zeros (not NaN) at stockout+MNAR.
    stock_status NOT in input (only in loss masks). No NaN in the input.
    """
    n_days_range = day_end - day_start
    n_complete = n_days_range // WINDOW_DAYS
    remaining = n_days_range - n_complete * WINDOW_DAYS
    n_win = n_complete + (1 if remaining > 0 else 0)
    total = n_series * n_win

    x_seq = np.zeros((total, WINDOW_HOURS, N_PER_STEP), dtype=np.float32)
    cat_ids = np.zeros((total, 3), dtype=np.int64)
    w_stock = np.zeros((total, WINDOW_HOURS), dtype=np.float32)
    w_mnar = np.zeros((total, WINDOW_HOURS), dtype=bool)
    w_gt = np.zeros((total, WINDOW_HOURS), dtype=np.float32)
    w_valid = np.zeros((total, WINDOW_HOURS), dtype=np.float32)
    w_meta = np.zeros((total, 2), dtype=np.int32)

    for w in range(n_win):
        sd = day_start + w * WINDOW_DAYS
        if w < n_complete:
            ed = sd + WINDOW_DAYS
            actual_days = WINDOW_DAYS
        else:
            ed = day_end
            actual_days = remaining

        actual_hours = actual_days * 24
        ws = w * n_series
        we = (w + 1) * n_series

        sales_w = sales_3d[:, sd:ed, :].reshape(n_series, actual_hours)
        stock_w = stock_3d[:, sd:ed, :].reshape(n_series, actual_hours)
        mnar_w = mnar_3d[:, sd:ed, :].reshape(n_series, actual_hours)
        gt_w = gt_3d[:, sd:ed, :].reshape(n_series, actual_hours)

        # Modified stock: original stockout OR MNAR masked
        stock_modified = np.where((stock_w == 1) | mnar_w, 1.0, 0.0)

        # S_obs: zero at stockout+MNAR, real value at in-stock
        sales_input = np.where(stock_modified == 1, 0.0, sales_w)

        # Feature 0: S_obs (zeros at stockout+MNAR)
        x_seq[ws:we, :actual_hours, 0] = sales_input
        # stock_status NOT in input — only in w_stock for loss masks
        # Feature 1: hour_norm (0-1)
        x_seq[ws:we, :actual_hours, 1] = hour_pattern[:actual_hours]
        # Feature 2: dow_norm (0-1)
        dow_w = dows_2d[:, sd:ed]
        dow_hourly = np.repeat(dow_w, 24, axis=1).astype(np.float32) / 6.0
        x_seq[ws:we, :actual_hours, 2] = dow_hourly
        # Features 3-9: continuous covariates
        cov_w = cov_daily[:, sd:ed, :]
        cov_hourly = np.repeat(cov_w, 24, axis=1)
        x_seq[ws:we, :actual_hours, 3:3 + len(CONT_COLS)] = cov_hourly

        # Categorical IDs for embeddings
        cat_ids[ws:we, 0] = store_ids_arr
        cat_ids[ws:we, 1] = product_ids_arr
        cat_ids[ws:we, 2] = city_ids_arr

        # Tracking
        w_stock[ws:we, :actual_hours] = stock_modified
        w_mnar[ws:we, :actual_hours] = mnar_w
        w_gt[ws:we, :actual_hours] = gt_w
        w_valid[ws:we, :actual_hours] = 1.0
        w_meta[ws:we, 0] = np.arange(n_series)
        w_meta[ws:we, 1] = w

    if label:
        n_valid = w_valid.sum()
        n_stock = (w_stock[w_valid > 0] == 1).sum()
        print(f"  {label}: {n_win} win/serie, {total:,} totali, "
              f"stockout+MNAR={n_stock / n_valid * 100:.1f}%, "
              f"MNAR={w_mnar.sum():,}")

    return {
        'x_seq': x_seq, 'cat_ids': cat_ids, 'stock': w_stock,
        'mnar': w_mnar, 'gt': w_gt, 'valid': w_valid, 'meta': w_meta,
    }


t_win = time.time()
train_data = build_sequential_windows(0, TRAIN_DAYS, f"Train (gg 1-{TRAIN_DAYS})")
val_data = build_sequential_windows(TRAIN_DAYS, n_days, f"Val (gg {TRAIN_DAYS + 1}-{MAX_DAY})")

# Full data for evaluation
all_data = {}
for k in train_data:
    all_data[k] = np.concatenate([train_data[k], val_data[k]], axis=0)

print(f"  Costruzione: {time.time()-t_win:.1f}s")
print(f"  x_seq train: {train_data['x_seq'].shape}, val: {val_data['x_seq'].shape}")
print(f"  MNAR totali: {all_data['mnar'].sum():,}")
print(f"  Memory x_seq all: {all_data['x_seq'].nbytes / 1024**3:.2f} GB")

# =========================================================================
# 5. DATASET & DATALOADER
# =========================================================================


class SequentialDataset(Dataset):
    def __init__(self, data):
        self.x_seq = torch.from_numpy(data['x_seq'])
        self.cat_ids = torch.from_numpy(data['cat_ids'])
        self.stock = torch.from_numpy(data['stock'])
        self.valid = torch.from_numpy(data['valid'])

    def __len__(self):
        return len(self.x_seq)

    def __getitem__(self, idx):
        return self.x_seq[idx], self.cat_ids[idx], self.stock[idx], self.valid[idx]


# =========================================================================
# 6. MODEL ARCHITECTURE
# =========================================================================
print("\n5. Architettura del modello...")


class SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class PINNSequential(nn.Module):
    """Transformer encoder with dual heads for demand (D*) and inventory (I*)."""

    def __init__(self, n_per_step, emb_dims, cardinalities,
                 d_model, n_layers, n_heads, d_ff, dropout):
        super().__init__()

        # Embeddings (per-series, broadcast over T)
        self.emb_store = nn.Embedding(cardinalities['store_id'], emb_dims['store_id'])
        self.emb_product = nn.Embedding(cardinalities['product_id'], emb_dims['product_id'])
        self.emb_city = nn.Embedding(cardinalities['city_id'], emb_dims['city_id'])

        total_emb = sum(emb_dims.values())
        input_dim = n_per_step + total_emb

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_enc = SinusoidalPE(d_model, max_len=512)

        # Transformer encoder (bidirectional, no causal mask)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Head D: demand latent (> 0 via Softplus)
        self.head_D = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, 1), nn.Softplus())

        # Head I: inventory latent (>= 0 via Softplus)
        self.head_I = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, 1), nn.Softplus())

    def forward(self, x_seq, cat_ids):
        # x_seq: (B, T, n_per_step), cat_ids: (B, 3)
        B, T, _ = x_seq.shape

        # Embeddings → broadcast over T
        e_s = self.emb_store(cat_ids[:, 0])      # (B, 32)
        e_p = self.emb_product(cat_ids[:, 1])     # (B, 32)
        e_c = self.emb_city(cat_ids[:, 2])        # (B, 8)
        emb = torch.cat([e_s, e_p, e_c], dim=1)  # (B, 72)
        emb = emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, 72)

        # Concatenate features + embeddings
        x = torch.cat([x_seq, emb], dim=2)  # (B, T, 83)

        # Project → PE → Transformer
        x = self.input_proj(x)   # (B, T, d_model)
        x = self.pos_enc(x)
        h = self.transformer(x)  # (B, T, d_model)

        # Output heads
        D_star = self.head_D(h).squeeze(-1)  # (B, T)
        I_star = self.head_I(h).squeeze(-1)  # (B, T)
        return D_star, I_star


model = PINNSequential(
    n_per_step=N_PER_STEP, emb_dims=EMB_DIMS, cardinalities=CARDINALITIES,
    d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
    d_ff=D_FF, dropout=DROPOUT)

n_params = sum(p.numel() for p in model.parameters())
print(f"  PINNSequential: {n_params:,} parametri")
print(f"  Input: ({N_PER_STEP} per-step + {sum(EMB_DIMS.values())} emb) = "
      f"{N_PER_STEP + sum(EMB_DIMS.values())} → d_model={D_MODEL}")

# =========================================================================
# 7. LOSS FUNCTION
# =========================================================================


def compute_pinn_loss(D_star, I_star, x_seq, stock, valid,
                      lambda_b, lambda_c, rho_b, rho_c):
    """PINN loss on sequential data.

    L_ALM = L_data + λ_b·V_b + (ρ_b/2)·Q_b + λ_c·V_c + (ρ_c/2)·Q_c
    """
    targets = x_seq[:, :, 0]  # S_obs

    in_mask = (stock == 0) & (valid > 0)
    so_mask = (stock == 1) & (valid > 0)

    # L_data: MSE on in-stock positions only
    n_in = in_mask.sum()
    if n_in > 0:
        L_data = (D_star[in_mask] - targets[in_mask]).pow(2).mean()
    else:
        L_data = torch.tensor(0.0, device=D_star.device)

    # L_boundary
    n_so = so_mask.sum()
    if n_so > 0:
        V_b1 = I_star[so_mask].mean()
        Q_b1 = I_star[so_mask].pow(2).mean()
    else:
        V_b1 = Q_b1 = torch.tensor(0.0, device=D_star.device)

    if n_in > 0:
        gap = F.relu(D_star[in_mask] - I_star[in_mask])
        V_b2 = gap.mean()
        Q_b2 = gap.pow(2).mean()
    else:
        V_b2 = Q_b2 = torch.tensor(0.0, device=D_star.device)

    V_b = V_b1 + V_b2
    Q_b = Q_b1 + Q_b2

    # L_cons: conservation over T-1 consecutive pairs (includes cross-day)
    min_DI = torch.min(D_star[:, :-1], I_star[:, :-1])
    delta_I = I_star[:, 1:] - I_star[:, :-1]
    implicit_R = delta_I + min_DI
    neg_R = F.relu(-implicit_R)

    valid_pairs = (valid[:, :-1] > 0) & (valid[:, 1:] > 0)
    n_vp = valid_pairs.sum()
    if n_vp > 0:
        V_c = neg_R[valid_pairs].mean()
        Q_c = neg_R[valid_pairs].pow(2).mean()
    else:
        V_c = Q_c = torch.tensor(0.0, device=D_star.device)

    # ALM loss
    L_total = (L_data
               + lambda_b * V_b + (rho_b / 2.0) * Q_b
               + lambda_c * V_c + (rho_c / 2.0) * Q_c)

    return L_total, L_data.item(), V_b.item(), V_c.item()


# =========================================================================
# 8. TRAINING
# =========================================================================
print(f"\n6. Training ({MODE})...")

model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train_ds = SequentialDataset(train_data)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=False)


def predict_sequential(model, data, device, chunk_size=PREDICT_CHUNK):
    """Chunked prediction. Returns D_pred, I_pred as numpy arrays."""
    model.eval()
    x_all = torch.from_numpy(data['x_seq'])
    c_all = torch.from_numpy(data['cat_ids'])
    N_total = len(x_all)
    all_D, all_I = [], []
    with torch.no_grad():
        for s in range(0, N_total, chunk_size):
            e = min(s + chunk_size, N_total)
            x_b = x_all[s:e].to(device)
            c_b = c_all[s:e].to(device)
            D, I = model(x_b, c_b)
            all_D.append(D.cpu().numpy())
            all_I.append(I.cpu().numpy())
    return np.concatenate(all_D, 0), np.concatenate(all_I, 0)


def compute_val_mae_mnar(model, val_data, device):
    """MAE on MNAR positions of val set (same criterion as paper baselines)."""
    D_pred, _ = predict_sequential(model, val_data, device)
    mnar = val_data['mnar']
    gt = val_data['gt']
    mask = mnar > 0
    if mask.sum() == 0:
        return float('inf')
    return np.abs(D_pred[mask] - gt[mask]).mean()


def run_epoch(model, train_loader, optimizer, device,
              lam_b, lam_c, r_b, r_c):
    """Run one training epoch."""
    model.train()
    sum_loss, sum_ld, sum_vb, sum_vc, nb = 0, 0, 0, 0, 0
    for x_seq, cat_ids, stock, valid in train_loader:
        x_seq = x_seq.to(device)
        cat_ids = cat_ids.to(device)
        stock = stock.to(device)
        valid = valid.to(device)

        D_star, I_star = model(x_seq, cat_ids)
        L_total, ld, vb, vc = compute_pinn_loss(
            D_star, I_star, x_seq, stock, valid,
            lam_b, lam_c, r_b, r_c)

        optimizer.zero_grad()
        L_total.backward()
        optimizer.step()

        sum_loss += L_total.item()
        sum_ld += ld
        sum_vb += vb
        sum_vc += vc
        nb += 1

    return sum_loss / nb, sum_ld / nb, sum_vb / nb, sum_vc / nb


def evaluate_constraints(model, train_data, device, max_samples=10000,
                         chunk_size=PREDICT_CHUNK):
    """Evaluate constraint violations on a subset of training data (chunked)."""
    model.eval()
    n = len(train_data['x_seq'])
    idx = np.random.choice(n, min(max_samples, n), replace=False)

    sum_ld, sum_vb, sum_vc, n_chunks = 0, 0, 0, 0
    with torch.no_grad():
        for s in range(0, len(idx), chunk_size):
            e = min(s + chunk_size, len(idx))
            batch_idx = idx[s:e]
            x = torch.from_numpy(train_data['x_seq'][batch_idx]).to(device)
            c = torch.from_numpy(train_data['cat_ids'][batch_idx]).to(device)
            st = torch.from_numpy(train_data['stock'][batch_idx]).to(device)
            v = torch.from_numpy(train_data['valid'][batch_idx]).to(device)

            D_star, I_star = model(x, c)
            _, ld, vb, vc = compute_pinn_loss(
                D_star, I_star, x, st, v, 0.0, 0.0, 0.0, 0.0)
            sum_ld += ld
            sum_vb += vb
            sum_vc += vc
            n_chunks += 1

    return sum_vb / n_chunks, sum_vc / n_chunks, sum_ld / n_chunks


# --- Training loop ---
t_train = time.time()
best_val_mae = float('inf')
best_state = None
best_info = {}
total_epochs = 0

if MODE == 'vanilla':
    # ---- Model D: simple epoch loop with early stopping ----
    no_improve = 0
    for epoch in range(1, MAX_EPOCHS_VANILLA + 1):
        total_epochs += 1
        avg_loss, avg_ld, _, _ = run_epoch(
            model, train_loader, optimizer, DEVICE, 0.0, 0.0, 0.0, 0.0)
        val_mae = compute_val_mae_mnar(model, val_data, DEVICE)
        print(f"  Epoch {epoch:3d}: L_data={avg_ld:.6f}, val_MAE_mnar={val_mae:.6f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_info = {'epoch': epoch, 'val_mae': val_mae, 'mode': 'vanilla'}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= PATIENCE_VANILLA:
            print(f"  Early stopping a epoca {epoch} (best: {best_info['epoch']})")
            break

else:
    # ---- Model B: warmup + ALM ----
    lambda_b, lambda_c = 0.0, 0.0
    rho_b, rho_c = RHO_INIT, RHO_INIT

    # Phase 1: Warmup
    print("  --- Warmup (solo L_data) ---")
    for epoch in range(1, WARMUP_EPOCHS + 1):
        total_epochs += 1
        avg_loss, avg_ld, _, _ = run_epoch(
            model, train_loader, optimizer, DEVICE, 0.0, 0.0, 0.0, 0.0)
        val_mae = compute_val_mae_mnar(model, val_data, DEVICE)
        print(f"    Warmup {epoch}: L_data={avg_ld:.6f}, val_MAE_mnar={val_mae:.6f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_info = {'epoch': total_epochs, 'phase': 'warmup',
                         'val_mae': val_mae, 'mode': 'pinn'}

    # Phase 2: ALM iterations
    print("  --- ALM ---")
    V_b_prev, V_c_prev = float('inf'), float('inf')
    alm_no_improve = 0

    for alm_iter in range(1, N_OUTER + 1):
        # Primal step: K_INNER epochs
        for inner in range(1, K_INNER + 1):
            total_epochs += 1
            run_epoch(model, train_loader, optimizer, DEVICE,
                      lambda_b, lambda_c, rho_b, rho_c)

        # Evaluate constraints
        V_b_eval, V_c_eval, L_data_eval = evaluate_constraints(
            model, train_data, DEVICE)
        val_mae = compute_val_mae_mnar(model, val_data, DEVICE)

        print(f"    ALM {alm_iter:2d} (ep {total_epochs:3d}): "
              f"L_data={L_data_eval:.6f}, V_b={V_b_eval:.5f}, "
              f"V_c={V_c_eval:.5f}, lam_b={lambda_b:.3f}, lam_c={lambda_c:.3f}, "
              f"rho_b={rho_b:.1f}, rho_c={rho_c:.1f}, val_MAE={val_mae:.6f}")

        # Dual step
        lambda_b = max(0.0, lambda_b + rho_b * V_b_eval)
        lambda_c = max(0.0, lambda_c + rho_c * V_c_eval)

        # Adaptation
        if V_b_eval > 0.25 * V_b_prev and V_b_eval > 1e-6:
            rho_b *= GAMMA
        if V_c_eval > 0.25 * V_c_prev and V_c_eval > 1e-6:
            rho_c *= GAMMA

        V_b_prev, V_c_prev = V_b_eval, V_c_eval

        # Best model tracking
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_info = {
                'epoch': total_epochs, 'alm_iter': alm_iter,
                'phase': 'alm', 'val_mae': val_mae, 'mode': 'pinn',
                'V_b': V_b_eval, 'V_c': V_c_eval,
                'lambda_b': lambda_b, 'lambda_c': lambda_c,
            }
            alm_no_improve = 0
        else:
            alm_no_improve += 1

        if alm_no_improve >= ALM_PATIENCE:
            print(f"    ALM early stopping a iter {alm_iter} "
                  f"(best: iter {best_info.get('alm_iter', '?')})")
            break

train_time = time.time() - t_train

# Restore best model
if best_state is not None:
    model.load_state_dict(best_state)
model.to(DEVICE)

print(f"\n  Training completato in {train_time:.1f}s ({train_time/60:.1f} min)")
print(f"  Best: epoch {best_info.get('epoch', '?')}, val_MAE_mnar={best_val_mae:.6f}")

# =========================================================================
# 9. EVALUATION — Recovery su posizioni MNAR
# =========================================================================
print(f"\n7. Valutazione recovery MNAR...")

# Move to CPU for prediction (avoid MPS OOM)
if DEVICE.type == 'mps':
    model.to(torch.device('cpu'))
    pred_device = torch.device('cpu')
    torch.mps.empty_cache()
    gc.collect()
else:
    pred_device = DEVICE

D_pred, I_pred = predict_sequential(model, all_data, pred_device)
print(f"  Predizione completata: D_pred {D_pred.shape}")

# Extract MNAR positions
mnar_flat = all_data['mnar'].ravel()
pred_flat = D_pred.ravel()
gt_flat = all_data['gt'].ravel()

mnar_idx = np.where(mnar_flat)[0]
pred_mnar = pred_flat[mnar_idx]
gt_mnar = gt_flat[mnar_idx]
n_mnar = len(mnar_idx)

# Pooled metrics
gt_sum = gt_mnar.sum()
wape = np.abs(pred_mnar - gt_mnar).sum() / gt_sum if gt_sum > 0 else np.nan
wpe = (pred_mnar - gt_mnar).sum() / gt_sum if gt_sum > 0 else np.nan
mae = np.abs(pred_mnar - gt_mnar).mean()

gt_pos = gt_mnar > 0
wape_pos = (np.abs(pred_mnar[gt_pos] - gt_mnar[gt_pos]).sum()
            / gt_mnar[gt_pos].sum()) if gt_pos.sum() > 0 else np.nan
wpe_pos = ((pred_mnar[gt_pos] - gt_mnar[gt_pos]).sum()
           / gt_mnar[gt_pos].sum()) if gt_pos.sum() > 0 else np.nan

print(f"\n  --- Metriche Recovery {MODE} (best epoch {best_info.get('epoch', '?')}) ---")
print(f"  WAPE_recovery (overall): {wape:.4f}  (GT>0: {wape_pos:.4f})")
print(f"  WPE_recovery  (overall): {wpe:.4f}  (GT>0: {wpe_pos:.4f})")
print(f"  MAE:                     {mae:.6f}")
print(f"  N MNAR: {n_mnar:,}, N GT>0: {gt_pos.sum():,}")

# Per-series metrics
mnar_window_idx = mnar_idx // WINDOW_HOURS
mnar_series_idx = all_data['meta'][mnar_window_idx, 0]

eval_df = pd.DataFrame({
    'series_idx': mnar_series_idx,
    'pred': pred_mnar, 'gt': gt_mnar,
    'abs_err': np.abs(pred_mnar - gt_mnar),
    'err': pred_mnar - gt_mnar,
})
agg = eval_df.groupby('series_idx').agg(
    gt_sum=('gt', 'sum'), abs_err_sum=('abs_err', 'sum'),
    err_sum=('err', 'sum'), n_mnar=('gt', 'count'))
agg['wape_recovery'] = np.where(agg['gt_sum'] > 0,
                                agg['abs_err_sum'] / agg['gt_sum'], np.nan)
agg['wpe_recovery'] = np.where(agg['gt_sum'] > 0,
                                agg['err_sum'] / agg['gt_sum'], np.nan)
agg = agg.reset_index()
agg['store_id'] = series_keys.iloc[agg['series_idx'].values]['store_id'].values
agg['product_id'] = series_keys.iloc[agg['series_idx'].values]['product_id'].values

ps_df = agg[['store_id', 'product_id', 'wape_recovery', 'wpe_recovery',
             'n_mnar', 'gt_sum']].copy()

suffix = f"pinn_seq_{MODE}"
ps_path = os.path.join(RESULTS_DIR, f'{suffix}_recovery_per_series.parquet')
ps_df.to_parquet(ps_path, index=False)

wape_med = ps_df['wape_recovery'].median()
wpe_med = ps_df['wpe_recovery'].median()

print(f"  WAPE_recovery mediana: {wape_med:.4f}")
print(f"  WPE_recovery mediana:  {wpe_med:.4f}")
print(f"  Salvato: {ps_path}")

# Constraint metrics (PINN only)
if MODE == 'pinn':
    stock_flat = all_data['stock'].ravel()
    valid_flat = all_data['valid'].ravel()
    i_flat = I_pred.ravel()

    so_valid = (stock_flat == 1) & (valid_flat > 0)
    in_valid = (stock_flat == 0) & (valid_flat > 0)
    v_b_so = i_flat[so_valid].mean() if so_valid.sum() > 0 else 0
    gap_in = np.maximum(0, pred_flat[in_valid] - i_flat[in_valid])
    v_b_gap = gap_in.mean() if in_valid.sum() > 0 else 0
    print(f"\n  Constraint V_boundary: {v_b_so + v_b_gap:.6f} "
          f"(I*_stockout={v_b_so:.6f}, gap_instock={v_b_gap:.6f})")

    # Conservation
    D2d = D_pred.reshape(-1, WINDOW_HOURS)
    I2d = I_pred.reshape(-1, WINDOW_HOURS)
    V2d = all_data['valid'].reshape(-1, WINDOW_HOURS)
    minDI = np.minimum(D2d[:, :-1], I2d[:, :-1])
    dI = I2d[:, 1:] - I2d[:, :-1]
    negR = np.maximum(0, -(dI + minDI))
    vp = (V2d[:, :-1] > 0) & (V2d[:, 1:] > 0)
    v_cons = negR[vp].mean() if vp.sum() > 0 else 0
    print(f"  Constraint V_conservation: {v_cons:.6f}")

# =========================================================================
# 10. COMPARISON TABLE
# =========================================================================
print(f"\n{'='*70}")
print("8. TABELLA CONFRONTO — Demand Recovery (Traccia A)")
print(f"{'='*70}")

# Load existing results
comp_models = []
for fname in os.listdir(RESULTS_DIR):
    if fname.endswith('_recovery_per_series.parquet'):
        name = fname.replace('_recovery_per_series.parquet', '')
        try:
            df_r = pd.read_parquet(os.path.join(RESULTS_DIR, fname))
            comp_models.append({
                'name': name,
                'wape_med': df_r['wape_recovery'].median(),
                'wpe_med': df_r['wpe_recovery'].median(),
                'n_series': len(df_r),
            })
        except Exception:
            pass

# Add current model
comp_models.append({
    'name': suffix,
    'wape_med': wape_med,
    'wpe_med': wpe_med,
    'n_series': len(ps_df),
})

# Sort by WAPE median
comp_models.sort(key=lambda x: x['wape_med'])

# Deduplicate
seen = set()
unique_models = []
for m in comp_models:
    if m['name'] not in seen:
        seen.add(m['name'])
        unique_models.append(m)

print(f"\n{'Model':<30} {'WAPE_med':>10} {'WPE_med':>10} {'N_series':>10}")
print(f"{'-'*30} {'-'*10} {'-'*10} {'-'*10}")
for m in unique_models:
    print(f"{m['name']:<30} {m['wape_med']:10.4f} {m['wpe_med']:10.4f} {m['n_series']:10d}")

total_time = time.time() - t0
print(f"\nTempo totale: {total_time / 60:.1f} min")
print("=" * 70)
