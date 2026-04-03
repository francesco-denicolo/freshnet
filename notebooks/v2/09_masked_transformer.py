"""
09_masked_transformer.py — Fase 2: Masked Transformer (Models A & C)
=====================================================================
Piano: CLAUDE_SEQUENTIAL-2.md, Fase 2

Architecture: Same as 08_pinn_sequential.py BUT with input masking.
At stockout positions, S_obs is replaced with a learnable mask token,
forcing the Transformer to infer D*(t) from surrounding context via
bidirectional self-attention.

Why masking is needed:
  Models D and B (Fase 1) showed WAPE_recovery=1.0 (zero recovery)
  because the Transformer copies S_obs from input. When S_obs=0 at
  stockout/MNAR positions, D*=0 regardless of physics constraints.
  Masking breaks this shortcut.

Additionally, during training we randomly mask a fraction of in-stock
positions (BERT-style) to prevent the model from learning a
"if visible -> copy, if masked -> infer" dichotomy. This teaches
the model to always reconstruct from context.

Models:
  A (masked):      L_data only + input masking
  C (PINN+masked): L_data + L_boundary + L_cons + ALM + input masking

Workflow:
  1. TUNING:     Train gg 1-83, Val gg 84-90
  2. RETRAINING: Retrain su gg 1-90
  3. TRACCIA A:  Recovery su MNAR test masks (seed=123, gg 1-90)
  4. TRACCIA B:  Forecasting su eval HF (gg 91-97)

Execute: freshnet/bin/python notebooks/v2/09_masked_transformer.py
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

# Architecture (same as 08)
T_WINDOW = 168        # 7 days x 24 hours
D_MODEL = 64
N_LAYERS = 2
N_HEADS = 4
D_FF = 128
DROPOUT = 0.1
EMB_DIMS = {'store_id': 32, 'product_id': 32, 'city_id': 8}
CARDINALITIES = {'store_id': 898, 'product_id': 865, 'city_id': 18}

# Masking
TRAIN_MASK_RATE = 0.15  # Randomly mask 15% of in-stock positions during training

# Training
BATCH_SIZE = 512
LR = 1e-3

# Model A (masked, no physics)
MAX_EPOCHS_A = 50
PATIENCE_A = 10

# Model C (PINN+masked)
WARMUP_EPOCHS = 3
ALM_MAX_ITER = 15
ALM_INNER_EPOCHS = 3
ALM_PATIENCE = 5
RHO_INIT = 1.0
RHO_GAMMA = 2.0

CONT_COLS = ['discount', 'avg_temperature', 'avg_humidity', 'precpt',
             'avg_wind_level', 'holiday_flag', 'activity_flag']

N_SERIES_SUBSAMPLE = None

t0 = time.time()
print("=" * 72)
print("  FASE 2 — MASKED TRANSFORMER (Models A & C)")
print("=" * 72)
print(f"  Device: {DEVICE}")
print(f"  Window: T={T_WINDOW} ({T_WINDOW//24} days)")
print(f"  d_model={D_MODEL}, layers={N_LAYERS}, heads={N_HEADS}, d_ff={D_FF}")
print(f"  Train mask rate: {TRAIN_MASK_RATE}")


# =========================================================================
# 2. Data loading (identical to 08)
# =========================================================================
print("\n1. Caricamento dati...")
t1 = time.time()

df_train = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df_eval = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_eval.parquet'))
print(f"  Train: {len(df_train):,}, Eval: {len(df_eval):,}")

df_full = pd.concat([df_train, df_eval], ignore_index=True)
print(f"  Full: {len(df_full):,}, {df_full['day_index'].nunique()} giorni")

# Subsample
if N_SERIES_SUBSAMPLE is not None:
    keys = df_full.groupby(['store_id', 'product_id']).size().reset_index()
    keys = keys.sample(N_SERIES_SUBSAMPLE, random_state=SEED)
    mask = df_full.set_index(['store_id', 'product_id']).index.isin(
        keys.set_index(['store_id', 'product_id']).index)
    df_full = df_full[mask].reset_index(drop=True)

# =========================================================================
# 3. Build 3D arrays
# =========================================================================
print("\n2. Building 3D arrays...")
t2 = time.time()

series_keys_df = df_full.groupby(['store_id', 'product_id']).size().reset_index()
series_keys_df = series_keys_df[['store_id', 'product_id']].sort_values(
    ['store_id', 'product_id']).reset_index(drop=True)
series_keys = list(zip(series_keys_df['store_id'], series_keys_df['product_id']))
series_map = {k: i for i, k in enumerate(series_keys)}

n_series = len(series_keys)
min_day = df_full['day_index'].min()
max_day = df_full['day_index'].max()
n_days = max_day - min_day + 1
print(f"  {n_series:,} serie, {n_days} giorni")

# Parse hourly arrays
sales_3d = np.zeros((n_series, n_days, 24), dtype=np.float32)
stock_3d = np.zeros((n_series, n_days, 24), dtype=np.float32)
conts_3d = np.zeros((n_series, n_days, len(CONT_COLS)), dtype=np.float32)

# Pre-compute IDs for embeddings
store_ids = np.array([k[0] for k in series_keys], dtype=np.int64)
product_ids = np.array([k[1] for k in series_keys], dtype=np.int64)
city_ids = np.zeros(n_series, dtype=np.int64)
dows_2d = np.zeros((n_series, n_days), dtype=np.int64)
doms_2d = np.zeros((n_series, n_days), dtype=np.int64)

for _, row in df_full.iterrows():
    key = (row['store_id'], row['product_id'])
    s_idx = series_map.get(key)
    if s_idx is None:
        continue
    d_idx = row['day_index'] - min_day

    # Parse hourly sales
    hs = row['hours_sale']
    if isinstance(hs, str):
        hs = np.array(eval(hs), dtype=np.float32)
    elif isinstance(hs, (list, np.ndarray)):
        hs = np.array(hs, dtype=np.float32)
    sales_3d[s_idx, d_idx] = hs[:24]

    # Parse stock status
    ss = row['hours_stock_status']
    if isinstance(ss, str):
        ss = np.array(eval(ss), dtype=np.float32)
    elif isinstance(ss, (list, np.ndarray)):
        ss = np.array(ss, dtype=np.float32)
    stock_3d[s_idx, d_idx] = ss[:24]

    # Continuous features
    for ci, col in enumerate(CONT_COLS):
        conts_3d[s_idx, d_idx, ci] = row[col]

    # City/dow/dom
    city_ids[s_idx] = row['city_id']
    dows_2d[s_idx, d_idx] = row['dow']
    doms_2d[s_idx, d_idx] = row['dom']

print(f"  sales_3d: {sales_3d.shape}, stock_3d: {stock_3d.shape}")
print(f"  conts_3d: {conts_3d.shape}")
print(f"  Tempo: {time.time()-t2:.1f}s")

del df_train, df_eval, df_full


# =========================================================================
# 4. Normalize continuous features
# =========================================================================
print("\n3. Normalizing continuous features...")
train_days = 83
cont_train = conts_3d[:, :train_days, :]
cont_mean = cont_train.mean(axis=(0, 1))
cont_std = cont_train.std(axis=(0, 1))
cont_std[cont_std < 1e-6] = 1.0
conts_3d_norm = (conts_3d - cont_mean) / cont_std
print(f"  cont_mean: {cont_mean}")
print(f"  cont_std:  {cont_std}")


# =========================================================================
# 5. Load MNAR masks (identical to 08)
# =========================================================================
print("\n4. Loading MNAR masks...")
t3 = time.time()

mnar_val_mask = np.zeros((n_series, n_days, 24), dtype=bool)
mnar_val_gt = np.zeros((n_series, n_days, 24), dtype=np.float32)
mnar_test_mask = np.zeros((n_series, n_days, 24), dtype=bool)
mnar_test_gt = np.zeros((n_series, n_days, 24), dtype=np.float32)

df_val_masks = pd.read_parquet(os.path.join(DATA_DIR, 'mnar_masks_val.parquet'))
df_test_masks = pd.read_parquet(os.path.join(DATA_DIR, 'mnar_masks_test.parquet'))
print(f"  Val masks: {len(df_val_masks):,} (gg 84-90, seed=42)")
print(f"  Test masks: {len(df_test_masks):,} (gg 1-90, seed=123)")

for _, row in df_val_masks.iterrows():
    key = (row['store_id'], row['product_id'])
    s_idx = series_map.get(key)
    if s_idx is None:
        continue
    d_idx = row['day_index'] - min_day
    h = row['hour']
    mnar_val_mask[s_idx, d_idx, h] = True
    mnar_val_gt[s_idx, d_idx, h] = row['true_sales']

n_val_mnar = mnar_val_mask.sum()
print(f"  Val MNAR positions: {n_val_mnar:,}")

for _, row in df_test_masks.iterrows():
    key = (row['store_id'], row['product_id'])
    s_idx = series_map.get(key)
    if s_idx is None:
        continue
    d_idx = row['day_index'] - min_day
    h = row['hour']
    mnar_test_mask[s_idx, d_idx, h] = True
    mnar_test_gt[s_idx, d_idx, h] = row['true_sales']

n_test_mnar = mnar_test_mask.sum()
print(f"  Test MNAR positions: {n_test_mnar:,}")
print(f"  Tempo: {time.time()-t3:.1f}s")

del df_val_masks, df_test_masks


# =========================================================================
# 6. Window construction (identical to 08)
# =========================================================================
def build_windows(day_start, day_end, apply_mnar_mask=None, apply_mnar_gt=None):
    """Build non-overlapping 7-day windows."""
    d0 = day_start - min_day
    d1 = day_end - min_day
    n_avail_days = d1 - d0 + 1
    n_windows_per_series = max(1, n_avail_days // (T_WINDOW // 24))
    N = n_series * n_windows_per_series

    x_seq = np.zeros((N, T_WINDOW, 11), dtype=np.float32)
    cat_ids = np.zeros((N, 3), dtype=np.int64)
    stock_arr = np.zeros((N, T_WINDOW), dtype=np.float32)
    mnar_arr = np.zeros((N, T_WINDOW), dtype=bool)
    gt_arr = np.zeros((N, T_WINDOW), dtype=np.float32)
    valid_arr = np.zeros((N, T_WINDOW), dtype=np.float32)
    meta_arr = np.zeros((N, 2), dtype=np.int64)

    win_idx = 0
    for s in range(n_series):
        for w in range(n_windows_per_series):
            start_d = d0 + w * (T_WINDOW // 24)
            end_d = start_d + (T_WINDOW // 24) - 1

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

                    is_mnar = False
                    if apply_mnar_mask is not None and apply_mnar_mask[s, d_abs, h]:
                        is_mnar = True
                        s_obs = 0.0
                        stk = 1.0

                    x_seq[win_idx, t_idx, 0] = s_obs
                    x_seq[win_idx, t_idx, 1] = h / 23.0
                    x_seq[win_idx, t_idx, 2] = dows_2d[s, d_abs] / 6.0
                    x_seq[win_idx, t_idx, 3] = (doms_2d[s, d_abs] - 1) / 30.0
                    x_seq[win_idx, t_idx, 4:11] = conts_3d_norm[s, d_abs]

                    stock_arr[win_idx, t_idx] = stk
                    valid_arr[win_idx, t_idx] = 1.0

                    if is_mnar:
                        mnar_arr[win_idx, t_idx] = True
                        gt_arr[win_idx, t_idx] = apply_mnar_gt[s, d_abs, h]

            cat_ids[win_idx] = [store_ids[s], product_ids[s], city_ids[s]]
            meta_arr[win_idx] = [s, start_d]
            win_idx += 1

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
t4 = time.time()

train_windows = build_windows(1, 84)
print(f"  Train: {len(train_windows['x_seq']):,} windows (days 1-84)")

val_windows = build_windows(84, 90)
print(f"  Val: {len(val_windows['x_seq']):,} windows (days 84-90)")

val_mnar_windows = build_windows(84, 90,
                                  apply_mnar_mask=mnar_val_mask,
                                  apply_mnar_gt=mnar_val_gt)
n_mnar_val = val_mnar_windows['mnar'].sum()
print(f"  Val MNAR: {len(val_mnar_windows['x_seq']):,} windows, "
      f"{n_mnar_val:,} masked positions")

print(f"  Tempo: {time.time()-t4:.1f}s")


# =========================================================================
# 7. Dataset & Model (MODIFIED: masking)
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


class MaskedPINNSequential(nn.Module):
    """Transformer Encoder with input masking at stockout positions.

    Key difference from PINNSequential (08):
      At stockout positions (stock==1), S_obs (feature 0) is replaced
      with a learnable mask_value. This forces the model to infer D*
      from context via self-attention instead of copying S_obs.

    During training, an additional random fraction of in-stock positions
    can be masked (BERT-style) to prevent the model from learning a
    "if visible -> copy, if masked -> infer" dichotomy.
    """

    def __init__(self, n_cont_feats, emb_dims, cardinalities,
                 d_model, n_layers, n_heads, d_ff, dropout):
        super().__init__()

        # Learnable mask value for S_obs replacement
        self.mask_value = nn.Parameter(torch.tensor(0.0))

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

    def forward(self, x_seq, cat_ids, stock, train_mask_rate=0.0):
        """
        x_seq: (B, T, n_cont_feats)
        cat_ids: (B, 3)
        stock: (B, T) stock status (0=in-stock, 1=stockout)
        train_mask_rate: fraction of in-stock positions to randomly mask (training only)

        Returns:
            D_star: (B, T), I_star: (B, T)
        """
        B, T, _ = x_seq.shape

        # --- Input masking ---
        # Replace S_obs with mask_value at stockout positions
        x_masked = x_seq.clone()
        mask = stock > 0.5  # stockout positions: (B, T)

        # During training: also randomly mask some in-stock positions
        if train_mask_rate > 0 and self.training:
            in_stock = ~mask
            rand_mask = torch.rand(B, T, device=x_seq.device) < train_mask_rate
            extra_mask = in_stock & rand_mask
            mask = mask | extra_mask

        # Replace S_obs (feature 0) with learnable mask_value
        x_masked[:, :, 0] = torch.where(mask, self.mask_value, x_masked[:, :, 0])

        # --- Standard forward ---
        emb_list = [self.embeddings[name](cat_ids[:, i])
                    for i, name in enumerate(self.emb_names)]
        emb_cat = torch.cat(emb_list, dim=1)
        emb_broadcast = emb_cat.unsqueeze(1).expand(-1, T, -1)

        x = torch.cat([x_masked, emb_broadcast], dim=2)

        h = self.input_proj(x)
        h = self.pe(h)
        h = self.encoder(h)

        D_star = self.head_D(h).squeeze(-1)
        I_star = self.head_I(h).squeeze(-1)

        return D_star, I_star


# =========================================================================
# 8. Loss function (same as 08)
# =========================================================================
def compute_loss(D_star, I_star, x_seq, stock, valid, mode='vanilla',
                 lam_b=0.0, rho_b=0.0, lam_c=0.0, rho_c=0.0):
    """Compute loss. Targets = original S_obs (x_seq[:,:,0])."""
    targets = x_seq[:, :, 0]  # Original S_obs (before masking)

    in_mask = (stock == 0) & (valid > 0)
    so_mask = (stock == 1) & (valid > 0)

    # L_data: MSE on in-stock hours only
    if in_mask.sum() > 0:
        L_data = ((D_star[in_mask] - targets[in_mask]) ** 2).mean()
    else:
        L_data = torch.tensor(0.0, device=D_star.device)

    if mode == 'vanilla':
        return L_data, {'L_data': L_data.item(), 'V_b': 0.0, 'V_c': 0.0}

    # ---- PINN constraints ----
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
    min_DI = torch.min(D_star[:, :-1], I_star[:, :-1])
    R_implied = I_star[:, 1:] - I_star[:, :-1] + min_DI
    neg_R = torch.relu(-R_implied)

    valid_pairs = (valid[:, :-1] > 0) & (valid[:, 1:] > 0)

    if valid_pairs.sum() > 0:
        V_c = neg_R[valid_pairs].mean()
        Q_c = neg_R[valid_pairs].pow(2).mean()
    else:
        V_c = torch.tensor(0.0, device=D_star.device)
        Q_c = torch.tensor(0.0, device=D_star.device)

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
    """Predict D*, I* for all windows (no masking randomness at eval)."""
    model.eval()
    x_seq_t = torch.from_numpy(data['x_seq'])
    cat_ids_t = torch.from_numpy(data['cat_ids'])
    stock_t = torch.from_numpy(data['stock'])

    all_D, all_I = [], []
    with torch.no_grad():
        for s in range(0, len(x_seq_t), batch_size):
            e = min(s + batch_size, len(x_seq_t))
            D, I = model(x_seq_t[s:e].to(device),
                         cat_ids_t[s:e].to(device),
                         stock_t[s:e].to(device),
                         train_mask_rate=0.0)  # No random masking at eval
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

    records = []
    for s_idx in range(n_series):
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
    targets = windows_data['x_seq'][:, :, 0]
    meta = windows_data['meta']

    in_mask = (stock == 0) & (valid > 0)
    preds_in = D_star[in_mask]
    obs_in = targets[in_mask]
    sae = np.abs(preds_in - obs_in).sum()
    sao = np.abs(obs_in).sum()
    pooled = {
        'wape_instock': sae / sao if sao > 0 else np.nan,
        'wpe_instock': (preds_in - obs_in).sum() / obs_in.sum() if obs_in.sum() != 0 else np.nan,
    }

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
# 10. Training functions (MODIFIED: pass stock to model, apply train masking)
# =========================================================================
def train_masked(model, train_data, val_data, device, max_epochs, patience, lr,
                 train_mask_rate):
    """Train Model A (masked, L_data only)."""
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

            # Pass stock to model for masking; random in-stock masking during training
            D_star, I_star = model(x_seq, cat_ids, stock,
                                    train_mask_rate=train_mask_rate)
            # IMPORTANT: compute_loss uses the ORIGINAL x_seq (before masking)
            # The model internally masks S_obs, but targets remain original
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


def train_pinn_masked(model, train_data, val_data, device, warmup_epochs,
                      alm_max_iter, alm_inner_epochs, alm_patience, lr,
                      rho_init, rho_gamma, train_mask_rate):
    """Train Model C (PINN + masked, ALM)."""
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
            stock = stock.to(device)
            valid = valid.to(device)

            D_star, I_star = model(x_seq, cat_ids, stock,
                                    train_mask_rate=train_mask_rate)
            loss, _ = compute_loss(D_star, I_star, x_seq, stock, valid, mode='vanilla')
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

                D_star, I_star = model(x_seq, cat_ids, stock,
                                        train_mask_rate=train_mask_rate)
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


def retrain_masked_fixed(model, train_data, device, lr, n_epochs, train_mask_rate):
    """Retrain masked model for fixed epochs (no validation)."""
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
            stock = stock.to(device)
            valid = valid.to(device)

            D_star, I_star = model(x_seq, cat_ids, stock,
                                    train_mask_rate=train_mask_rate)
            loss, _ = compute_loss(D_star, I_star, x_seq, stock, valid, mode='vanilla')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if epoch % 5 == 0 or epoch == 1:
            print(f"      Epoch {epoch:3d}/{n_epochs}: loss={total_loss/n_batches:.6f}")


def retrain_pinn_masked_fixed(model, train_data, device, lr, warmup_epochs,
                               alm_iters, inner_epochs, rho_init, rho_gamma,
                               train_mask_rate):
    """Retrain PINN masked for fixed ALM iterations."""
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
            stock = stock.to(device)
            valid = valid.to(device)

            D_star, I_star = model(x_seq, cat_ids, stock,
                                    train_mask_rate=train_mask_rate)
            loss, _ = compute_loss(D_star, I_star, x_seq, stock, valid, mode='vanilla')
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

                D_star, I_star = model(x_seq, cat_ids, stock,
                                        train_mask_rate=train_mask_rate)
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
# 11. Run Model A (Masked, no physics)
# =========================================================================
N_CONT_FEATS = 11  # S_obs + hour + dow + dom + 7 cont

print("\n\n" + "=" * 72)
print("  MODEL A — MASKED TRANSFORMER (L_data only, input masking)")
print("=" * 72)
t_a = time.time()

torch.manual_seed(SEED)
model_A = MaskedPINNSequential(N_CONT_FEATS, EMB_DIMS, CARDINALITIES,
                                D_MODEL, N_LAYERS, N_HEADS, D_FF, DROPOUT)
n_params = sum(p.numel() for p in model_A.parameters())
print(f"  Params: {n_params:,}")

# Train
print("\n  Training Model A...")
best_wape_A, best_epoch_A = train_masked(
    model_A, train_windows, val_windows, DEVICE, MAX_EPOCHS_A, PATIENCE_A, LR,
    TRAIN_MASK_RATE)
print(f"  Best: epoch={best_epoch_A}, val WAPE_in={best_wape_A:.6f}")
print(f"  Training time: {time.time()-t_a:.0f}s")
print(f"  Learned mask_value: {model_A.mask_value.item():.6f}")

torch.save(model_A.state_dict(), os.path.join(RESULTS_DIR, 'masked_A.pt'))

# Val recovery evaluation
print("\n  Val recovery (MNAR val, seed=42):")
rec_val_A = eval_recovery_mnar(model_A, val_mnar_windows, DEVICE)
print(f"    WAPE_recovery: {rec_val_A['wape_pool']:.4f}")
print(f"    WPE_recovery:  {rec_val_A['wpe_pool']:.4f}")
print(f"    pred_mean: {rec_val_A['pred_mean']:.6f}, gt_mean: {rec_val_A['gt_mean']:.6f}")

# Retrain on days 1-90
print(f"\n  Retraining Model A on days 1-90 ({best_epoch_A} epochs)...")
t_rt = time.time()
retrain_windows_A = build_windows(1, 90)
print(f"  Retrain windows: {len(retrain_windows_A['x_seq']):,}")

torch.manual_seed(SEED)
model_A_rt = MaskedPINNSequential(N_CONT_FEATS, EMB_DIMS, CARDINALITIES,
                                    D_MODEL, N_LAYERS, N_HEADS, D_FF, DROPOUT)
retrain_masked_fixed(model_A_rt, retrain_windows_A, DEVICE, LR, best_epoch_A,
                      TRAIN_MASK_RATE)
print(f"  Retrain time: {time.time()-t_rt:.0f}s")

torch.save(model_A_rt.state_dict(), os.path.join(RESULTS_DIR, 'masked_A_retrained.pt'))

# Test recovery (MNAR test, seed=123)
print("\n  Test recovery (MNAR test, seed=123):")
test_mnar_windows = build_windows(1, 90,
                                   apply_mnar_mask=mnar_test_mask,
                                   apply_mnar_gt=mnar_test_gt)
rec_test_A = eval_recovery_mnar(model_A_rt, test_mnar_windows, DEVICE)
print(f"    WAPE_recovery: {rec_test_A['wape_pool']:.4f}")
print(f"    WPE_recovery:  {rec_test_A['wpe_pool']:.4f}")
print(f"    N positions:   {rec_test_A['n_positions']:,}")
print(f"    pred_mean: {rec_test_A['pred_mean']:.6f}, gt_mean: {rec_test_A['gt_mean']:.6f}")

ps_rec_A = eval_recovery_per_series(model_A_rt, test_mnar_windows, DEVICE)
ps_rec_A.to_parquet(os.path.join(RESULTS_DIR, 'masked_A_recovery_per_series.parquet'),
                     index=False)
print(f"    WAPE_recovery median: {ps_rec_A['wape_recovery'].median():.4f}")
print(f"    WPE_recovery median:  {ps_rec_A['wpe_recovery'].median():.4f}")

# Test forecasting (eval HF)
print("\n  Test forecasting (eval HF, gg 91-97):")
test_fc_windows = build_windows(91, 97)
pooled_fc_A, ps_fc_A = eval_forecasting(model_A_rt, test_fc_windows, DEVICE)
ps_fc_A.to_parquet(os.path.join(RESULTS_DIR, 'masked_A_test_per_series.parquet'),
                    index=False)
print(f"    WAPE_in pooled: {pooled_fc_A['wape_instock']:.4f}")
print(f"    WAPE_in median: {ps_fc_A['wape_instock'].median():.4f}")
print(f"    WPE_in pooled:  {pooled_fc_A['wpe_instock']:.4f}")
print(f"    WPE_in median:  {ps_fc_A['wpe_instock'].median():.4f}")

print(f"\n  Total Model A time: {time.time()-t_a:.0f}s")

del model_A, model_A_rt, retrain_windows_A
if DEVICE == 'mps':
    torch.mps.empty_cache()


# =========================================================================
# 12. Run Model C (PINN + Masked)
# =========================================================================
print("\n\n" + "=" * 72)
print("  MODEL C — PINN + MASKED (L_data + L_boundary + L_cons + ALM + masking)")
print("=" * 72)
t_c = time.time()

torch.manual_seed(SEED)
model_C = MaskedPINNSequential(N_CONT_FEATS, EMB_DIMS, CARDINALITIES,
                                D_MODEL, N_LAYERS, N_HEADS, D_FF, DROPOUT)

# Train
print("\n  Training Model C...")
best_wape_C, total_epochs_C, best_alm_C = train_pinn_masked(
    model_C, train_windows, val_windows, DEVICE,
    WARMUP_EPOCHS, ALM_MAX_ITER, ALM_INNER_EPOCHS, ALM_PATIENCE,
    LR, RHO_INIT, RHO_GAMMA, TRAIN_MASK_RATE)
print(f"  Best: ALM iter={best_alm_C}, total_epochs={total_epochs_C}, "
      f"val WAPE_in={best_wape_C:.6f}")
print(f"  Training time: {time.time()-t_c:.0f}s")
print(f"  Learned mask_value: {model_C.mask_value.item():.6f}")

torch.save(model_C.state_dict(), os.path.join(RESULTS_DIR, 'masked_C.pt'))

# Val recovery
print("\n  Val recovery (MNAR val, seed=42):")
rec_val_C = eval_recovery_mnar(model_C, val_mnar_windows, DEVICE)
print(f"    WAPE_recovery: {rec_val_C['wape_pool']:.4f}")
print(f"    WPE_recovery:  {rec_val_C['wpe_pool']:.4f}")
print(f"    pred_mean: {rec_val_C['pred_mean']:.6f}, gt_mean: {rec_val_C['gt_mean']:.6f}")

# Retrain
print(f"\n  Retraining Model C on days 1-90...")
t_rt2 = time.time()
retrain_windows_C = build_windows(1, 90)

torch.manual_seed(SEED)
model_C_rt = MaskedPINNSequential(N_CONT_FEATS, EMB_DIMS, CARDINALITIES,
                                    D_MODEL, N_LAYERS, N_HEADS, D_FF, DROPOUT)
retrain_pinn_masked_fixed(model_C_rt, retrain_windows_C, DEVICE, LR,
                           WARMUP_EPOCHS, best_alm_C, ALM_INNER_EPOCHS,
                           RHO_INIT, RHO_GAMMA, TRAIN_MASK_RATE)
print(f"  Retrain time: {time.time()-t_rt2:.0f}s")

torch.save(model_C_rt.state_dict(), os.path.join(RESULTS_DIR, 'masked_C_retrained.pt'))

# Test recovery
print("\n  Test recovery (MNAR test, seed=123):")
rec_test_C = eval_recovery_mnar(model_C_rt, test_mnar_windows, DEVICE)
print(f"    WAPE_recovery: {rec_test_C['wape_pool']:.4f}")
print(f"    WPE_recovery:  {rec_test_C['wpe_pool']:.4f}")
print(f"    N positions:   {rec_test_C['n_positions']:,}")
print(f"    pred_mean: {rec_test_C['pred_mean']:.6f}, gt_mean: {rec_test_C['gt_mean']:.6f}")

ps_rec_C = eval_recovery_per_series(model_C_rt, test_mnar_windows, DEVICE)
ps_rec_C.to_parquet(os.path.join(RESULTS_DIR, 'masked_C_recovery_per_series.parquet'),
                     index=False)
print(f"    WAPE_recovery median: {ps_rec_C['wape_recovery'].median():.4f}")
print(f"    WPE_recovery median:  {ps_rec_C['wpe_recovery'].median():.4f}")

# Test forecasting
print("\n  Test forecasting (eval HF, gg 91-97):")
pooled_fc_C, ps_fc_C = eval_forecasting(model_C_rt, test_fc_windows, DEVICE)
ps_fc_C.to_parquet(os.path.join(RESULTS_DIR, 'masked_C_test_per_series.parquet'),
                    index=False)
print(f"    WAPE_in pooled: {pooled_fc_C['wape_instock']:.4f}")
print(f"    WAPE_in median: {ps_fc_C['wape_instock'].median():.4f}")
print(f"    WPE_in pooled:  {pooled_fc_C['wpe_instock']:.4f}")
print(f"    WPE_in median:  {ps_fc_C['wpe_instock'].median():.4f}")

print(f"\n  Total Model C time: {time.time()-t_c:.0f}s")


# =========================================================================
# 13. Confronto finale
# =========================================================================
print("\n\n" + "=" * 72)
print("  CONFRONTO FINALE (tutti i modelli)")
print("=" * 72)

# Recovery
print(f"\n  --- TRACCIA A: Recovery (MNAR test, seed=123) ---")
print(f"  {'Modello':<25} {'WAPE_pool':>12} {'WAPE_med':>12} "
      f"{'WPE_pool':>12} {'WPE_med':>12}")
print("  " + "-" * 68)

# Load imputation results
for label, prefix in [('LGB Imputer', 'imputation_lgb'),
                       ('Cond Mean', 'imputation_cond_mean'),
                       ('Global Mean', 'imputation_global_mean')]:
    path = os.path.join(RESULTS_DIR, f'{prefix}_test_per_series.parquet')
    if os.path.exists(path):
        ps = pd.read_parquet(path)
        print(f"  {label:<25} {'—':>12} {ps['wape_recovery'].median():>12.4f} "
              f"{'—':>12} {ps['wpe_recovery'].median():>12.4f}")

# Models D, B from Fase 1
for label, prefix in [('Model D (vanilla)', 'pinn_seq_D'),
                       ('Model B (PINN)', 'pinn_seq_B')]:
    path = os.path.join(RESULTS_DIR, f'{prefix}_recovery_per_series.parquet')
    if os.path.exists(path):
        ps = pd.read_parquet(path)
        print(f"  {label:<25} {'—':>12} {ps['wape_recovery'].median():>12.4f} "
              f"{'—':>12} {ps['wpe_recovery'].median():>12.4f}")

# Models A, C
print(f"  {'Model A (masked)':<25} {rec_test_A['wape_pool']:>12.4f} "
      f"{ps_rec_A['wape_recovery'].median():>12.4f} "
      f"{rec_test_A['wpe_pool']:>12.4f} "
      f"{ps_rec_A['wpe_recovery'].median():>12.4f}")
print(f"  {'Model C (PINN+masked)':<25} {rec_test_C['wape_pool']:>12.4f} "
      f"{ps_rec_C['wape_recovery'].median():>12.4f} "
      f"{rec_test_C['wpe_pool']:>12.4f} "
      f"{ps_rec_C['wpe_recovery'].median():>12.4f}")

# Forecasting
print(f"\n  --- TRACCIA B: Forecasting (eval HF, gg 91-97) ---")
print(f"  {'Modello':<25} {'WAPE_in_pool':>14} {'WAPE_in_med':>14} "
      f"{'WPE_in_pool':>14} {'WPE_in_med':>14}")
print("  " + "-" * 80)

# Fase A results
for label, prefix in [('MLP F (dirty)', 'mlp_f'), ('LGB F (dirty)', 'lgb_f'),
                       ('DoW Mean (dirty)', 'dow_mean')]:
    path = os.path.join(RESULTS_DIR, f'{prefix}_test_per_series.parquet')
    if os.path.exists(path):
        ps = pd.read_parquet(path)
        print(f"  {label:<25} {'—':>14} {ps['wape_instock'].median():>14.4f} "
              f"{'—':>14} {ps['wpe_instock'].median():>14.4f}")

# Clean
for label, prefix in [('MLP F (clean)', 'clean_mlp_f'), ('LGB F (clean)', 'clean_lgb_f')]:
    path = os.path.join(RESULTS_DIR, f'{prefix}_test_per_series.parquet')
    if os.path.exists(path):
        ps = pd.read_parquet(path)
        print(f"  {label:<25} {'—':>14} {ps['wape_instock'].median():>14.4f} "
              f"{'—':>14} {ps['wpe_instock'].median():>14.4f}")

# Models D, B from Fase 1
for label, prefix in [('Model D (vanilla)', 'pinn_seq_D'),
                       ('Model B (PINN)', 'pinn_seq_B')]:
    path = os.path.join(RESULTS_DIR, f'{prefix}_test_per_series.parquet')
    if os.path.exists(path):
        ps = pd.read_parquet(path)
        print(f"  {label:<25} {'—':>14} {ps['wape_instock'].median():>14.4f} "
              f"{'—':>14} {ps['wpe_instock'].median():>14.4f}")

# Models A, C
print(f"  {'Model A (masked)':<25} {pooled_fc_A['wape_instock']:>14.4f} "
      f"{ps_fc_A['wape_instock'].median():>14.4f} "
      f"{pooled_fc_A['wpe_instock']:>14.4f} "
      f"{ps_fc_A['wpe_instock'].median():>14.4f}")
print(f"  {'Model C (PINN+masked)':<25} {pooled_fc_C['wape_instock']:>14.4f} "
      f"{ps_fc_C['wape_instock'].median():>14.4f} "
      f"{pooled_fc_C['wpe_instock']:>14.4f} "
      f"{ps_fc_C['wpe_instock'].median():>14.4f}")

print(f"\n  Tempo totale: {time.time()-t0:.0f}s")
print("=" * 72)
