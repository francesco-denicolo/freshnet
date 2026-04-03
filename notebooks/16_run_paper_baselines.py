"""
16_run_paper_baselines.py — Modelli paper baseline per demand recovery (Traccia A)
===================================================================================
Esegue SAITS, iTransformer, DLinear dalla libreria PyPOTS
sui nostri dati con le nostre maschere MNAR.

Procedura (nostra, non del paper):
1. Carica dati giorni 1-83 con arrays orari 0-23
2. Applica nostre maschere MNAR (seed=42, 30%) + stockout naturale -> NaN
3. Crea finestre di 7 giorni (168 timestep) per serie
4. Split train (giorni 1-76) / val (giorni 77-83) per early stopping
5. Allena ogni modello di imputation (self-supervised) con early stopping su val
6. Valuta su posizioni MNAR (ground truth noto)

Metriche:
- WAPE_recovery = sum|D_hat - gt| / sum(gt)  sulle posizioni MNAR
- WPE_recovery  = sum(D_hat - gt) / sum(gt)  sulle posizioni MNAR

Eseguire con: freshnet/bin/python notebooks/16_run_paper_baselines.py
"""

import os
import sys
import time
import gc
import numpy as np
import pandas as pd
import torch

# PyPOTS imputation models
from pypots.imputation import SAITS, iTransformer, DLinear
from pypots.optim import Adam
from pypots.nn.modules.loss import MAE as PyPOTS_MAE

# =========================================================================
# CONFIG
# =========================================================================
SEED = 42
MAX_DAY = 83
TRAIN_DAYS = 76       # days 1-76 for training (0-indexed: 0-75)
WINDOW_DAYS = 7
WINDOW_HOURS = WINDOW_DAYS * 24  # 168

# PyPOTS training hyperparameters
EPOCHS = 50           # max epochs — early stopping decides the optimum
BATCH_SIZE = 128
PATIENCE = 5
D_MODEL = 64
N_LAYERS = 2
N_HEADS = 4
D_FFN = 128
LR = 0.001
DROPOUT = 0.0

# Subsample series for speed (None = all 50,000)
N_SERIES_SUBSAMPLE = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
SAVE_DIR = os.path.join(BASE_DIR, 'baseline_paper', 'pypots_saves')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# Device
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

np.random.seed(SEED)

print("=" * 70)
print("FASE 0 — Step 0.3: Modelli paper baseline (demand recovery)")
print("=" * 70)
print(f"  Device:     {DEVICE}")
print(f"  Window:     {WINDOW_DAYS} giorni ({WINDOW_HOURS} ore)")
print(f"  Epochs:     {EPOCHS} (max, early stopping patience={PATIENCE})")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  d_model:    {D_MODEL}")
print(f"  Train/Val:  giorni 1-{TRAIN_DAYS} / {TRAIN_DAYS+1}-{MAX_DAY}")
print(f"  Subsample:  {N_SERIES_SUBSAMPLE or 'all'}")

# =========================================================================
# 1. LOAD DATA
# =========================================================================
t0 = time.time()
print("\n1. Caricamento dati...")

df = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df['dt_parsed'] = pd.to_datetime(df['dt'])
min_date = df['dt_parsed'].min()
df['day_num'] = (df['dt_parsed'] - min_date).dt.days + 1
df['dow'] = df['dt_parsed'].dt.dayofweek

# Filter days 1-83
df = df[df['day_num'] <= MAX_DAY].copy()

# Sort by series and date
df = df.sort_values(['store_id', 'product_id', 'dt_parsed']).reset_index(drop=True)
N = len(df)
print(f"  Righe: {N:,}")

# Parse hourly arrays
sales_all = np.array(df['hours_sale'].tolist(), dtype=np.float32)       # (N, 24)
stock_all = np.array(df['hours_stock_status'].tolist(), dtype=np.float32)  # (N, 24)

# Get series info
series_keys_sorted = df.groupby(['store_id', 'product_id'], sort=False).first().reset_index()[['store_id', 'product_id']]
n_series_total = len(series_keys_sorted)
n_days = MAX_DAY  # 83

print(f"  Serie: {n_series_total:,}, Giorni: {n_days}")
assert N == n_series_total * n_days, f"Expected {n_series_total * n_days}, got {N}"

# Reshape to 3D: (n_series, n_days, 24)
sales_3d = sales_all.reshape(n_series_total, n_days, 24)
stock_3d = stock_all.reshape(n_series_total, n_days, 24)

# Covariates (daily)
CONT_COLS = ['discount', 'holiday_flag', 'avg_temperature', 'precpt']
cov_daily = df[CONT_COLS].values.astype(np.float32).reshape(n_series_total, n_days, len(CONT_COLS))
# Day of week
dows_2d = df['dow'].values.reshape(n_series_total, n_days)

# Normalize continuous covariates (z-score from TRAIN days only)
for ci, col in enumerate(CONT_COLS):
    train_vals = cov_daily[:, :TRAIN_DAYS, ci]
    mu, sig = train_vals.mean(), train_vals.std()
    if sig > 0:
        cov_daily[:, :, ci] = (cov_daily[:, :, ci] - mu) / sig
    print(f"  Normalizzato {col}: mean={mu:.4f}, std={sig:.4f} (da train)")

print(f"  Tempo loading: {time.time()-t0:.1f}s")

# =========================================================================
# 2. LOAD MNAR MASKS
# =========================================================================
print("\n2. Caricamento maschere MNAR...")
mask_df = pd.read_parquet(os.path.join(DATA_DIR, 'mnar_masks.parquet'))
print(f"  Maschere totali: {len(mask_df):,}")

# Create series index mapping
series_keys_sorted['series_idx'] = np.arange(n_series_total)
mask_df = mask_df.merge(
    series_keys_sorted[['store_id', 'product_id', 'series_idx']],
    on=['store_id', 'product_id'],
    how='inner'
)

# Date to day_idx mapping
mask_df['dt_parsed'] = pd.to_datetime(mask_df['dt'])
mask_df['day_idx'] = (mask_df['dt_parsed'] - min_date).dt.days  # 0-indexed
mask_df = mask_df[(mask_df['day_idx'] >= 0) & (mask_df['day_idx'] < MAX_DAY)].copy()

# Create 3D mask: (n_series, n_days, 24)
mnar_3d = np.zeros((n_series_total, n_days, 24), dtype=bool)
gt_3d = np.zeros((n_series_total, n_days, 24), dtype=np.float32)

si = mask_df['series_idx'].values
di = mask_df['day_idx'].values
hi = mask_df['hour'].values.astype(np.int32)
mnar_3d[si, di, hi] = True
gt_3d[si, di, hi] = mask_df['ground_truth'].values.astype(np.float32)

print(f"  Maschere mappate: {len(mask_df):,}")
print(f"  MNAR slots totali: {mnar_3d.sum():,}")
print(f"  MNAR train (gg 1-{TRAIN_DAYS}): {mnar_3d[:, :TRAIN_DAYS, :].sum():,}")
print(f"  MNAR val   (gg {TRAIN_DAYS+1}-{MAX_DAY}): {mnar_3d[:, TRAIN_DAYS:, :].sum():,}")

mask_df = mask_df.drop(columns=['series_idx', 'dt_parsed', 'day_idx'])

# =========================================================================
# 3. SUBSAMPLE SERIES (optional)
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
    series_keys_sorted = series_keys_sorted.iloc[sel_idx].reset_index(drop=True)
    n_series = N_SERIES_SUBSAMPLE
else:
    n_series = n_series_total
    print(f"\n3. Usando tutte le {n_series} serie")

# =========================================================================
# 4. CREATE WINDOWED DATASET (train/val split)
# =========================================================================
print("\n4. Creazione dataset a finestre (con split train/val)...")

N_FEATURES = 1 + 1 + 1 + len(CONT_COLS)  # sales + hour + dow + 4 cov = 7

# Hour normalization pattern
hour_pattern = np.tile(np.arange(24, dtype=np.float32) / 23.0, WINDOW_DAYS)  # (168,)


def build_windows(day_start, day_end, label=""):
    """Build windowed dataset from day_start to day_end (0-indexed, exclusive).
    Returns X (masked), X_ori (original, no NaN masking on sales), and tracking arrays.
    """
    n_days_range = day_end - day_start
    n_complete = n_days_range // WINDOW_DAYS
    remaining = n_days_range - n_complete * WINDOW_DAYS
    n_win = n_complete + (1 if remaining > 0 else 0)
    total = n_series * n_win

    X = np.full((total, WINDOW_HOURS, N_FEATURES), np.nan, dtype=np.float32)
    X_ori = np.full((total, WINDOW_HOURS, N_FEATURES), np.nan, dtype=np.float32)
    w_valid = np.zeros((total, WINDOW_HOURS), dtype=bool)
    w_mnar = np.zeros((total, WINDOW_HOURS), dtype=bool)
    w_gt = np.zeros((total, WINDOW_HOURS), dtype=np.float32)
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

        # Sales (feature 0) — NaN where stockout or MNAR-masked
        sales_w = sales_3d[:, sd:ed, :].reshape(n_series, actual_hours)
        stock_w = stock_3d[:, sd:ed, :].reshape(n_series, actual_hours)
        mnar_w = mnar_3d[:, sd:ed, :].reshape(n_series, actual_hours)
        gt_w = gt_3d[:, sd:ed, :].reshape(n_series, actual_hours)

        missing_w = (stock_w == 1) | mnar_w
        sales_masked = np.where(missing_w, np.nan, sales_w)
        X[ws:we, :actual_hours, 0] = sales_masked

        # X_ori: original sales (no masking) for PyPOTS validation
        X_ori[ws:we, :actual_hours, 0] = sales_w

        # Hour norm (feature 1)
        X[ws:we, :actual_hours, 1] = hour_pattern[:actual_hours]
        X_ori[ws:we, :actual_hours, 1] = hour_pattern[:actual_hours]

        # DoW norm (feature 2)
        dow_w = dows_2d[:, sd:ed]
        dow_hourly = np.repeat(dow_w, 24, axis=1).astype(np.float32) / 6.0
        X[ws:we, :actual_hours, 2] = dow_hourly
        X_ori[ws:we, :actual_hours, 2] = dow_hourly

        # Covariates (features 3-6)
        cov_w = cov_daily[:, sd:ed, :]
        cov_hourly = np.repeat(cov_w, 24, axis=1)
        X[ws:we, :actual_hours, 3:7] = cov_hourly
        X_ori[ws:we, :actual_hours, 3:7] = cov_hourly

        # Tracking
        w_valid[ws:we, :actual_hours] = True
        w_mnar[ws:we, :actual_hours] = mnar_w
        w_gt[ws:we, :actual_hours] = gt_w
        w_meta[ws:we, 0] = np.arange(n_series)
        w_meta[ws:we, 1] = w

    if label:
        print(f"  {label}: {n_win} finestre/serie, {total:,} totali, "
              f"NaN sales={np.isnan(X[:,:,0][w_valid]).sum()/w_valid.sum()*100:.1f}%, "
              f"MNAR={w_mnar.sum():,}")

    return X, X_ori, w_valid, w_mnar, w_gt, w_meta, n_win


t_win = time.time()

# Train: days 0 to TRAIN_DAYS-1 (day_num 1-76)
X_train, X_ori_train, valid_train, mnar_train, gt_train, meta_train, nw_train = \
    build_windows(0, TRAIN_DAYS, label=f"Train (gg 1-{TRAIN_DAYS})")

# Val: days TRAIN_DAYS to MAX_DAY-1 (day_num 77-83)
X_val, X_ori_val, valid_val, mnar_val, gt_val, meta_val, nw_val = \
    build_windows(TRAIN_DAYS, n_days, label=f"Val (gg {TRAIN_DAYS+1}-{MAX_DAY})")

# Full dataset for prediction and evaluation
X_all = np.concatenate([X_train, X_val], axis=0)
valid_all = np.concatenate([valid_train, valid_val], axis=0)
mnar_all = np.concatenate([mnar_train, mnar_val], axis=0)
gt_all = np.concatenate([gt_train, gt_val], axis=0)
meta_all = np.concatenate([meta_train, meta_val], axis=0)
total_windows = len(X_all)

print(f"  Costruzione completata in {time.time()-t_win:.1f}s")
print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}, X_all: {X_all.shape}")
print(f"  MNAR totali: {mnar_all.sum():,}")
print(f"  Memory X_all: {X_all.nbytes / 1024**3:.2f} GB")

# =========================================================================
# 5. RUN MODELS
# =========================================================================
print(f"\n{'='*70}")
print("5. Training e valutazione modelli")
print(f"{'='*70}")

n_steps = WINDOW_HOURS
n_features = N_FEATURES


def make_model(name):
    saving_path = os.path.join(SAVE_DIR, name)
    os.makedirs(saving_path, exist_ok=True)
    common = dict(
        n_steps=n_steps,
        n_features=n_features,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        patience=PATIENCE,
        optimizer=Adam(lr=LR),
        training_loss=PyPOTS_MAE(),
        validation_metric=PyPOTS_MAE(),
        device=DEVICE,
        saving_path=saving_path,
    )
    if name == 'SAITS':
        return SAITS(
            n_layers=N_LAYERS,
            d_model=D_MODEL,
            d_ffn=D_FFN,
            n_heads=N_HEADS,
            d_k=D_MODEL // N_HEADS,
            d_v=D_MODEL // N_HEADS,
            dropout=DROPOUT,
            attn_dropout=DROPOUT,
            **common,
        )
    elif name == 'iTransformer':
        return iTransformer(
            n_layers=N_LAYERS,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            d_k=D_MODEL // N_HEADS,
            d_v=D_MODEL // N_HEADS,
            d_ffn=D_FFN,
            dropout=DROPOUT,
            attn_dropout=DROPOUT,
            **common,
        )
    elif name == 'DLinear':
        return DLinear(
            moving_avg_window_size=25,
            d_model=D_MODEL,
            **common,
        )
    else:
        raise ValueError(f"Unknown model: {name}")


MODEL_NAMES = ['SAITS', 'DLinear', 'iTransformer']
PREDICT_CHUNK = 5_000

all_results = {}

for model_name in MODEL_NAMES:
    sys.stdout.write(f"\n{'='*60}\n  Modello: {model_name}\n{'='*60}\n")
    sys.stdout.flush()

    # Skip if already done
    ps_path_check = os.path.join(RESULTS_DIR, f'{model_name.lower()}_recovery_per_series.parquet')
    if os.path.exists(ps_path_check):
        sys.stdout.write(f"  GIA' COMPLETATO, skip. ({ps_path_check})\n")
        sys.stdout.flush()
        ps_existing = pd.read_parquet(ps_path_check)
        all_results[model_name] = {
            'wape_overall': np.nan,
            'wpe_overall': np.nan,
            'wape_median': ps_existing['wape_recovery'].median(),
            'wpe_median': ps_existing['wpe_recovery'].median(),
            'best_epoch': 0,
            'train_time_s': 0,
        }
        continue

    t_model = time.time()

    # 5a. Initialize model
    sys.stdout.write(f"  Inizializzazione...\n"); sys.stdout.flush()
    model = make_model(model_name)

    # 5b. Train with val set for early stopping
    sys.stdout.write(f"  Training (max {EPOCHS} epoche, patience={PATIENCE}, "
                     f"val={len(X_val):,} finestre)...\n"); sys.stdout.flush()
    t_train = time.time()
    model.fit(train_set={"X": X_train}, val_set={"X": X_val, "X_ori": X_ori_val})
    train_time = time.time() - t_train

    # Extract best epoch from model info
    best_epoch = getattr(model, 'best_epoch', EPOCHS)
    sys.stdout.write(f"  Training completato in {train_time:.1f}s ({train_time/60:.1f} min), "
                     f"best epoch: {best_epoch}\n"); sys.stdout.flush()

    # 5c. Predict on FULL dataset (train+val) in small chunks on CPU
    sys.stdout.write(f"  Spostando modello su CPU per predizione...\n"); sys.stdout.flush()
    if hasattr(model, 'model') and hasattr(model.model, 'to'):
        model.model.to(torch.device('cpu'))
    model.device = torch.device('cpu')
    if DEVICE.type == 'mps':
        torch.mps.empty_cache()
    gc.collect()

    sys.stdout.write(f"  Predizione su {total_windows:,} finestre in chunks da "
                     f"{PREDICT_CHUNK:,} (CPU)...\n"); sys.stdout.flush()
    t_pred = time.time()
    imputed_sales = np.zeros((total_windows, WINDOW_HOURS), dtype=np.float32)

    for ci in range(0, total_windows, PREDICT_CHUNK):
        ce = min(ci + PREDICT_CHUNK, total_windows)
        chunk_result = model.predict({"X": X_all[ci:ce]})
        imp_chunk = chunk_result["imputation"][:, :, 0]  # feature 0 = sales
        imputed_sales[ci:ce] = np.clip(imp_chunk, 0, None)
        del chunk_result, imp_chunk
        gc.collect()
        if ci % 50_000 == 0:
            sys.stdout.write(f"    Chunk {ci:,}-{ce:,} / {total_windows:,}\n"); sys.stdout.flush()

    pred_time = time.time() - t_pred
    sys.stdout.write(f"  Predizione completata in {pred_time:.1f}s ({pred_time/60:.1f} min)\n")
    sys.stdout.flush()

    # 5d. Evaluate on MNAR positions
    sys.stdout.write(f"  Valutazione su posizioni MNAR...\n"); sys.stdout.flush()

    mnar_flat = mnar_all.ravel()
    pred_flat = imputed_sales.ravel()
    gt_flat = gt_all.ravel()

    mnar_idx = np.where(mnar_flat)[0]
    pred_mnar = pred_flat[mnar_idx]
    gt_mnar = gt_flat[mnar_idx]

    n_mnar = len(mnar_idx)

    # Overall metrics
    gt_sum = gt_mnar.sum()
    if gt_sum > 0:
        wape = np.abs(pred_mnar - gt_mnar).sum() / gt_sum
        wpe = (pred_mnar - gt_mnar).sum() / gt_sum
    else:
        wape = np.nan
        wpe = np.nan

    # Metrics on GT>0 only
    gt_pos_mask = gt_mnar > 0
    n_pos = gt_pos_mask.sum()
    gt_pos_sum = gt_mnar[gt_pos_mask].sum()
    if gt_pos_sum > 0:
        wape_pos = np.abs(pred_mnar[gt_pos_mask] - gt_mnar[gt_pos_mask]).sum() / gt_pos_sum
        wpe_pos = (pred_mnar[gt_pos_mask] - gt_mnar[gt_pos_mask]).sum() / gt_pos_sum
    else:
        wape_pos = np.nan
        wpe_pos = np.nan

    mae = np.abs(pred_mnar - gt_mnar).mean()

    sys.stdout.write(f"\n  --- Metriche Recovery {model_name} (best epoch {best_epoch}) ---\n")
    sys.stdout.write(f"  WAPE_recovery (overall): {wape:.4f}  (GT>0: {wape_pos:.4f})\n")
    sys.stdout.write(f"  WPE_recovery  (overall): {wpe:.4f}  (GT>0: {wpe_pos:.4f})\n")
    sys.stdout.write(f"  MAE:                     {mae:.6f}\n")
    sys.stdout.write(f"  N MNAR: {n_mnar:,}, N GT>0: {n_pos:,}\n")
    sys.stdout.flush()

    # 5e. Per-series recovery metrics
    sys.stdout.write(f"  Calcolo metriche per-serie...\n"); sys.stdout.flush()

    mnar_window_idx = mnar_idx // WINDOW_HOURS
    mnar_series_idx = meta_all[mnar_window_idx, 0]

    eval_df = pd.DataFrame({
        'series_idx': mnar_series_idx,
        'pred': pred_mnar,
        'gt': gt_mnar,
        'abs_err': np.abs(pred_mnar - gt_mnar),
        'err': pred_mnar - gt_mnar,
    })

    agg = eval_df.groupby('series_idx').agg(
        gt_sum=('gt', 'sum'),
        abs_err_sum=('abs_err', 'sum'),
        err_sum=('err', 'sum'),
        n_mnar=('gt', 'count'),
    )
    agg['wape_recovery'] = np.where(agg['gt_sum'] > 0, agg['abs_err_sum'] / agg['gt_sum'], np.nan)
    agg['wpe_recovery'] = np.where(agg['gt_sum'] > 0, agg['err_sum'] / agg['gt_sum'], np.nan)

    agg = agg.reset_index()
    agg['store_id'] = series_keys_sorted.iloc[agg['series_idx'].values]['store_id'].values
    agg['product_id'] = series_keys_sorted.iloc[agg['series_idx'].values]['product_id'].values

    ps_df = agg[['store_id', 'product_id', 'wape_recovery', 'wpe_recovery', 'n_mnar', 'gt_sum']].copy()
    ps_path = os.path.join(RESULTS_DIR, f'{model_name.lower()}_recovery_per_series.parquet')
    ps_df.to_parquet(ps_path, index=False)

    wape_med = ps_df['wape_recovery'].median()
    wpe_med = ps_df['wpe_recovery'].median()

    sys.stdout.write(f"  WAPE_recovery mediana: {wape_med:.4f}\n")
    sys.stdout.write(f"  WPE_recovery mediana:  {wpe_med:.4f}\n")

    total_time = time.time() - t_model
    sys.stdout.write(f"  Salvato: {ps_path}\n")
    sys.stdout.write(f"  Tempo totale {model_name}: {total_time:.1f}s ({total_time/60:.1f} min)\n")
    sys.stdout.flush()

    all_results[model_name] = {
        'wape_overall': wape,
        'wpe_overall': wpe,
        'wape_gt_pos': wape_pos,
        'wpe_gt_pos': wpe_pos,
        'mae': mae,
        'wape_median': wape_med,
        'wpe_median': wpe_med,
        'best_epoch': best_epoch,
        'train_time_s': train_time,
        'n_mnar': n_mnar,
    }

    # Free memory
    del model, imputed_sales
    if DEVICE.type == 'mps':
        torch.mps.empty_cache()
    gc.collect()

# =========================================================================
# 6. SUMMARY TABLE
# =========================================================================
sys.stdout.write(f"\n{'='*70}\n")
sys.stdout.write("6. TABELLA RIEPILOGATIVA — Demand Recovery (Traccia A)\n")
sys.stdout.write(f"{'='*70}\n")

sys.stdout.write(f"\n{'Model':<16} {'WAPE_rec':>10} {'WPE_rec':>10} {'WAPE_med':>10} "
                 f"{'WPE_med':>10} {'MAE':>10} {'Epoch':>6} {'Time':>8}\n")
sys.stdout.write(f"{'-'*16} {'-'*10:>10} {'-'*10:>10} {'-'*10:>10} "
                 f"{'-'*10:>10} {'-'*10:>10} {'-'*6:>6} {'-'*8:>8}\n")

for name, r in sorted(all_results.items(), key=lambda x: x[1].get('wape_median', 999)):
    wape_o = r.get('wape_overall', np.nan)
    wpe_o = r.get('wpe_overall', np.nan)
    mae_v = r.get('mae', np.nan)
    t_v = r.get('train_time_s', 0)
    ep = r.get('best_epoch', 0)
    sys.stdout.write(f"{name:<16} {wape_o:10.4f} {wpe_o:10.4f} "
                     f"{r['wape_median']:10.4f} {r['wpe_median']:10.4f} "
                     f"{mae_v:10.6f} {ep:6d} {t_v/60:7.1f}m\n")

sys.stdout.write(f"\nTempo totale: {(time.time()-t0)/60:.1f} min\n")
sys.stdout.write("=" * 70 + "\n")
sys.stdout.flush()
