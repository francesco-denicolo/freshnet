"""
cap_sweep_imputer.py — capacity sweep for the deep-imputer robustness check
===========================================================================
Re-train one deep imputer at INCREASED capacity (toward the original-paper
sizes, within the CPU/16 GB budget) to test whether the imputer's downstream
immateriality is an artefact of the slim, untuned configurations used in the
main benchmark. Outputs are suffixed `_hicap` so they sit alongside the slim
cells without overwriting them.

Usage:
    python3 pipeline/cap_sweep_imputer.py itransformer
    python3 pipeline/cap_sweep_imputer.py timesnet

Then run the two lag-based forecasters on the produced completed series:
    HPO_VARIANT=1 python3 pipeline/08_fase_b2_forecast_mlp.py <name>_hicap
    HPO_VARIANT=1 python3 pipeline/07_fase_b2_forecast_lgb.py <name>_hicap

Mirrors the data construction and MNAR evaluation of 27_/28_fase_b1_*.py
exactly; only the model capacity and the output names differ.
"""
import sys, os, gc, time, functools, warnings
import numpy as np, pandas as pd

warnings.filterwarnings('ignore')
print = functools.partial(print, flush=True)

if len(sys.argv) != 2 or sys.argv[1] not in ('itransformer', 'timesnet', 'saits', 'dlinear'):
    print('Usage: python3 pipeline/cap_sweep_imputer.py <itransformer|timesnet|saits|dlinear>')
    sys.exit(1)
MODEL = sys.argv[1]

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
COMPLETED_DIR = os.path.join(DATA_DIR, 'completed_sales_622')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(COMPLETED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

import torch
from pypots.imputation import iTransformer, TimesNet, SAITS, DLinear
from pypots.optim import Adam

SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)
H_START, H_END = 6, 23; N_HOURS = H_END - H_START
WINDOW_DAYS = 30; N_WINDOWS = 3; N_STEPS = WINDOW_DAYS * N_HOURS

# ---------------------------------------------------------------------------
# High-capacity configurations (scaled up from the slim main-benchmark sizes)
#   iTransformer slim: d_model=32, heads=4, layers=2, ffn=64,  30 ep
#   TimesNet     slim: d_model=16, layers=1, ffn=32, top_k=2, k=3, 10 ep
# ---------------------------------------------------------------------------
if MODEL == 'itransformer':
    CFG = dict(d_model=128, n_heads=8, n_layers=3, d_ffn=256, dropout=0.1,
               batch_size=16, max_epochs=40, patience=6, lr=1e-3)
    DEVICE = 'mps' if torch.backends.mps.is_available() else (
        'cuda' if torch.cuda.is_available() else 'cpu')
    LABEL = 'iTransformer (hi-cap)'
elif MODEL == 'saits':
    # slim SAITS: d_model=32, heads=4, layers=2, ffn=64 -> several-fold scale-up
    CFG = dict(d_model=128, n_heads=8, n_layers=3, d_ffn=256, dropout=0.1,
               batch_size=16, max_epochs=40, patience=6, lr=1e-3)
    DEVICE = 'mps' if torch.backends.mps.is_available() else (
        'cuda' if torch.cuda.is_available() else 'cpu')
    LABEL = 'SAITS (hi-cap)'
elif MODEL == 'dlinear':
    # slim DLinear: d_model=64 -> 4x width scale-up (linear architecture)
    CFG = dict(d_model=256, moving_avg=N_HOURS // 2 * 2 + 1, individual=False,
               batch_size=128, max_epochs=40, patience=5, lr=1e-3)
    DEVICE = 'mps' if torch.backends.mps.is_available() else (
        'cuda' if torch.cuda.is_available() else 'cpu')
    LABEL = 'DLinear (hi-cap)'
else:  # timesnet
    # Scale up the dominant capacity levers (width 16->32, depth 1->2, ffn 32->64)
    # but keep top_k/n_kernels at the slim values: on CPU the FFT/inception
    # multipliers (top_k, n_kernels) dominate per-epoch cost and made the full bump
    # impractical (>40 min/epoch). This remains a ~20x parameter scale-up.
    CFG = dict(d_model=32, n_layers=2, d_ffn=64, top_k=2, n_kernels=3, dropout=0.1,
               batch_size=64, max_epochs=15, patience=3, lr=1e-3)
    DEVICE = 'cpu'   # MPS produces NaN with TimesNet (PyPOTS bug); CPU is stable
    LABEL = 'TimesNet (hi-cap)'

OUT_NAME = f'{MODEL}_hicap'

print('=' * 72)
print(f'  Capacity sweep: {LABEL}  (ore 6-22)')
print('=' * 72)
print(f'  Config: n_steps={N_STEPS}, {CFG}')
print(f'  Output: completed_sales_622/{OUT_NAME}.parquet, traccia_a_{OUT_NAME}.parquet')

# ===========================================================================
# 1. Load and prepare data (identical to 27_/28_fase_b1_*.py)
# ===========================================================================
print('\n1. Caricamento dati...')
df = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df = df.sort_values(['store_id', 'product_id', 'dt']).reset_index(drop=True)

sales_17 = np.array(df['hours_sale'].tolist(), dtype=np.float32)[:, H_START:H_END]
stock_17 = np.array(df['hours_stock_status'].tolist(), dtype=np.int8)[:, H_START:H_END]

n_rows = len(df)
n_series = n_rows // 90

sales_win = sales_17.reshape(n_series * N_WINDOWS, WINDOW_DAYS, N_HOURS)
stock_win = stock_17.reshape(n_series * N_WINDOWS, WINDOW_DAYS, N_HOURS)
n_samples = sales_win.shape[0]

sales_nan = np.where(stock_win == 1, np.nan, sales_win)

covs_cols = ['discount', 'holiday_flag', 'precpt', 'avg_temperature']
covs = df[covs_cols].values.astype(np.float32)
covs_win = covs.reshape(n_series * N_WINDOWS, WINDOW_DAYS, len(covs_cols))
covs_max = covs_win.max(axis=1, keepdims=True) + 0.1
covs_norm = covs_win / covs_max

hour_pos = np.arange(N_HOURS, dtype=np.float32)[None, None, :] / (N_HOURS - 1)
hour_pos = np.broadcast_to(hour_pos, (n_samples, WINDOW_DAYS, N_HOURS))

sales_flat = sales_nan.reshape(n_samples, N_STEPS, 1)
covs_broadcast = np.broadcast_to(covs_norm[:, :, None, :],
                                 (n_samples, WINDOW_DAYS, N_HOURS, len(covs_cols)))
covs_flat = covs_broadcast.reshape(n_samples, N_STEPS, len(covs_cols))
hour_flat = hour_pos.reshape(n_samples, N_STEPS, 1)

X_all = np.concatenate([sales_flat, covs_flat, hour_flat], axis=-1)
n_features = X_all.shape[-1]
print(f'  Samples: {n_samples:,}, Steps: {N_STEPS}, Features: {n_features}')

sales_origin = sales_win.copy()
del covs, covs_win, covs_norm, hour_pos, sales_flat, covs_broadcast, covs_flat, hour_flat
gc.collect()

# ===========================================================================
# 2. Train/val split (80/20 by series)
# ===========================================================================
print('\n2. Train/val split...')
series_indices = np.arange(n_series)
np.random.shuffle(series_indices)
n_train_series = int(n_series * 0.8)
train_series = set(series_indices[:n_train_series])

train_idx, val_idx = [], []
for s in range(n_series):
    for w in range(N_WINDOWS):
        sample_idx = s * N_WINDOWS + w
        (train_idx if s in train_series else val_idx).append(sample_idx)
train_idx = np.array(train_idx); val_idx = np.array(val_idx)

X_train = X_all[train_idx]
X_val = X_all[val_idx]

sales_origin_flat = sales_origin.reshape(n_samples, N_STEPS, 1)
X_all_ori = X_all.copy()
X_all_ori[:, :, 0:1] = sales_origin_flat
X_val_ori = X_all_ori[val_idx]
del X_all_ori, sales_origin_flat
print(f'  Train: {len(X_train):,} samples ({n_train_series:,} series)')
print(f'  Val:   {len(X_val):,} samples ({n_series - n_train_series:,} series)')

# ===========================================================================
# 3. Train with early stopping
# ===========================================================================
print(f'\n3. Training {LABEL} (patience={CFG["patience"]}, max {CFG["max_epochs"]} epochs)...')
print(f'  Device: {DEVICE}')
saving_path = os.path.join(RESULTS_DIR, f'pypots_{OUT_NAME}_val')

def make_model(device):
    optimizer = Adam(lr=CFG['lr'], weight_decay=1e-5)
    if MODEL == 'itransformer':
        return iTransformer(
            n_steps=N_STEPS, n_features=n_features,
            n_layers=CFG['n_layers'], d_model=CFG['d_model'], n_heads=CFG['n_heads'],
            d_k=CFG['d_model'] // CFG['n_heads'], d_v=CFG['d_model'] // CFG['n_heads'],
            d_ffn=CFG['d_ffn'], dropout=CFG['dropout'], attn_dropout=CFG['dropout'],
            ORT_weight=1, MIT_weight=1,
            batch_size=CFG['batch_size'], epochs=CFG['max_epochs'],
            patience=CFG['patience'], optimizer=optimizer,
            device=device, saving_path=saving_path, verbose=True)
    elif MODEL == 'saits':
        return SAITS(
            n_steps=N_STEPS, n_features=n_features,
            n_layers=CFG['n_layers'], d_model=CFG['d_model'], n_heads=CFG['n_heads'],
            d_k=CFG['d_model'] // CFG['n_heads'], d_v=CFG['d_model'] // CFG['n_heads'],
            d_ffn=CFG['d_ffn'], dropout=CFG['dropout'], attn_dropout=CFG['dropout'],
            ORT_weight=1, MIT_weight=1,
            batch_size=CFG['batch_size'], epochs=CFG['max_epochs'],
            patience=CFG['patience'], optimizer=optimizer,
            device=device, saving_path=saving_path, verbose=True)
    elif MODEL == 'dlinear':
        return DLinear(
            n_steps=N_STEPS, n_features=n_features,
            moving_avg_window_size=CFG['moving_avg'],
            individual=CFG['individual'], d_model=CFG['d_model'],
            ORT_weight=1, MIT_weight=1,
            batch_size=CFG['batch_size'], epochs=CFG['max_epochs'],
            patience=CFG['patience'], optimizer=optimizer,
            device=device, saving_path=saving_path, verbose=True)
    else:
        return TimesNet(
            n_steps=N_STEPS, n_features=n_features,
            n_layers=CFG['n_layers'], top_k=CFG['top_k'], d_model=CFG['d_model'],
            d_ffn=CFG['d_ffn'], n_kernels=CFG['n_kernels'], dropout=CFG['dropout'],
            batch_size=CFG['batch_size'], epochs=CFG['max_epochs'],
            patience=CFG['patience'], optimizer=optimizer,
            device=device, saving_path=saving_path, verbose=True)

model = make_model(DEVICE)
print(f'  Model params: {sum(p.numel() for p in model.model.parameters() if p.requires_grad):,}')

t0 = time.time()
try:
    model.fit(train_set={'X': X_train}, val_set={'X': X_val, 'X_ori': X_val_ori})
    print(f'  Training time: {time.time()-t0:.0f}s')
except RuntimeError as e:
    if 'out of memory' in str(e).lower() or 'mps' in str(e).lower():
        print(f'\n  OOM su {DEVICE}. Riprovo su CPU...')
        if DEVICE == 'mps':
            torch.mps.empty_cache()
        DEVICE = 'cpu'
        model = make_model(DEVICE)
        t0 = time.time()
        model.fit(train_set={'X': X_train}, val_set={'X': X_val, 'X_ori': X_val_ori})
        print(f'  Training time CPU: {time.time()-t0:.0f}s')
    else:
        raise

del X_train, X_val
gc.collect()

# ===========================================================================
# 4. Predict on all data
# ===========================================================================
print(f'\n4. Predict su tutti i {n_samples:,} samples...')
PREDICT_BATCH = 5000
imputed_flat = np.zeros((n_samples, N_STEPS), dtype=np.float32)
for start in range(0, n_samples, PREDICT_BATCH):
    end = min(start + PREDICT_BATCH, n_samples)
    res = model.predict({'X': X_all[start:end]})
    imp = res['imputation']
    if len(imp.shape) == 4:
        imp = imp.mean(axis=1)
    imputed_flat[start:end] = np.clip(imp[:, :, 0], 0, None)
    del res, imp; gc.collect()
    if DEVICE == 'mps':
        torch.mps.empty_cache()
imputed_sales = imputed_flat.reshape(n_samples, WINDOW_DAYS, N_HOURS)
print(f'  Mean imputed: {imputed_sales.mean():.4f}, '
      f'Mean original (non-NaN): {sales_origin[~np.isnan(sales_nan)].mean():.4f}')

# ===========================================================================
# 5. Eval MNAR (Track A)
# ===========================================================================
print('\n5. Valutazione MNAR...')
masks_val_df = pd.read_parquet(os.path.join(DATA_DIR, 'mnar_masks_val.parquet'))
masks_val_df = masks_val_df[(masks_val_df['hour'] >= H_START) &
                            (masks_val_df['hour'] < H_END)].reset_index(drop=True)
print(f'  MNAR masks: {len(masks_val_df):,}')

df['row_idx'] = np.arange(n_rows)
rl = df.set_index(['store_id', 'product_id', 'dt'])['row_idx']

X_mnar = X_all.copy()
n_added = 0
for _, r in masks_val_df.iterrows():
    key = (r['store_id'], r['product_id'], r['dt'])
    if key in rl.index:
        row_idx = rl[key]
        if isinstance(row_idx, pd.Series):
            row_idx = row_idx.iloc[0]
        si = row_idx // 90; di = row_idx % 90
        wi = di // WINDOW_DAYS; dw = di % WINDOW_DAYS
        samp = si * N_WINDOWS + wi
        step = dw * N_HOURS + (int(r['hour']) - H_START)
        X_mnar[samp, step, 0] = np.nan
        n_added += 1
print(f'  MNAR NaN aggiunti: {n_added:,}')

imputed_mnar_flat = np.zeros((n_samples, N_STEPS), dtype=np.float32)
for start in range(0, n_samples, PREDICT_BATCH):
    end = min(start + PREDICT_BATCH, n_samples)
    res = model.predict({'X': X_mnar[start:end]})
    imp = res['imputation']
    if len(imp.shape) == 4:
        imp = imp.mean(axis=1)
    imputed_mnar_flat[start:end] = np.clip(imp[:, :, 0], 0, None)
    del res, imp; gc.collect()
    if DEVICE == 'mps':
        torch.mps.empty_cache()
imputed_mnar = imputed_mnar_flat.reshape(n_samples, WINDOW_DAYS, N_HOURS)

preds_mnar = np.zeros(len(masks_val_df), dtype=np.float64)
gt = masks_val_df['ground_truth'].values.astype(np.float64)
for i in range(len(masks_val_df)):
    r = masks_val_df.iloc[i]
    key = (r['store_id'], r['product_id'], r['dt'])
    if key in rl.index:
        row_idx = rl[key]
        if isinstance(row_idx, pd.Series):
            row_idx = row_idx.iloc[0]
        si = row_idx // 90; di = row_idx % 90
        wi = di // WINDOW_DAYS; dw = di % WINDOW_DAYS
        samp = si * N_WINDOWS + wi
        hi = int(r['hour']) - H_START
        preds_mnar[i] = imputed_mnar[samp, dw, hi]

sao = np.abs(gt).sum()
wape = np.abs(preds_mnar - gt).sum() / sao if sao > 0 else np.nan
wpe = (preds_mnar - gt).sum() / gt.sum() if gt.sum() != 0 else np.nan
print(f'  {LABEL}: WAPE_recovery={wape:.4f}, WPE_recovery={wpe:.4f}')

ta_df = pd.DataFrame([{'imputer': OUT_NAME, 'wape_recovery': wape, 'wpe_recovery': wpe}])
ta_df.to_parquet(os.path.join(RESULTS_DIR, f'traccia_a_{OUT_NAME}.parquet'), index=False)

# ===========================================================================
# 6. Save completed_sales
# ===========================================================================
print('\n6. Salvataggio completed_sales...')
imputed_rows = imputed_sales.reshape(n_rows, N_HOURS)
completed = sales_17.copy()
so_mask = stock_17 == 1
completed[so_mask] = np.clip(imputed_rows[so_mask], 0, None)

df_out = df[['store_id', 'product_id', 'dt']].copy()
df_out['dt_parsed'] = pd.to_datetime(df_out['dt'])
all_dates = sorted(df_out['dt_parsed'].unique())
date_to_day = {d: i + 1 for i, d in enumerate(all_dates)}
df_out['day_num'] = df_out['dt_parsed'].map(date_to_day)
df_out['dow'] = df_out['dt_parsed'].dt.dayofweek
df_out['hours_sale'] = list(completed)
df_out['hours_stock_status'] = list(stock_17)
df_out.drop(columns=['dt_parsed', 'row_idx'], inplace=True, errors='ignore')

out_path = os.path.join(COMPLETED_DIR, f'{OUT_NAME}.parquet')
df_out.to_parquet(out_path, index=False)
print(f'  Salvato: {out_path}')
print(f'  Media imputata (stockout): {completed[so_mask].mean():.4f}')
print(f'  Media S_obs (in-stock):    {sales_17[~so_mask].mean():.4f}')

print('\n' + '=' * 72)
print(f'  DONE — {LABEL}')
print('=' * 72)
