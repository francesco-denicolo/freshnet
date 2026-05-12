"""
28_fase_b1_imputation_timesnet.py — TimesNet imputer (ore 6-22)
================================================================
Adattato da 16_fase_b1_imputation_saits.py.

TimesNet (Wu et al. ICLR 2023) usa decomposizione multi-periodicity via inception.
Nel paper FreshRetailNet baseline è il best imputer in Table 2 (Multi-Periodicity Models).
"""
import sys, os, gc, time, functools, warnings
import numpy as np, pandas as pd

warnings.filterwarnings('ignore')
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
COMPLETED_DIR = os.path.join(DATA_DIR, 'completed_sales_622')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(COMPLETED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

import torch
from pypots.imputation import TimesNet
from pypots.optim import Adam

SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)
H_START, H_END = 6, 23; N_HOURS = H_END - H_START
WINDOW_DAYS = 30; N_WINDOWS = 3; N_STEPS = WINDOW_DAYS * N_HOURS

DEVICE = 'mps' if torch.backends.mps.is_available() else (
    'cuda' if torch.cuda.is_available() else 'cpu')

# TimesNet hyperparameters (small per coerenza con SAITS)
D_MODEL = 32
N_LAYERS = 2
D_FFN = 64
TOP_K = 3        # periodicità da considerare nel decomposition
N_KERNELS = 6    # kernel inception per TimesBlock
DROPOUT = 0.1
BATCH_SIZE = 16
MAX_EPOCHS = 30
PATIENCE = 5
LR = 1e-3

print('=' * 72)
print('  TimesNet (small) imputer (ore 6-22)')
print('=' * 72)
print(f'  Config: n_steps={N_STEPS}, d_model={D_MODEL}, n_layers={N_LAYERS}, '
      f'd_ffn={D_FFN}, top_k={TOP_K}, n_kernels={N_KERNELS}, batch={BATCH_SIZE}')

# ===========================================================================
# 1. Load and prepare data
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

# Features
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
        if s in train_series:
            train_idx.append(sample_idx)
        else:
            val_idx.append(sample_idx)

train_idx = np.array(train_idx)
val_idx = np.array(val_idx)

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
# 3. Train TimesNet with early stopping
# ===========================================================================
print(f'\n3. Training TimesNet (patience={PATIENCE}, max {MAX_EPOCHS} epochs)...')
print(f'  Device: {DEVICE}')

optimizer = Adam(lr=LR, weight_decay=1e-5)
saving_path = os.path.join(RESULTS_DIR, 'pypots_timesnet_val')

def make_model(device):
    return TimesNet(
        n_steps=N_STEPS, n_features=n_features,
        n_layers=N_LAYERS, top_k=TOP_K, d_model=D_MODEL,
        d_ffn=D_FFN, n_kernels=N_KERNELS,
        dropout=DROPOUT,
        batch_size=BATCH_SIZE, epochs=MAX_EPOCHS,
        patience=PATIENCE, optimizer=optimizer,
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
    print(f'  {start:,}-{end:,} / {n_samples:,}...')
    res = model.predict({'X': X_all[start:end]})
    imp = res['imputation']
    if len(imp.shape) == 4:
        imp = imp.mean(axis=1)
    imputed_flat[start:end] = np.clip(imp[:, :, 0], 0, None)
    del res, imp
    gc.collect()
    if DEVICE == 'mps':
        torch.mps.empty_cache()

imputed_sales = imputed_flat.reshape(n_samples, WINDOW_DAYS, N_HOURS)
print(f'  Mean imputed: {imputed_sales.mean():.4f}, '
      f'Mean original (non-NaN): {sales_origin[~np.isnan(sales_nan)].mean():.4f}')

# ===========================================================================
# 5. Eval MNAR (Traccia A)
# ===========================================================================
print(f'\n5. Valutazione MNAR...')

masks_val_df = pd.read_parquet(os.path.join(DATA_DIR, 'mnar_masks_val.parquet'))
masks_val_df = masks_val_df[(masks_val_df['hour'] >= H_START) & (masks_val_df['hour'] < H_END)].reset_index(drop=True)
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
        si = row_idx // 90
        di = row_idx % 90
        wi = di // WINDOW_DAYS
        dw = di % WINDOW_DAYS
        samp = si * N_WINDOWS + wi
        hi = int(r['hour']) - H_START
        step = dw * N_HOURS + hi
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
    del res, imp
    gc.collect()
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
        si = row_idx // 90
        di = row_idx % 90
        wi = di // WINDOW_DAYS
        dw = di % WINDOW_DAYS
        samp = si * N_WINDOWS + wi
        hi = int(r['hour']) - H_START
        preds_mnar[i] = imputed_mnar[samp, dw, hi]

sao = np.abs(gt).sum()
wape = np.abs(preds_mnar - gt).sum() / sao if sao > 0 else np.nan
wpe = (preds_mnar - gt).sum() / gt.sum() if gt.sum() != 0 else np.nan
print(f'  TimesNet (val-tuned): WAPE_recovery={wape:.4f}, WPE_recovery={wpe:.4f}')

ta_df = pd.DataFrame([{'imputer': 'TimesNet', 'wape_recovery': wape, 'wpe_recovery': wpe}])
ta_df.to_parquet(os.path.join(RESULTS_DIR, 'traccia_a_timesnet.parquet'), index=False)

# ===========================================================================
# 6. Save completed_sales
# ===========================================================================
print(f'\n6. Salvataggio completed_sales...')
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

out_path = os.path.join(COMPLETED_DIR, 'timesnet.parquet')
df_out.to_parquet(out_path, index=False)
print(f'  Salvato: {out_path}')
print(f'  Media imputata (stockout): {completed[so_mask].mean():.4f}')
print(f'  Media S_obs (in-stock):    {sales_17[~so_mask].mean():.4f}')

print('\n' + '=' * 72)
print('  DONE — TimesNet imputer')
print('=' * 72)
