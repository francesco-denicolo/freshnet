"""
31_hpo_tft.py — Hyperparameter Optimization per TFT (Optuna TPE + MedianPruner)
================================================================================
HPO su tutte le 50K serie, S_obs RAW (no imputation).
Train: gg 1-83, val: gg 84-90 in-stock filter, metrica WAPE_med per-serie (min_hours=34).
30 trial. MedianPruner. SQLite storage (resume on crash).
"""
import sys, os, gc, time, json, functools, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
print = functools.partial(print, flush=True)

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE
from torch.utils.data import DataLoader
import optuna
from optuna.integration import PyTorchLightningPruningCallback

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED); pl.seed_everything(SEED)

# --- Setup data parameters ---
H_START, H_END = 6, 23; N_HOURS = H_END - H_START
ENCODER_LENGTH = 119; PRED_LENGTH = 119
TRAINING_CUTOFF = 83 * N_HOURS - 1   # idx 1410
VAL_CUTOFF = 90 * N_HOURS - 1        # idx 1529
MIN_HOURS_VAL = 34
MAX_TRAIN_SAMPLES = 200_000           # subsampling delle finestre training
DEVICE = 'cpu'

# --- HPO config ---
N_TRIALS = 30
STUDY_NAME = 'hpo_tft'
STORAGE = f'sqlite:///{RESULTS_DIR}/hpo_tft.db'

# --- Fixed HP ---
MAX_EPOCHS = 25
PATIENCE = 5

# --- Smoke test mode (env: HPO_SMOKE=1) ---
if os.getenv('HPO_SMOKE') == '1':
    N_TRIALS = 2
    MAX_EPOCHS = 3
    PATIENCE = 2
    STUDY_NAME = STUDY_NAME + '_smoke'
    STORAGE = STORAGE.replace('.db', '_smoke.db')
    print('*** SMOKE TEST MODE: N_TRIALS=2, MAX_EPOCHS=3 ***')

T_START = time.time()
print('=' * 72)
print('  HPO TFT — 30 trial, full 50K, MAE loss')
print('=' * 72)

# =========================================================================
# 1. Setup dati (UNA SOLA VOLTA, cache su disco)
# =========================================================================
CACHE_PATH = os.path.join(RESULTS_DIR, 'hpo_tft_long_data_cache.parquet')
if os.path.exists(CACHE_PATH):
    print(f'[{time.time()-T_START:.0f}s] Loading long_data from cache...')
    long_data = pd.read_parquet(CACHE_PATH)
else:
    print(f'[{time.time()-T_START:.0f}s] Building long_data (first time)...')
    df_train = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
    df_eval = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_eval.parquet'))
    df_full = pd.concat([df_train, df_eval], ignore_index=True)
    df_full['dt_parsed'] = pd.to_datetime(df_full['dt'])
    df_full = df_full.sort_values(['store_id','product_id','dt_parsed']).reset_index(drop=True)
    all_dates = sorted(df_full['dt_parsed'].unique())
    date_to_day = {d: i for i, d in enumerate(all_dates)}
    df_full['day_num'] = df_full['dt_parsed'].map(date_to_day)
    df_full['dow'] = df_full['dt_parsed'].dt.dayofweek

    sales_arr = np.array(df_full['hours_sale'].tolist(), dtype=np.float32)[:, H_START:H_END]
    stock_arr = np.array(df_full['hours_stock_status'].tolist(), dtype=np.int8)[:, H_START:H_END]

    n_rows = len(df_full)
    print(f'[{time.time()-T_START:.0f}s]   Full data: {n_rows:,} righe, sales {sales_arr.shape}')

    print(f'[{time.time()-T_START:.0f}s] Long format orario...')
    long_data = pd.DataFrame({
        'store_id':         np.repeat(df_full['store_id'].values, N_HOURS),
        'product_id':       np.repeat(df_full['product_id'].values, N_HOURS),
        'city_id':          np.repeat(df_full['city_id'].values, N_HOURS),
        'day_num':          np.repeat(df_full['day_num'].values, N_HOURS),
        'dow':              np.repeat(df_full['dow'].values, N_HOURS),
        'discount':         np.repeat(df_full['discount'].values, N_HOURS),
        'avg_temperature':  np.repeat(df_full['avg_temperature'].values, N_HOURS),
        'avg_humidity':     np.repeat(df_full['avg_humidity'].values, N_HOURS),
        'precpt':           np.repeat(df_full['precpt'].values, N_HOURS),
        'avg_wind_level':   np.repeat(df_full['avg_wind_level'].values, N_HOURS),
        'holiday_flag':     np.repeat(df_full['holiday_flag'].values, N_HOURS),
        'activity_flag':    np.repeat(df_full['activity_flag'].values, N_HOURS),
        'hour':             np.tile(np.arange(H_START, H_END), n_rows),
        'sales':            sales_arr.reshape(-1).astype(np.float32),
        'stock':            stock_arr.reshape(-1).astype(np.int8),
    })
    long_data['time_idx'] = long_data['day_num'] * N_HOURS + (long_data['hour'] - H_START)

    # Dtype compression
    print(f'[{time.time()-T_START:.0f}s]   Compressing dtypes...')
    for c in ['store_id','product_id','city_id','dow','hour','holiday_flag','activity_flag']:
        long_data[c] = long_data[c].astype('category')
    for c in ['discount','avg_temperature','avg_humidity','precpt','avg_wind_level','sales']:
        long_data[c] = long_data[c].astype('float32')
    long_data['stock'] = long_data['stock'].astype('int8')
    long_data['day_num'] = long_data['day_num'].astype('int16')
    long_data['time_idx'] = long_data['time_idx'].astype('int32')

    long_data.to_parquet(CACHE_PATH, index=False)
    print(f'[{time.time()-T_START:.0f}s]   Long_data cached: {long_data.shape}')
    del df_train, df_eval, df_full, sales_arr, stock_arr; gc.collect()

print(f'[{time.time()-T_START:.0f}s] Long_data shape: {long_data.shape}')

# =========================================================================
# 2. Build TimeSeriesDataSet (UNA SOLA VOLTA, riusato per ogni trial)
# =========================================================================
print(f'\n[{time.time()-T_START:.0f}s] Build TimeSeriesDataSet (UNA SOLA VOLTA)...')

# Training: idx <= TRAINING_CUTOFF
t0 = time.time()
training = TimeSeriesDataSet(
    long_data[long_data.time_idx <= TRAINING_CUTOFF],
    time_idx='time_idx', target='sales', group_ids=['store_id','product_id'],
    min_encoder_length=ENCODER_LENGTH, max_encoder_length=ENCODER_LENGTH,
    min_prediction_length=PRED_LENGTH, max_prediction_length=PRED_LENGTH,
    static_categoricals=['store_id','product_id','city_id'],
    time_varying_known_categoricals=['dow','hour','holiday_flag','activity_flag'],
    time_varying_known_reals=['discount','avg_temperature','avg_humidity','precpt','avg_wind_level'],
    time_varying_unknown_reals=['sales'],
    target_normalizer=GroupNormalizer(groups=['store_id','product_id'], transformation='softplus'),
    add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
    allow_missing_timesteps=True,
)
print(f'[{time.time()-T_START:.0f}s]   Training: {len(training):,} samples (build {time.time()-t0:.0f}s)')

t0 = time.time()
validation = TimeSeriesDataSet.from_dataset(
    training, long_data[long_data.time_idx <= VAL_CUTOFF],
    predict=True, stop_randomization=True
)
print(f'[{time.time()-T_START:.0f}s]   Validation: {len(validation):,} samples (build {time.time()-t0:.0f}s)')

# Subsampling training (200K)
N_TRAINING = len(training)
if N_TRAINING > MAX_TRAIN_SAMPLES:
    rng = np.random.RandomState(SEED)
    idx_subset = rng.choice(N_TRAINING, MAX_TRAIN_SAMPLES, replace=False)
    training_sub = torch.utils.data.Subset(training, idx_subset.tolist())
    print(f'[{time.time()-T_START:.0f}s]   Subsampled training to {MAX_TRAIN_SAMPLES:,} samples')
else:
    training_sub = training
    print(f'[{time.time()-T_START:.0f}s]   Using full {N_TRAINING:,} training samples')

# =========================================================================
# 3. WAPE_med computation function (per-serie, in-stock, min_hours=34)
# =========================================================================
def compute_wape_med_val(model, val_loader, val_data):
    """Calcola WAPE_med per-serie su val (gg 84-90), in-stock, min_hours=34."""
    preds = model.predict(val_loader, return_y=False, mode='prediction',
                          trainer_kwargs={'accelerator':'cpu','devices':1,'logger':False,
                                          'enable_progress_bar':False})
    if hasattr(preds, 'cpu'):
        preds = preds.cpu().numpy()
    preds = np.maximum(preds, 0.0)  # softplus output, ma per safety

    # Recupero ground truth e stock mask per ogni serie
    val_index = validation.x_to_index(next(iter(val_loader)))
    # Più semplice: usa l'attributo .decoded_index
    # Ricostruisco da long_data
    val_filtered = long_data[(long_data.time_idx > TRAINING_CUTOFF) &
                              (long_data.time_idx <= VAL_CUTOFF)].copy()
    # Per ogni (store, product), prendi 119 ore consecutive del val period
    series_keys = val_filtered.groupby(['store_id','product_id']).size()
    series_with_full_val = series_keys[series_keys == PRED_LENGTH].index.tolist()
    # Costruisci array predizioni e ground truth per ogni serie
    # NOTA: preds è in ordine consistente con val_loader, che è in ordine di TimeSeriesDataSet
    # Per semplicità, recuperiamo l'ordine via attributo internals
    val_index_df = validation.x_to_index(next(iter(DataLoader(validation, batch_size=len(validation), num_workers=0))))
    wapes = []
    n_preds = preds.shape[0]
    for i in range(n_preds):
        sid = val_index_df.iloc[i]['store_id']
        pid = val_index_df.iloc[i]['product_id']
        series_data = val_filtered[
            (val_filtered['store_id']==sid) & (val_filtered['product_id']==pid)
        ].sort_values('time_idx')
        if len(series_data) < PRED_LENGTH:
            continue
        y_true = series_data['sales'].values[-PRED_LENGTH:]
        stock = series_data['stock'].values[-PRED_LENGTH:]
        in_stock = stock == 0
        if in_stock.sum() < MIN_HOURS_VAL:
            continue
        y_pred = preds[i][-PRED_LENGTH:]
        y_t, y_p = y_true[in_stock], y_pred[in_stock]
        denom = max(np.abs(y_t).sum(), 1e-8)
        wape = np.abs(y_t - y_p).sum() / denom
        wapes.append(wape)
    return float(np.median(wapes)) if wapes else float('nan')

# =========================================================================
# 4. Optuna objective
# =========================================================================
def objective(trial):
    t_trial = time.time()
    # HP space
    head_dim = trial.suggest_categorical('head_dim', [8, 16, 32])
    n_heads = trial.suggest_categorical('attention_heads', [2, 4, 8])
    hidden_size = head_dim * n_heads
    dropout = trial.suggest_float('dropout', 0.0, 0.3, step=0.05)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    print(f'\n[Trial {trial.number}] HP: head_dim={head_dim}, heads={n_heads}, hidden={hidden_size}, '
          f'dropout={dropout:.2f}, lr={lr:.1e}, batch={batch_size}, wd={weight_decay:.1e}')

    # DataLoader
    train_loader = DataLoader(training_sub, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    tft = TemporalFusionTransformer.from_dataset(
        training, learning_rate=lr, hidden_size=hidden_size, attention_head_size=n_heads,
        dropout=dropout, hidden_continuous_size=min(hidden_size, 8),
        output_size=1, loss=MAE(), log_interval=0,
        reduce_on_plateau_patience=2, optimizer='adam', weight_decay=weight_decay
    )

    # Trainer with pruning callback
    pruning_cb = PyTorchLightningPruningCallback(trial, monitor='val_loss')
    early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE, mode='min', min_delta=1e-4)
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS, accelerator='cpu', devices=1,
        callbacks=[pruning_cb, early_stop],
        gradient_clip_val=0.1, enable_checkpointing=False, enable_progress_bar=False,
        logger=False, deterministic=False,
    )

    try:
        trainer.fit(tft, train_loader, val_loader)
    except optuna.TrialPruned:
        print(f'[Trial {trial.number}] PRUNED at epoch {trainer.current_epoch}')
        raise

    # Evaluate val WAPE_med
    val_wape = compute_wape_med_val(tft, val_loader, validation)
    elapsed = time.time() - t_trial
    print(f'[Trial {trial.number}] val_WAPE_med={val_wape:.4f}, '
          f'best_epoch~{trainer.current_epoch - PATIENCE}, elapsed={elapsed:.0f}s')
    return val_wape

# =========================================================================
# 5. Run Optuna study
# =========================================================================
print(f'\n[{time.time()-T_START:.0f}s] Creating Optuna study...')
study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=5),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2, interval_steps=1),
    storage=STORAGE,
    study_name=STUDY_NAME,
    load_if_exists=True,
)

print(f'  Existing trials: {len(study.trials)}')
remaining = max(0, N_TRIALS - len(study.trials))
print(f'  Remaining to run: {remaining}')

if remaining > 0:
    study.optimize(objective, n_trials=remaining, gc_after_trial=True)

# =========================================================================
# 6. Save results
# =========================================================================
print(f'\n[{time.time()-T_START:.0f}s] Saving results...')
best = study.best_trial
print(f'  Best trial: #{best.number}')
print(f'  Best val_WAPE_med: {best.value:.4f}')
print(f'  Best params: {best.params}')

with open(os.path.join(RESULTS_DIR, 'hpo_tft_best.json'), 'w') as f:
    json.dump({'best_trial': best.number, 'best_value': best.value,
               'best_params': best.params, 'n_trials': len(study.trials)}, f, indent=2)

trials_df = study.trials_dataframe()
trials_df.to_parquet(os.path.join(RESULTS_DIR, 'hpo_tft_trials.parquet'), index=False)

import pickle
with open(os.path.join(RESULTS_DIR, 'hpo_tft_study.pkl'), 'wb') as f:
    pickle.dump(study, f)

print(f'\n[{time.time()-T_START:.0f}s] DONE in {(time.time()-T_START)/60:.1f} min')
print('=' * 72)
