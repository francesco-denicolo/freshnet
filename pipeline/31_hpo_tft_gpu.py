"""
31_hpo_tft_gpu.py — HPO TFT su GPU cloud (Optuna TPE + MedianPruner)
====================================================================
Versione GPU dello HPO del TFT. Rispetto a 31_hpo_tft.py (CPU, 16 GB):
  - NIENTE cap `hidden_size > 32` (era il vincolo che handicappava il TFT su 16 GB).
  - Spazio HP AMPIO: head_dim {8,16,32,64}, heads {2,4,8} -> hidden fino a 256.
  - accelerator='gpu' con auto-detect CUDA (fallback CPU).
  - Budget pieno: MAX_EPOCHS=30, PATIENCE=5, MAX_TRAIN_SAMPLES=400K, ~48 trial.
  - Study NUOVO (hpo_tft_gpu) per NON ereditare i trial crippled del run CPU.

HPO su tutte le 50K serie, S_obs RAW (no imputation).
Train gg 1-83, val gg 84-90 in-stock, metrica WAPE_med per-serie (min_hours=34).
SQLite storage (resume on crash).
"""
import sys, os, gc, time, json, functools, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
print = functools.partial(print, flush=True)

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE
from torch.utils.data import DataLoader
import optuna
try:
    from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback
except ImportError:
    from optuna.integration import PyTorchLightningPruningCallback

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED); pl.seed_everything(SEED)

# --- Device auto-detect (GPU se disponibile, altrimenti CPU) ---
if torch.cuda.is_available():
    ACCEL, DEVICES = 'gpu', 1
    PRECISION = os.getenv('TFT_PRECISION', '32-true')   # '16-mixed' per più velocità/VRAM
    print(f'*** CUDA: {torch.cuda.get_device_name(0)} | precision={PRECISION} ***')
else:
    ACCEL, DEVICES, PRECISION = 'cpu', 1, '32-true'
    print('*** No CUDA -> CPU fallback (lento). ***')

# --- Data params ---
H_START, H_END = 6, 23; N_HOURS = H_END - H_START
ENCODER_LENGTH = 119; PRED_LENGTH = 119
TRAINING_CUTOFF = 83 * N_HOURS - 1
VAL_CUTOFF = 90 * N_HOURS - 1
MIN_HOURS_VAL = 34
MAX_TRAIN_SAMPLES = int(os.getenv('TFT_MAX_TRAIN', 400_000))

# --- HPO config (AMPIA) ---
N_TRIALS = int(os.getenv('TFT_N_TRIALS', 48))
HIDDEN_CAP = int(os.getenv('TFT_HIDDEN_CAP', 256))       # 'ampia' = hidden fino a 256
STUDY_NAME = 'hpo_tft_gpu'
STORAGE = f'sqlite:///{RESULTS_DIR}/hpo_tft_gpu.db'

# --- Budget training pieno ---
MAX_EPOCHS = int(os.getenv('TFT_MAX_EPOCHS', 30))
PATIENCE = int(os.getenv('TFT_PATIENCE', 5))

# --- Smoke test (env: HPO_SMOKE=1) ---
if os.getenv('HPO_SMOKE') == '1':
    N_TRIALS, MAX_EPOCHS, PATIENCE = 2, 3, 2
    MAX_TRAIN_SAMPLES = 20_000
    STUDY_NAME += '_smoke'; STORAGE = STORAGE.replace('.db', '_smoke.db')
    print('*** SMOKE TEST MODE ***')

T_START = time.time()
print('=' * 72)
print(f'  HPO TFT GPU — {N_TRIALS} trial | hidden<= {HIDDEN_CAP} | '
      f'epochs<= {MAX_EPOCHS} | max_train={MAX_TRAIN_SAMPLES:,}')
print('=' * 72)

# =========================================================================
# 1. long_data (cache condivisa con la versione CPU, formato identico)
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
    for c in ['store_id','product_id','city_id','dow','hour','holiday_flag','activity_flag']:
        long_data[c] = long_data[c].astype(str).astype('category')
    for c in ['discount','avg_temperature','avg_humidity','precpt','avg_wind_level','sales']:
        long_data[c] = long_data[c].astype('float32')
    long_data['stock'] = long_data['stock'].astype('int8')
    long_data['day_num'] = long_data['day_num'].astype('int16')
    long_data['time_idx'] = long_data['time_idx'].astype('int32')
    long_data.to_parquet(CACHE_PATH, index=False)
    del df_train, df_eval, df_full, sales_arr, stock_arr; gc.collect()
print(f'[{time.time()-T_START:.0f}s] Long_data shape: {long_data.shape}')

# =========================================================================
# 2. TimeSeriesDataSet (cache condivisa)
# =========================================================================
TSD_TRAIN_CACHE = os.path.join(RESULTS_DIR, 'hpo_tft_tsd_train.pkl')
TSD_VAL_CACHE = os.path.join(RESULTS_DIR, 'hpo_tft_tsd_val.pkl')
if os.path.exists(TSD_TRAIN_CACHE) and os.path.exists(TSD_VAL_CACHE):
    print(f'[{time.time()-T_START:.0f}s] Loading TimeSeriesDataSet from cache...')
    _orig = torch.load
    torch.load = lambda *a, **kw: _orig(*a, **{**kw, 'weights_only': False})
    try:
        training = TimeSeriesDataSet.load(TSD_TRAIN_CACHE)
        validation = TimeSeriesDataSet.load(TSD_VAL_CACHE)
    finally:
        torch.load = _orig
else:
    print(f'[{time.time()-T_START:.0f}s] Build TimeSeriesDataSet (cache miss)...')
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
    validation = TimeSeriesDataSet.from_dataset(
        training, long_data[long_data.time_idx <= VAL_CUTOFF], predict=True, stop_randomization=True)
    training.save(TSD_TRAIN_CACHE); validation.save(TSD_VAL_CACHE)
print(f'[{time.time()-T_START:.0f}s]   Training: {len(training):,} | Validation: {len(validation):,}')

N_TRAINING = len(training)
if N_TRAINING > MAX_TRAIN_SAMPLES:
    rng = np.random.RandomState(SEED)
    idx_subset = rng.choice(N_TRAINING, MAX_TRAIN_SAMPLES, replace=False)
    training_sub = torch.utils.data.Subset(training, idx_subset.tolist())
    print(f'[{time.time()-T_START:.0f}s]   Subsampled training to {MAX_TRAIN_SAMPLES:,}')
else:
    training_sub = training

# =========================================================================
# 3. Val stock mask (una volta)
# =========================================================================
val_long = long_data[(long_data['time_idx'] > TRAINING_CUTOFF) & (long_data['time_idx'] <= VAL_CUTOFF)].copy()
val_long_sorted = val_long.sort_values(['store_id','product_id','time_idx'])
val_stock_by_serie = {}
for (sid, pid), grp in val_long_sorted.groupby(['store_id','product_id'], sort=False):
    val_stock_by_serie[(int(sid), int(pid))] = grp['stock'].values[:PRED_LENGTH]
print(f'[{time.time()-T_START:.0f}s]   Stock mask cached for {len(val_stock_by_serie):,} serie')
del val_long, val_long_sorted; gc.collect()

def compute_wape_med_val(model, val_loader):
    res = model.predict(
        val_loader, return_y=True, return_index=True, mode='prediction',
        trainer_kwargs={'accelerator': ACCEL, 'devices': DEVICES, 'logger': False,
                        'enable_progress_bar': False})
    preds = np.clip(res.output.cpu().numpy(), 0, None)
    truths = res.y[0].cpu().numpy()
    idx_df = res.index
    wapes = []
    for i in range(len(idx_df)):
        sid, pid = int(idx_df.iloc[i]['store_id']), int(idx_df.iloc[i]['product_id'])
        if (sid, pid) not in val_stock_by_serie: continue
        stk = val_stock_by_serie[(sid, pid)]
        if len(stk) < PRED_LENGTH: continue
        in_stock = stk == 0
        if in_stock.sum() < MIN_HOURS_VAL: continue
        p_in, t_in = preds[i][in_stock], truths[i][in_stock]
        wapes.append(np.abs(p_in - t_in).sum() / max(np.abs(t_in).sum(), 1e-8))
    return float(np.median(wapes)) if wapes else float('nan')

# =========================================================================
# 4. Optuna objective (spazio AMPIO, nessun cap a 32)
# =========================================================================
def objective(trial):
    t_trial = time.time()
    head_dim = trial.suggest_categorical('head_dim', [8, 16, 32, 64])
    n_heads = trial.suggest_categorical('attention_heads', [2, 4, 8])
    hidden_size = head_dim * n_heads
    dropout = trial.suggest_float('dropout', 0.0, 0.3, step=0.05)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [512, 1024, 2048])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    # unico vincolo: rispetta il budget 'ampia' (hidden <= HIDDEN_CAP). NON per OOM, per costo.
    if hidden_size > HIDDEN_CAP:
        print(f'\n[Trial {trial.number}] SKIP hidden={hidden_size} > cap {HIDDEN_CAP}')
        raise optuna.TrialPruned()

    print(f'\n[Trial {trial.number}] head_dim={head_dim} heads={n_heads} hidden={hidden_size} '
          f'dropout={dropout:.2f} lr={lr:.1e} batch={batch_size} wd={weight_decay:.1e}')

    train_loader = DataLoader(training_sub, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=training._collate_fn)
    val_loader = validation.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        training, learning_rate=lr, hidden_size=hidden_size, attention_head_size=n_heads,
        dropout=dropout, hidden_continuous_size=min(hidden_size, 16),
        output_size=1, loss=MAE(), log_interval=0,
        reduce_on_plateau_patience=2, optimizer='adam', weight_decay=weight_decay)

    pruning_cb = PyTorchLightningPruningCallback(trial, monitor='val_loss')
    early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE, mode='min', min_delta=1e-4)
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS, accelerator=ACCEL, devices=DEVICES, precision=PRECISION,
        callbacks=[pruning_cb, early_stop], gradient_clip_val=0.1,
        enable_checkpointing=False, enable_progress_bar=False, logger=False, deterministic=False)
    try:
        trainer.fit(tft, train_loader, val_loader)
    except optuna.TrialPruned:
        print(f'[Trial {trial.number}] PRUNED at epoch {trainer.current_epoch}')
        raise

    val_wape = compute_wape_med_val(tft, val_loader)
    print(f'[Trial {trial.number}] val_WAPE_med={val_wape:.4f} '
          f'epoch~{trainer.current_epoch} elapsed={time.time()-t_trial:.0f}s')
    del tft, trainer, train_loader, val_loader; gc.collect()
    if ACCEL == 'gpu': torch.cuda.empty_cache()
    return val_wape

# =========================================================================
# 5. Study
# =========================================================================
study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=3, interval_steps=1),
    storage=STORAGE, study_name=STUDY_NAME, load_if_exists=True)
print(f'  Existing trials: {len(study.trials)} | remaining: {max(0, N_TRIALS - len(study.trials))}')
remaining = max(0, N_TRIALS - len(study.trials))
if remaining > 0:
    study.optimize(objective, n_trials=remaining, gc_after_trial=True)

# =========================================================================
# 6. Save
# =========================================================================
best = study.best_trial
print(f'\n[{time.time()-T_START:.0f}s] Best trial #{best.number} | val_WAPE_med={best.value:.4f}')
print(f'  Best params: {best.params}')
with open(os.path.join(RESULTS_DIR, 'hpo_tft_gpu_best.json'), 'w') as f:
    json.dump({'best_trial': best.number, 'best_value': best.value,
               'best_params': best.params, 'n_trials': len(study.trials),
               'hidden_cap': HIDDEN_CAP, 'max_epochs': MAX_EPOCHS}, f, indent=2)
study.trials_dataframe().to_parquet(os.path.join(RESULTS_DIR, 'hpo_tft_gpu_trials.parquet'), index=False)
print(f'[{time.time()-T_START:.0f}s] DONE in {(time.time()-T_START)/60:.1f} min')
print('=' * 72)
