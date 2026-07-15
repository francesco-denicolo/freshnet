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
# Serie di validazione usate DURANTE l'HPO. La validazione serve solo a ORDINARE le
# configurazioni, non a produrre i numeri del paper (quelli escono dalle 14 celle, su
# tutte le 50K). Lightning valida a OGNI epoca: su 50K serie x 12 epoche il costo domina
# il trial. Un campione stratificato-per-caso di 8K basta per il ranking.
MAX_VAL_SERIES = int(os.getenv('TFT_MAX_VAL_SERIES', 8_000))
# Budget VRAM: hidden*batch <= 65536 è il valore PROVATO sulla T4
# (hidden=32 x batch=2048 gira; hidden=128 x batch=2048 -> CUDA OOM).
VRAM_BUDGET = int(os.getenv('TFT_VRAM_BUDGET', 65_536))
# Subsample delle SERIE (0 = tutte). Su Colab free (~12.7 GB RAM) il TimeSeriesDataSet
# su 50K serie va OOM: usa es. SERIES_SUBSAMPLE=15000. Cache dedicata per non collidere.
SERIES_SUBSAMPLE = int(os.getenv('SERIES_SUBSAMPLE', 0))
SUB_TAG = f'_sub{SERIES_SUBSAMPLE}' if SERIES_SUBSAMPLE > 0 else ''

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
# Cache pesanti (long_data ~3GB, TimeSeriesDataSet): di default in RESULTS_DIR, ma su
# SageMaker vanno su disco locale (TFT_CACHE_DIR=/tmp/...) per non finire nei checkpoint
# sincronizzati su S3. Lo studio Optuna resta invece in RESULTS_DIR (serve al resume).
CACHE_DIR = os.getenv('TFT_CACHE_DIR', RESULTS_DIR)
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_PATH = os.path.join(CACHE_DIR, f'hpo_tft_long_data_cache{SUB_TAG}.parquet')
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
    if SERIES_SUBSAMPLE > 0:
        keys = df_full[['store_id','product_id']].drop_duplicates()
        if len(keys) > SERIES_SUBSAMPLE:
            sel = keys.sample(n=SERIES_SUBSAMPLE, random_state=SEED)
            df_full = (df_full.merge(sel, on=['store_id','product_id'], how='inner')
                       .sort_values(['store_id','product_id','dt_parsed']).reset_index(drop=True))
            print(f'[{time.time()-T_START:.0f}s]   Subsampled to {SERIES_SUBSAMPLE} series -> {len(df_full):,} rows')
    sales_arr = np.array(df_full['hours_sale'].tolist(), dtype=np.float32)[:, H_START:H_END]
    stock_arr = np.array(df_full['hours_stock_status'].tolist(), dtype=np.int8)[:, H_START:H_END]
    n_rows = len(df_full)
    # Memory-efficient long build: categoricals via from_codes (evita di materializzare
    # ~82M stringhe Python in un colpo, che è ciò che manda in OOM Colab a 12.7 GB).
    NH = N_HOURS
    day_rep = np.repeat(df_full['day_num'].values.astype(np.int32), NH)
    hour_code = np.tile(np.arange(NH, dtype=np.int16), n_rows)
    cols = {}
    for c in ['store_id','product_id','city_id','dow','holiday_flag','activity_flag']:
        cc = df_full[c].astype(str).astype('category')          # piccolo: n_rows righe
        cols[c] = pd.Categorical.from_codes(np.repeat(cc.cat.codes.values, NH), cc.cat.categories)
        del cc
    cols['hour'] = pd.Categorical.from_codes(hour_code, [str(h) for h in range(H_START, H_END)])
    for c in ['discount','avg_temperature','avg_humidity','precpt','avg_wind_level']:
        cols[c] = np.repeat(df_full[c].values.astype(np.float32), NH)
    cols['sales'] = sales_arr.reshape(-1).astype(np.float32)
    cols['stock'] = stock_arr.reshape(-1).astype(np.int8)
    cols['day_num'] = day_rep.astype(np.int16)
    cols['time_idx'] = (day_rep * NH + hour_code.astype(np.int32)).astype(np.int32)
    long_data = pd.DataFrame(cols)
    del cols, day_rep, hour_code, sales_arr, stock_arr; gc.collect()
    long_data.to_parquet(CACHE_PATH, index=False)
    del df_train, df_eval, df_full; gc.collect()
print(f'[{time.time()-T_START:.0f}s] Long_data shape: {long_data.shape}')

# =========================================================================
# 2. TimeSeriesDataSet (cache condivisa)
# =========================================================================
TSD_TRAIN_CACHE = os.path.join(CACHE_DIR, f'hpo_tft_tsd_train{SUB_TAG}.pkl')
TSD_VAL_CACHE = os.path.join(CACHE_DIR, f'hpo_tft_tsd_val{SUB_TAG}.pkl')
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

# Sottocampiona le serie di validazione: Lightning valida a OGNI epoca, quindi su 50K
# serie il costo si moltiplica per il numero di epoche e domina il trial.
# Il campione è STRATIFICATO per quartile di volume: il volume è la dimensione attorno
# a cui ruota l'analisi (RQ4), quindi le quattro fasce devono pesare uguale nel criterio
# che sceglie la configurazione. Fallback su uniforme se la stratificazione non è
# applicabile: meglio un campione uniforme che un job da ore che crasha qui.
N_VAL = len(validation)
if MAX_VAL_SERIES and N_VAL > MAX_VAL_SERIES:
    val_idx = None
    try:
        dec = validation.decoded_index.reset_index(drop=True)
        dec['pos'] = np.arange(len(dec))
        dec['store_id'] = dec['store_id'].astype(int)
        dec['product_id'] = dec['product_id'].astype(int)
        strat = pd.read_parquet(os.path.join(RESULTS_DIR, 'stratification.parquet'))[
            ['store_id', 'product_id', 'vol_bin']]
        m = dec.merge(strat, on=['store_id', 'product_id'], how='inner')
        per_q = MAX_VAL_SERIES // m['vol_bin'].nunique()
        val_idx = (m.groupby('vol_bin', group_keys=False)
                     .apply(lambda g: g.sample(min(len(g), per_q), random_state=SEED))['pos']
                     .to_numpy())
        counts = m[m['pos'].isin(val_idx)]['vol_bin'].value_counts().to_dict()
        print(f'[{time.time()-T_START:.0f}s]   Validation stratificata per volume: '
              f'{N_VAL:,} -> {len(val_idx):,} serie {counts}')
    except Exception as e:
        print(f'[{time.time()-T_START:.0f}s]   Stratificazione non riuscita ({e}); uso campione uniforme')
        val_idx = None
    if val_idx is None:
        val_idx = np.random.RandomState(SEED).choice(N_VAL, MAX_VAL_SERIES, replace=False)
        print(f'[{time.time()-T_START:.0f}s]   Validation subsampled (uniforme): '
              f'{N_VAL:,} -> {MAX_VAL_SERIES:,} serie')
    validation_sub = torch.utils.data.Subset(validation, [int(i) for i in val_idx])
else:
    validation_sub = validation

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
    # Vettorizzato: il vecchio loop con idx_df.iloc[i] su decine di migliaia di righe
    # costava minuti da solo. Qui si itera su array numpy per costruire la maschera,
    # poi WAPE si calcola in blocco.
    sids = idx_df['store_id'].to_numpy().astype(np.int64)
    pids = idx_df['product_id'].to_numpy().astype(np.int64)
    mask = np.zeros(preds.shape, dtype=bool)
    for i in range(len(sids)):
        stk = val_stock_by_serie.get((int(sids[i]), int(pids[i])))
        if stk is not None and len(stk) >= PRED_LENGTH:
            mask[i] = (stk[:PRED_LENGTH] == 0)
    valid = mask.sum(axis=1) >= MIN_HOURS_VAL
    if not valid.any():
        return float('nan')
    num = np.abs((preds - truths) * mask).sum(axis=1)
    den = np.maximum(np.abs(truths * mask).sum(axis=1), 1e-8)
    return float(np.median((num / den)[valid]))

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
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    # unico vincolo: rispetta il budget 'ampia' (hidden <= HIDDEN_CAP). NON per OOM, per costo.
    if hidden_size > HIDDEN_CAP:
        print(f'\n[Trial {trial.number}] SKIP hidden={hidden_size} > cap {HIDDEN_CAP}')
        raise optuna.TrialPruned()

    # Batch DERIVATO, non cercato: su una GPU fissa si usa il più grande che entra.
    # Cercarlo insieme a hidden produce combinazioni che vanno in CUDA OOM (è ciò che
    # ha ucciso il run precedente: hidden=128 x batch=2048 su una T4 da 15 GB).
    batch_size = 128
    for b in (2048, 1024, 512, 256, 128):
        if hidden_size * b <= VRAM_BUDGET:
            batch_size = b
            break

    print(f'\n[Trial {trial.number}] head_dim={head_dim} heads={n_heads} hidden={hidden_size} '
          f'dropout={dropout:.2f} lr={lr:.1e} batch={batch_size} (derivato) wd={weight_decay:.1e}')

    train_loader = DataLoader(training_sub, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=training._collate_fn)
    val_loader = DataLoader(validation_sub, batch_size=batch_size * 2, shuffle=False,
                            num_workers=0, collate_fn=validation._collate_fn)

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
    # Strumentazione: separa fit e valutazione, così si vede DOVE va il tempo invece
    # di stimarlo a occhio.
    t_fit0 = time.time()
    try:
        trainer.fit(tft, train_loader, val_loader)
    except optuna.TrialPruned:
        print(f'[Trial {trial.number}] PRUNED at epoch {trainer.current_epoch} '
              f'(fit={time.time()-t_fit0:.0f}s)')
        raise
    t_fit = time.time() - t_fit0

    t_val0 = time.time()
    val_wape = compute_wape_med_val(tft, val_loader)
    t_val = time.time() - t_val0
    print(f'[Trial {trial.number}] val_WAPE_med={val_wape:.4f} epoch~{trainer.current_epoch} '
          f'fit={t_fit:.0f}s val={t_val:.0f}s elapsed={time.time()-t_trial:.0f}s')
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
