"""
33_hpo_mlp.py — HPO MLP_M5 (Optuna TPE + MedianPruner)
========================================================
HPO su tutte le 50K serie, S_obs RAW (no imputation), MAE loss (L1Loss).
Train: gg 1-83, val: gg 84-90 in-stock filter, metrica WAPE_med per-serie (min_hours=34).
30 trial. SQLite storage.
"""
import sys, os, gc, time, json, functools, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
print = functools.partial(print, flush=True)

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import optuna

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

H_START, H_END = 6, 23; N_HOURS = H_END - H_START
CONT_FEATURES = ['discount','avg_temperature','avg_humidity','precpt','avg_wind_level',
                 'holiday_flag','activity_flag']
LAG_NAMES = ['lag_1d','lag_7d','lag_14d','rmean_7d','rmean_14d','rstd_7d',
             'lag_dow','rmean_dow','daily_total_lag1','daily_total_rmean7','momentum_1d_7d']
N_LAGS_PER_FEAT = N_HOURS
N_LAG_DIM_M5 = len(LAG_NAMES) * N_LAGS_PER_FEAT + len(LAG_NAMES)  # 11*17+11=198

MIN_HOURS_VAL = 34
N_TRIALS = 30
MAX_EPOCHS = 100; PATIENCE = 10
STUDY_NAME = 'hpo_mlp'
STORAGE = f'sqlite:///{RESULTS_DIR}/hpo_mlp.db'

# --- Smoke test mode (env: HPO_SMOKE=1) ---
if os.getenv('HPO_SMOKE') == '1':
    N_TRIALS = 2
    MAX_EPOCHS = 5
    PATIENCE = 3
    STUDY_NAME = STUDY_NAME + '_smoke'
    STORAGE = STORAGE.replace('.db', '_smoke.db')
    print('*** SMOKE TEST MODE: N_TRIALS=2, MAX_EPOCHS=5 ***')

EMB_DIMS_DEFAULT = {'store_id':32,'product_id':32,'city_id':8,'dow':4}
CARDINALITIES = {'store_id':898,'product_id':865,'city_id':18,'dow':7}

T_START = time.time()
print('=' * 72)
print('  HPO MLP_M5 — 30 trial, full 50K, MAE loss')
print('=' * 72)

# =========================================================================
# 1. Build dataset (cache) — riusa logica analoga a 32_hpo_lgb.py
# =========================================================================
# Per evitare codice duplicato, riusiamo le matrici LGB se esistono (stessa logica train/val)
CACHE_TRAIN_X = os.path.join(RESULTS_DIR, 'hpo_lgb_train_X.parquet')
CACHE_TRAIN_Y = os.path.join(RESULTS_DIR, 'hpo_lgb_train_y.npy')
CACHE_VAL_X = os.path.join(RESULTS_DIR, 'hpo_lgb_val_X.parquet')
CACHE_VAL_Y = os.path.join(RESULTS_DIR, 'hpo_lgb_val_y.npy')
CACHE_VAL_STOCK = os.path.join(RESULTS_DIR, 'hpo_lgb_val_stock.npy')
CACHE_VAL_SIDS = os.path.join(RESULTS_DIR, 'hpo_lgb_val_sids.npy')
CACHE_VAL_PIDS = os.path.join(RESULTS_DIR, 'hpo_lgb_val_pids.npy')

if not all(os.path.exists(p) for p in [CACHE_TRAIN_X, CACHE_TRAIN_Y, CACHE_VAL_X, CACHE_VAL_Y, CACHE_VAL_STOCK]):
    print('ERROR: Run 32_hpo_lgb.py first to build the cache.')
    sys.exit(1)

print(f'[{time.time()-T_START:.0f}s] Loading cached dataset (from 32_hpo_lgb cache)...')
X_tr = pd.read_parquet(CACHE_TRAIN_X)
y_tr = np.load(CACHE_TRAIN_Y).astype(np.float32)
X_va = pd.read_parquet(CACHE_VAL_X)
y_va = np.load(CACHE_VAL_Y).astype(np.float32)
stk_va = np.load(CACHE_VAL_STOCK)
sids_va = np.load(CACHE_VAL_SIDS)
pids_va = np.load(CACHE_VAL_PIDS)

# === SUBSAMPLE TRAIN (per evitare OOM/thrashing) ===
# Train ha ~70M righe (per-hour). 5M righe sufficienti per HPO.
MAX_TRAIN_ROWS = 5_000_000
if len(X_tr) > MAX_TRAIN_ROWS:
    print(f'[{time.time()-T_START:.0f}s] Subsampling train: {len(X_tr):,} -> {MAX_TRAIN_ROWS:,} rows')
    rng = np.random.RandomState(SEED)
    sub_idx = rng.choice(len(X_tr), MAX_TRAIN_ROWS, replace=False)
    sub_idx.sort()  # mantieni ordine per cache locality
    X_tr = X_tr.iloc[sub_idx].reset_index(drop=True)
    y_tr = y_tr[sub_idx]
    gc.collect()

# Calcola normalizzazione lag features su training
print(f'[{time.time()-T_START:.0f}s] Normalizing lag features...')
lag_mean = X_tr[LAG_NAMES].fillna(0).mean().values.astype(np.float32)
lag_std = X_tr[LAG_NAMES].fillna(0).std().values.astype(np.float32) + 1e-6

# Normalizzazione continui
cont_mean = X_tr[CONT_FEATURES].mean().values.astype(np.float32)
cont_std = X_tr[CONT_FEATURES].std().values.astype(np.float32) + 1e-6

def prepare_tensors(X, y, cont_mean, cont_std, lag_mean, lag_std):
    """Converte X (DataFrame) in tensori per MLP. Returns: cat_t (int16), cont_t (f32), lag_t (f32), y_t (f32).
    cat_t è in int16 per risparmiare RAM (max cardinality=898). Cast a long fatto al momento del transfer GPU.
    """
    # cat_t: int16 risparmia 4× rispetto a long (int64). Max cardinality 898 << 32K.
    cat_vals = X[['store_id','product_id','city_id','dow']].values.astype(np.int16)
    cat_t = torch.from_numpy(cat_vals)  # dtype=int16
    cont = (X[CONT_FEATURES].values.astype(np.float32) - cont_mean) / cont_std
    cont_t = torch.tensor(cont, dtype=torch.float32)
    # Lag: 11 cols (norm) + 11 mask = 22 dim per row (per-hour)
    lag_vals = X[LAG_NAMES].fillna(0).values.astype(np.float32)
    lag_mask = (~X[LAG_NAMES].isna().values).astype(np.float32)
    lag_vals_norm = (lag_vals - lag_mean) / lag_std
    lag_t = torch.tensor(np.concatenate([lag_vals_norm, lag_mask], axis=1), dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    return cat_t, cont_t, lag_t, y_t

print(f'[{time.time()-T_START:.0f}s] Preparing tensors...')
t0 = time.time()
cat_tr, cont_tr, lag_tr, y_tr_t = prepare_tensors(X_tr, y_tr, cont_mean, cont_std, lag_mean, lag_std)
cat_va, cont_va, lag_va, y_va_t = prepare_tensors(X_va, y_va, cont_mean, cont_std, lag_mean, lag_std)
LAG_DIM = lag_tr.shape[1]
print(f'[{time.time()-T_START:.0f}s]   train={len(y_tr_t):,}, val={len(y_va_t):,}, lag_dim={LAG_DIM} (took {time.time()-t0:.0f}s)')
del X_tr, X_va; gc.collect()

# =========================================================================
# 2. MLP model definition
# =========================================================================
class MLPModel(nn.Module):
    def __init__(self, hidden_layers, emb_dims, cardinalities, cont_dim, lag_dim, dropout=0.0):
        super().__init__()
        self.embs = nn.ModuleDict({
            k: nn.Embedding(cardinalities[k], emb_dims[k]) for k in ['store_id','product_id','city_id','dow']
        })
        emb_total = sum(emb_dims.values())
        in_dim = emb_total + cont_dim + lag_dim
        layers = []
        prev = in_dim
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Softplus()]
        self.net = nn.Sequential(*layers)

    def forward(self, cat, cont, lag):
        emb = torch.cat([self.embs['store_id'](cat[:,0]), self.embs['product_id'](cat[:,1]),
                         self.embs['city_id'](cat[:,2]), self.embs['dow'](cat[:,3])], dim=1)
        x = torch.cat([emb, cont, lag], dim=1)
        return self.net(x).squeeze(-1)

# =========================================================================
# 3. WAPE_med per-serie (val)
# =========================================================================
def wape_med_per_series(preds):
    instock = stk_va == 0
    wapes = []
    series_key = sids_va * 100000 + pids_va
    for sk in np.unique(series_key):
        mask = (series_key == sk) & instock
        if mask.sum() < MIN_HOURS_VAL:
            continue
        yt, yp = y_va[mask], preds[mask]
        denom = max(np.abs(yt).sum(), 1e-8)
        wapes.append(np.abs(yt - yp).sum() / denom)
    return float(np.median(wapes)) if wapes else float('nan')

# =========================================================================
# 4. Optuna objective
# =========================================================================
def objective(trial):
    t_trial = time.time()
    hidden_str = trial.suggest_categorical('hidden', [
        '[64,32]', '[128,64]', '[256,128]', '[128,128,64]', '[256,128,64]',
    ])
    hidden_layers = json.loads(hidden_str)
    dropout = trial.suggest_float('dropout', 0.0, 0.3, step=0.05)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [1024, 4096, 16384])
    emb_scale = trial.suggest_float('emb_scale', 0.5, 2.0, step=0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    emb_dims = {k: max(2, int(v * emb_scale)) for k, v in EMB_DIMS_DEFAULT.items()}
    print(f'\n[Trial {trial.number}] hidden={hidden_layers}, dropout={dropout:.2f}, '
          f'lr={lr:.1e}, bs={batch_size}, emb_scale={emb_scale}, wd={weight_decay:.1e}, '
          f'emb_dims={emb_dims}')

    model = MLPModel(hidden_layers, emb_dims, CARDINALITIES, len(CONT_FEATURES), LAG_DIM, dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.L1Loss()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Model: {n_params:,} params')

    # Train
    n_train = len(y_tr_t)
    best_val = float('inf'); patience_cnt = 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n_train)
        tl, nb = 0., 0
        for s in range(0, n_train, batch_size):
            idx = perm[s:s+batch_size]
            c = cat_tr[idx].long().to(DEVICE); co = cont_tr[idx].to(DEVICE)  # int16 → long per embedding
            l = lag_tr[idx].to(DEVICE); t = y_tr_t[idx].to(DEVICE)
            optimizer.zero_grad()
            p = model(c, co, l)
            loss = loss_fn(p, t)
            loss.backward(); optimizer.step()
            tl += loss.item(); nb += 1
        # Val
        model.eval()
        with torch.no_grad():
            val_preds = []
            for s in range(0, len(y_va_t), batch_size):
                c = cat_va[s:s+batch_size].long().to(DEVICE); co = cont_va[s:s+batch_size].to(DEVICE)  # int16 → long
                l = lag_va[s:s+batch_size].to(DEVICE)
                p = model(c, co, l)
                val_preds.append(p.cpu().numpy())
            val_preds = np.concatenate(val_preds)
        val_wape = wape_med_per_series(val_preds)
        if epoch % 5 == 0 or epoch < 3:
            print(f'  Epoch {epoch:3d}: train_loss={tl/nb:.4f}, val_WAPE_med={val_wape:.4f}')

        # Pruning report (ogni 5 epoche)
        if epoch >= 2 and epoch % 5 == 0:
            trial.report(val_wape, step=epoch)
            if trial.should_prune():
                print(f'  Pruned at epoch {epoch}')
                raise optuna.TrialPruned()

        # Early stopping
        if val_wape < best_val - 1e-4:
            best_val = val_wape; patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f'  Early stop at epoch {epoch}, best val_WAPE_med={best_val:.4f}')
                break

    elapsed = time.time() - t_trial
    print(f'[Trial {trial.number}] best_val_WAPE_med={best_val:.4f}, elapsed={elapsed:.0f}s')
    return best_val

# =========================================================================
# 5. Run Optuna
# =========================================================================
print(f'\n[{time.time()-T_START:.0f}s] Creating Optuna study (device={DEVICE})...')
study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=5),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    storage=STORAGE, study_name=STUDY_NAME, load_if_exists=True,
)
remaining = max(0, N_TRIALS - len(study.trials))
print(f'  Existing trials: {len(study.trials)}, remaining: {remaining}')
if remaining > 0:
    study.optimize(objective, n_trials=remaining, gc_after_trial=True)

# =========================================================================
# 6. Save results
# =========================================================================
best = study.best_trial
print(f'\nBest trial: #{best.number}, val_WAPE_med={best.value:.4f}, params={best.params}')
with open(os.path.join(RESULTS_DIR, 'hpo_mlp_best.json'), 'w') as f:
    json.dump({'best_trial': best.number, 'best_value': best.value,
               'best_params': best.params, 'n_trials': len(study.trials)}, f, indent=2)
study.trials_dataframe().to_parquet(os.path.join(RESULTS_DIR, 'hpo_mlp_trials.parquet'), index=False)
import pickle
with open(os.path.join(RESULTS_DIR, 'hpo_mlp_study.pkl'), 'wb') as f:
    pickle.dump(study, f)
print(f'\n[{time.time()-T_START:.0f}s] DONE in {(time.time()-T_START)/60:.1f} min')
