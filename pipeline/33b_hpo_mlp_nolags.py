"""
33b_hpo_mlp_nolags.py — HPO MLP_no_lags (Optuna TPE + MedianPruner)
====================================================================
Architettura PER-DAY matches production (03_fase_a_mlp.py).
Variant NO LAGS: only embeddings + continuous features (lag_dim=0).
Train: gg 1-83, val: gg 84-90, S_obs RAW (no imputation), MAE loss (L1Loss).
Output 17-dim. Metrica: WAPE_med per-serie sulle ore in-stock val (min_hours=34).
30 trial. SQLite storage.
"""
import sys, os, gc, time, json, functools, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
print = functools.partial(print, flush=True)

import torch, torch.nn as nn
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
N_LAG_DIM_M5 = len(LAG_NAMES) * N_LAGS_PER_FEAT + len(LAG_NAMES)  # 11×17+11=198

EMB_DIMS_DEFAULT = {'store_id':32,'product_id':32,'city_id':8,'dow':4}
CARDINALITIES = {'store_id':898,'product_id':865,'city_id':18,'dow':7}

MIN_HOURS_VAL = 34
N_TRIALS = 45
MAX_EPOCHS = 100; PATIENCE = 10
STUDY_NAME = 'hpo_mlp_nolags'
STORAGE = f'sqlite:///{RESULTS_DIR}/hpo_mlp_nolags.db'
USE_LAGS = False  # NO LAGS variant
LAG_DIM = 0       # no lag features

# --- Smoke test mode (env: HPO_SMOKE=1) ---
if os.getenv('HPO_SMOKE') == '1':
    N_TRIALS = 2
    MAX_EPOCHS = 5
    PATIENCE = 3
    STUDY_NAME = STUDY_NAME + '_smoke'
    STORAGE = STORAGE.replace('.db', '_smoke.db')
    print('*** SMOKE TEST MODE: N_TRIALS=2, MAX_EPOCHS=5 ***')

T_START = time.time()
print('=' * 72)
print('  HPO MLP_no_lags (per-day) — 30 trial, full 50K, MAE loss')
print('=' * 72)

# =========================================================================
# 1. Build dataset PER-DAY (cache)
# =========================================================================
CACHE_TR_CAT = os.path.join(RESULTS_DIR, 'hpo_mlp_nolags_train_cat.npy')
CACHE_TR_CONT = os.path.join(RESULTS_DIR, 'hpo_mlp_nolags_train_cont.npy')
CACHE_TR_Y = os.path.join(RESULTS_DIR, 'hpo_mlp_nolags_train_y.npy')
CACHE_VA_CAT = os.path.join(RESULTS_DIR, 'hpo_mlp_nolags_val_cat.npy')
CACHE_VA_CONT = os.path.join(RESULTS_DIR, 'hpo_mlp_nolags_val_cont.npy')
CACHE_VA_Y = os.path.join(RESULTS_DIR, 'hpo_mlp_nolags_val_y.npy')
CACHE_VA_STK = os.path.join(RESULTS_DIR, 'hpo_mlp_nolags_val_stk.npy')
CACHE_VA_SIDS = os.path.join(RESULTS_DIR, 'hpo_mlp_nolags_val_sids.npy')
CACHE_VA_PIDS = os.path.join(RESULTS_DIR, 'hpo_mlp_nolags_val_pids.npy')
CACHE_NORM = os.path.join(RESULTS_DIR, 'hpo_mlp_nolags_norm.npz')

ALL_CACHES = [CACHE_TR_CAT, CACHE_TR_CONT, CACHE_TR_Y,
              CACHE_VA_CAT, CACHE_VA_CONT, CACHE_VA_Y,
              CACHE_VA_STK, CACHE_VA_SIDS, CACHE_VA_PIDS, CACHE_NORM]

def compute_lags(avail_sales, avail_dows, dow, K):
    """11 lag features per-day (17-dim each). Matches 03_fase_a_mlp.py."""
    z = np.float32; NH = N_HOURS
    L = {n: np.full(NH, np.nan, dtype=z) for n in LAG_NAMES}
    if K == 0: return L
    L['lag_1d'] = avail_sales[-1]
    if K >= 7: L['lag_7d'] = avail_sales[-7]
    if K >= 14: L['lag_14d'] = avail_sales[-14]
    if K >= 7: L['rmean_7d'] = avail_sales[-7:].mean(0)
    if K >= 14: L['rmean_14d'] = avail_sales[-14:].mean(0)
    if K >= 2: L['rstd_7d'] = avail_sales[-min(7,K):].std(0)
    sd = avail_dows == dow
    if sd.any():
        ds = avail_sales[sd]
        L['lag_dow'] = ds[-1]
        L['rmean_dow'] = ds.mean(0)
    dt = avail_sales.sum(1)
    L['daily_total_lag1'] = np.full(NH, dt[-1], dtype=z)
    if K >= 7: L['daily_total_rmean7'] = np.full(NH, dt[-7:].mean(), dtype=z)
    r, l = L['rmean_7d'], L['lag_1d']
    if not np.isnan(r).all():
        v = (~np.isnan(l)) & (~np.isnan(r)) & (r > 0)
        if v.any():
            m = np.full(NH, np.nan, dtype=z); m[v] = l[v] / r[v]
            L['momentum_1d_7d'] = m
    return L

def build_dataset(split, series_cache, cont_mean=None, cont_std=None):
    """Build per-day dataset (matches 03_fase_a_mlp.py) — NO LAGS variant."""
    if split == 'train': d_min, d_max = 2, 83
    elif split == 'val': d_min, d_max = 84, 90
    else: raise ValueError(split)
    cat_l, cont_l, tgt_l, stk_l, sid_l, pid_l = [], [], [], [], [], []
    nd = 0
    for (sid, pid), sd in series_cache.items():
        nd += 1
        if nd % 10000 == 0: print(f'      ... {nd:,}/{len(series_cache):,}')
        days, dows, sales, stock = sd['days'], sd['dows'], sd['sales'], sd['stock']
        city, conts = sd['city_id'], sd['conts']
        for idx in range(len(days)):
            d = days[idx]
            if d < d_min or d > d_max: continue
            cat_l.append([sid, pid, city, dows[idx]])
            cont_l.append(conts[idx])
            tgt_l.append(sales[idx]); stk_l.append(stock[idx])
            sid_l.append(sid); pid_l.append(pid)
    cat_arr = np.array(cat_l, dtype=np.int16)
    cont_arr = np.array(cont_l, dtype=np.float32)
    tgt_arr = np.array(tgt_l, dtype=np.float32)
    stk_arr = np.array(stk_l, dtype=np.int8)
    if cont_mean is None:
        cont_mean = cont_arr.mean(0); cont_std = cont_arr.std(0)
        cont_std[cont_std < 1e-8] = 1.0
    cont_arr = (cont_arr - cont_mean) / cont_std
    return (cat_arr, cont_arr.astype(np.float32),
            tgt_arr, stk_arr, np.array(sid_l, dtype=np.int32), np.array(pid_l, dtype=np.int32),
            cont_mean, cont_std)

if not all(os.path.exists(p) for p in ALL_CACHES):
    print(f'[{time.time()-T_START:.0f}s] Building per-day dataset (cache miss)...')
    df_train = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
    df_eval = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_eval.parquet'))
    df_full = pd.concat([df_train, df_eval], ignore_index=True)
    df_full['dt_parsed'] = pd.to_datetime(df_full['dt'])
    df_full = df_full.sort_values(['store_id','product_id','dt_parsed']).reset_index(drop=True)
    all_dates = sorted(df_full['dt_parsed'].unique())
    date_to_day = {d: i+1 for i, d in enumerate(all_dates)}
    df_full['day_num'] = df_full['dt_parsed'].map(date_to_day)
    df_full['dow'] = df_full['dt_parsed'].dt.dayofweek
    sales_all = np.array(df_full['hours_sale'].tolist(), dtype=np.float32)[:, H_START:H_END]
    stock_all = np.array(df_full['hours_stock_status'].tolist(), dtype=np.float32)[:, H_START:H_END]

    print(f'[{time.time()-T_START:.0f}s]   Building series cache...')
    series_cache = {}
    for (sid, pid), grp in df_full.groupby(['store_id','product_id'], sort=False):
        gs = grp.sort_values('day_num')
        idx = gs.index.values
        series_cache[(sid, pid)] = {
            'days': gs['day_num'].values, 'dows': gs['dow'].values,
            'sales': sales_all[idx], 'stock': stock_all[idx],
            'city_id': gs['city_id'].values[0],
            'conts': gs[CONT_FEATURES].values.astype(np.float32),
        }
    print(f'[{time.time()-T_START:.0f}s]   {len(series_cache):,} serie')
    del df_train, df_eval, df_full, sales_all, stock_all; gc.collect()

    print(f'[{time.time()-T_START:.0f}s]   Building train dataset (NO lags)...')
    cat_tr, cont_tr, tgt_tr, _, _, _, cm, cs = build_dataset('train', series_cache)
    print(f'[{time.time()-T_START:.0f}s]   Train: {cat_tr.shape[0]:,} day-samples')
    np.save(CACHE_TR_CAT, cat_tr); np.save(CACHE_TR_CONT, cont_tr)
    np.save(CACHE_TR_Y, tgt_tr)
    del cat_tr, cont_tr, tgt_tr; gc.collect()

    print(f'[{time.time()-T_START:.0f}s]   Building val dataset (NO lags)...')
    cat_va, cont_va, tgt_va, stk_va, sids_va, pids_va, _, _ = build_dataset(
        'val', series_cache, cm, cs)
    print(f'[{time.time()-T_START:.0f}s]   Val: {cat_va.shape[0]:,} day-samples')
    np.save(CACHE_VA_CAT, cat_va); np.save(CACHE_VA_CONT, cont_va)
    np.save(CACHE_VA_Y, tgt_va)
    np.save(CACHE_VA_STK, stk_va); np.save(CACHE_VA_SIDS, sids_va); np.save(CACHE_VA_PIDS, pids_va)
    np.savez(CACHE_NORM, cont_mean=cm, cont_std=cs)
    del cat_va, cont_va, tgt_va, stk_va, sids_va, pids_va, series_cache; gc.collect()
else:
    print(f'[{time.time()-T_START:.0f}s] Loading cached per-day dataset (NO lags)...')

# Load cached
cat_tr = np.load(CACHE_TR_CAT); cont_tr = np.load(CACHE_TR_CONT)
tgt_tr = np.load(CACHE_TR_Y)
cat_va = np.load(CACHE_VA_CAT); cont_va = np.load(CACHE_VA_CONT)
tgt_va = np.load(CACHE_VA_Y)
stk_va = np.load(CACHE_VA_STK); sids_va = np.load(CACHE_VA_SIDS); pids_va = np.load(CACHE_VA_PIDS)
print(f'[{time.time()-T_START:.0f}s] Dataset loaded: train={cat_tr.shape[0]:,}, val={cat_va.shape[0]:,} (NO lag features)')

# To tensors (cat as int16, cast to long on transfer)
cat_tr_t = torch.from_numpy(cat_tr)  # int16
cont_tr_t = torch.from_numpy(cont_tr)
tgt_tr_t = torch.from_numpy(tgt_tr)
cat_va_t = torch.from_numpy(cat_va)
cont_va_t = torch.from_numpy(cont_va)
tgt_va_t = torch.from_numpy(tgt_va)
# Placeholder lag tensors (zero-dim) per compatibilità con model forward
lag_tr_t = torch.zeros(len(cat_tr), 0, dtype=torch.float32)
lag_va_t = torch.zeros(len(cat_va), 0, dtype=torch.float32)

# =========================================================================
# 2. Model
# =========================================================================
class MLPPerDay(nn.Module):
    def __init__(self, hidden, emb_dims, cardinalities, cont_dim, lag_dim, dropout=0.0):
        super().__init__()
        self.embs = nn.ModuleDict({
            k: nn.Embedding(cardinalities[k], emb_dims[k]) for k in ['store_id','product_id','city_id','dow']
        })
        self.names = ['store_id','product_id','city_id','dow']
        in_dim = sum(emb_dims.values()) + cont_dim + lag_dim
        layers = []
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, N_HOURS), nn.Softplus()]  # 17-dim output
        self.mlp = nn.Sequential(*layers)

    def forward(self, cat, cont, lag):
        e = [self.embs[n](cat[:,i]) for i, n in enumerate(self.names)]
        x = torch.cat(e + [cont, lag], dim=1)
        return self.mlp(x)

# =========================================================================
# 3. WAPE_med per-serie su val (in-stock, min_hours=34)
# =========================================================================
def wape_med_per_series(preds_2d):
    """preds_2d: (N_val_days, 17). Confronto con tgt_va (N_val_days, 17), stk_va (N, 17), sids_va, pids_va."""
    # Per ogni (store, product), prendi tutte le ore in-stock dei giorni val
    series_key = sids_va * 100000 + pids_va  # un id univoco per serie
    wapes = []
    for sk in np.unique(series_key):
        mask = series_key == sk
        y_t = tgt_va[mask].reshape(-1)        # flatten across days
        y_p = preds_2d[mask].reshape(-1)
        stk = stk_va[mask].reshape(-1)
        instock = stk == 0
        if instock.sum() < MIN_HOURS_VAL:
            continue
        yt, yp = y_t[instock], y_p[instock]
        denom = max(np.abs(yt).sum(), 1e-8)
        wapes.append(np.abs(yt - yp).sum() / denom)
    return float(np.median(wapes)) if wapes else float('nan')

# =========================================================================
# 4. Optuna objective
# =========================================================================
def objective(trial):
    t_trial = time.time()
    hidden_str = trial.suggest_categorical('hidden', [
        '[64]', '[128]', '[256]',
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

    model = MLPPerDay(hidden_layers, emb_dims, CARDINALITIES,
                      len(CONT_FEATURES), LAG_DIM, dropout).to(DEVICE)  # LAG_DIM=0 for nolags
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.L1Loss()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Model: {n_params:,} params, input_dim={sum(emb_dims.values())+len(CONT_FEATURES)+LAG_DIM}')

    n_train = len(tgt_tr_t)
    best_val = float('inf'); patience_cnt = 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n_train)
        tl, nb = 0., 0
        for s in range(0, n_train, batch_size):
            idx = perm[s:s+batch_size]
            c = cat_tr_t[idx].long().to(DEVICE)
            co = cont_tr_t[idx].to(DEVICE)
            l = lag_tr_t[idx].to(DEVICE)
            t = tgt_tr_t[idx].to(DEVICE)
            optimizer.zero_grad()
            p = model(c, co, l)
            loss = loss_fn(p, t)
            loss.backward(); optimizer.step()
            tl += loss.item(); nb += 1
        # Val
        model.eval()
        with torch.no_grad():
            preds_list = []
            for s in range(0, len(tgt_va_t), batch_size):
                c = cat_va_t[s:s+batch_size].long().to(DEVICE)
                co = cont_va_t[s:s+batch_size].to(DEVICE)
                l = lag_va_t[s:s+batch_size].to(DEVICE)
                p = model(c, co, l)
                preds_list.append(p.cpu().numpy())
            preds = np.concatenate(preds_list, axis=0)
        val_wape = wape_med_per_series(preds)
        if epoch % 5 == 0 or epoch < 3:
            print(f'  Epoch {epoch:3d}: train_loss={tl/nb:.4f}, val_WAPE_med={val_wape:.4f}')

        # Pruning (ogni 5 epoche dopo warmup)
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
with open(os.path.join(RESULTS_DIR, 'hpo_mlp_nolags_best.json'), 'w') as f:
    json.dump({'best_trial': best.number, 'best_value': best.value,
               'best_params': best.params, 'n_trials': len(study.trials)}, f, indent=2)
study.trials_dataframe().to_parquet(os.path.join(RESULTS_DIR, 'hpo_mlp_nolags_trials.parquet'), index=False)
import pickle
with open(os.path.join(RESULTS_DIR, 'hpo_mlp_nolags_study.pkl'), 'wb') as f:
    pickle.dump(study, f)
print(f'\n[{time.time()-T_START:.0f}s] DONE in {(time.time()-T_START)/60:.1f} min')
