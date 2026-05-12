"""
32_hpo_lgb.py — HPO LGB_M5 (Optuna TPE + MedianPruner)
=======================================================
HPO su tutte le 50K serie, S_obs RAW (no imputation), MAE loss (regression_l1).
Train: gg 1-83, val: gg 84-90 in-stock filter, metrica WAPE_med per-serie (min_hours=34).
30 trial. MedianPruner via callback. SQLite storage.
"""
import sys, os, gc, time, json, functools, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
print = functools.partial(print, flush=True)

import lightgbm as lgb
import optuna

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42; np.random.seed(SEED)
H_START, H_END = 6, 23; N_HOURS = H_END - H_START
HOURS_RANGE = np.arange(H_START, H_END, dtype=np.int32)
MIN_HOURS_VAL = 34
N_TRIALS = 30
NUM_BOOST_ROUND = 1000
STUDY_NAME = 'hpo_lgb'
STORAGE = f'sqlite:///{RESULTS_DIR}/hpo_lgb.db'

# --- Smoke test mode (env: HPO_SMOKE=1) ---
if os.getenv('HPO_SMOKE') == '1':
    N_TRIALS = 2
    NUM_BOOST_ROUND = 100
    STUDY_NAME = STUDY_NAME + '_smoke'
    STORAGE = STORAGE.replace('.db', '_smoke.db')
    print('*** SMOKE TEST MODE: N_TRIALS=2, NUM_BOOST_ROUND=100 ***')

CONT_FEATURES = ['discount','avg_temperature','avg_humidity','precpt','avg_wind_level',
                 'holiday_flag','activity_flag']
CAT_FEATURES = ['store_id','product_id','city_id','dow','hour']
LAG_NAMES = ['lag_1d','lag_7d','lag_14d','rmean_7d','rmean_14d','rstd_7d',
             'lag_dow','rmean_dow','daily_total_lag1','daily_total_rmean7','momentum_1d_7d']

T_START = time.time()
print('=' * 72)
print('  HPO LGB_M5 — 30 trial, full 50K, MAE loss')
print('=' * 72)

# =========================================================================
# 1. Load & build dataset (UNA SOLA VOLTA, cache)
# =========================================================================
CACHE_TRAIN_X = os.path.join(RESULTS_DIR, 'hpo_lgb_train_X.parquet')
CACHE_TRAIN_Y = os.path.join(RESULTS_DIR, 'hpo_lgb_train_y.npy')
CACHE_VAL_X = os.path.join(RESULTS_DIR, 'hpo_lgb_val_X.parquet')
CACHE_VAL_Y = os.path.join(RESULTS_DIR, 'hpo_lgb_val_y.npy')
CACHE_VAL_STOCK = os.path.join(RESULTS_DIR, 'hpo_lgb_val_stock.npy')
CACHE_VAL_SIDS = os.path.join(RESULTS_DIR, 'hpo_lgb_val_sids.npy')
CACHE_VAL_PIDS = os.path.join(RESULTS_DIR, 'hpo_lgb_val_pids.npy')

def compute_lags(sales_arr, dows_arr, dv, K):
    """11 lag features M5-style. sales_arr e dows_arr fino al giorno corrente."""
    lags = {n: np.full(N_HOURS, np.nan, dtype=np.float32) for n in LAG_NAMES}
    if K < 1:
        return lags
    lags['lag_1d'] = sales_arr[-1].astype(np.float32) if K >= 1 else np.full(N_HOURS, np.nan)
    if K >= 7:
        lags['lag_7d'] = sales_arr[-7].astype(np.float32)
    if K >= 14:
        lags['lag_14d'] = sales_arr[-14].astype(np.float32)
        lags['rmean_14d'] = sales_arr[-14:].mean(0).astype(np.float32)
        lags['rstd_7d'] = sales_arr[-7:].std(0).astype(np.float32)
    if K >= 7:
        lags['rmean_7d'] = sales_arr[-7:].mean(0).astype(np.float32)
    # DoW-specific (ultimi 4 valori dello stesso DoW)
    dow_mask = dows_arr == dv
    if dow_mask.any():
        lags['lag_dow'] = sales_arr[dow_mask][-1].astype(np.float32)
        last_dow = sales_arr[dow_mask][-min(4, dow_mask.sum()):]
        lags['rmean_dow'] = last_dow.mean(0).astype(np.float32)
    # Daily aggregates (somma giornaliera)
    if K >= 1:
        daily_total = sales_arr.sum(axis=1)
        lags['daily_total_lag1'] = np.full(N_HOURS, daily_total[-1])
    if K >= 7:
        lags['daily_total_rmean7'] = np.full(N_HOURS, daily_total[-7:].mean())
    if K >= 7:
        rmean1 = sales_arr[-1].mean()
        rmean7 = sales_arr[-7:].mean(axis=(0,1))
        lags['momentum_1d_7d'] = np.full(N_HOURS, rmean1 - rmean7)
    return lags

def build_dataset_for_split(split, df_full, sales_arr, stock_arr, series_cache):
    """Costruisce X, y, stock, sids, pids per train (gg 1-83) o val (gg 84-90)."""
    if split == 'train':
        mask = (df_full['day_num'] >= 1) & (df_full['day_num'] <= 83)
    else:  # val
        mask = (df_full['day_num'] >= 84) & (df_full['day_num'] <= 90)
    sel = df_full[mask].reset_index()
    nd = len(sel)
    nh = nd * N_HOURS
    sids = np.repeat(sel['store_id'].values, N_HOURS)
    pids = np.repeat(sel['product_id'].values, N_HOURS)
    cids = np.repeat(sel['city_id'].values, N_HOURS)
    dows = np.repeat(sel['dow'].values, N_HOURS)
    dnums = np.repeat(sel['day_num'].values, N_HOURS)
    hours = np.tile(HOURS_RANGE, nd)
    y = sales_arr[sel['index'].values].reshape(-1)
    stk = stock_arr[sel['index'].values].reshape(-1)

    # Continuous features
    fd = {'store_id':sids,'product_id':pids,'city_id':cids,'dow':dows,'hour':hours}
    coh = np.zeros((nh, len(CONT_FEATURES)), dtype=np.float32)
    for j, c in enumerate(CONT_FEATURES):
        coh[:, j] = np.repeat(sel[c].values, N_HOURS)
    for j, c in enumerate(CONT_FEATURES):
        fd[c] = coh[:, j]

    # Lags (UNA volta per ogni (sid, pid, day))
    la = {n: np.full(nh, np.nan, dtype=np.float32) for n in LAG_NAMES}
    print(f'    Computing lags for {nd:,} days...')
    for ri in range(nd):
        if (ri+1) % 500000 == 0: print(f'      ... {ri+1:,}/{nd:,}')
        sid, pid, d, dv = sids[ri*N_HOURS], pids[ri*N_HOURS], dnums[ri*N_HOURS], dows[ri*N_HOURS]
        sc = series_cache[(sid, pid)]
        ad = d - 1 if split == 'train' else 83
        am = sc['days'] <= ad
        K = int(am.sum())
        hs = ri * N_HOURS
        if K > 0:
            lg = compute_lags(sc['sales'][am], sc['dows'][am], dv, K)
            for n in LAG_NAMES:
                la[n][hs:hs+N_HOURS] = lg[n]
    for n in LAG_NAMES:
        fd[n] = la[n]
    X = pd.DataFrame(fd)
    for c in CAT_FEATURES:
        X[c] = X[c].astype('category')
    return X, y, stk, sids, pids

if not all(os.path.exists(p) for p in [CACHE_TRAIN_X, CACHE_TRAIN_Y, CACHE_VAL_X, CACHE_VAL_Y, CACHE_VAL_STOCK, CACHE_VAL_SIDS, CACHE_VAL_PIDS]):
    print(f'[{time.time()-T_START:.0f}s] Building dataset (cache miss)...')
    df_train = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
    df_eval = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_eval.parquet'))
    df_full = pd.concat([df_train, df_eval], ignore_index=True)
    df_full['dt_parsed'] = pd.to_datetime(df_full['dt'])
    df_full = df_full.sort_values(['store_id','product_id','dt_parsed']).reset_index(drop=True)
    all_dates = sorted(df_full['dt_parsed'].unique())
    date_to_day = {d: i+1 for i, d in enumerate(all_dates)}
    df_full['day_num'] = df_full['dt_parsed'].map(date_to_day)
    df_full['dow'] = df_full['dt_parsed'].dt.dayofweek

    sales_arr = np.array(df_full['hours_sale'].tolist(), dtype=np.float32)[:, H_START:H_END]
    stock_arr = np.array(df_full['hours_stock_status'].tolist(), dtype=np.int8)[:, H_START:H_END]

    print(f'[{time.time()-T_START:.0f}s]   Building series cache...')
    series_cache = {}
    for (sid, pid), grp in df_full.groupby(['store_id','product_id'], sort=False):
        gs = grp.sort_values('day_num')
        idx = gs.index.values
        series_cache[(sid, pid)] = {
            'days': df_full.loc[idx, 'day_num'].values,
            'dows': df_full.loc[idx, 'dow'].values,
            'sales': sales_arr[idx],
        }
    print(f'[{time.time()-T_START:.0f}s]   {len(series_cache):,} serie')

    print(f'[{time.time()-T_START:.0f}s] Building train set...')
    X_tr, y_tr, _, _, _ = build_dataset_for_split('train', df_full, sales_arr, stock_arr, series_cache)
    X_tr.to_parquet(CACHE_TRAIN_X, index=False)
    np.save(CACHE_TRAIN_Y, y_tr)
    print(f'[{time.time()-T_START:.0f}s]   Train: {len(X_tr):,} rows')

    print(f'[{time.time()-T_START:.0f}s] Building val set...')
    X_va, y_va, stk_va, sids_va, pids_va = build_dataset_for_split('val', df_full, sales_arr, stock_arr, series_cache)
    X_va.to_parquet(CACHE_VAL_X, index=False)
    np.save(CACHE_VAL_Y, y_va); np.save(CACHE_VAL_STOCK, stk_va)
    np.save(CACHE_VAL_SIDS, sids_va); np.save(CACHE_VAL_PIDS, pids_va)
    print(f'[{time.time()-T_START:.0f}s]   Val: {len(X_va):,} rows')
    del df_train, df_eval, df_full, sales_arr, stock_arr, series_cache; gc.collect()
else:
    print(f'[{time.time()-T_START:.0f}s] Loading cached dataset...')

X_tr = pd.read_parquet(CACHE_TRAIN_X)
y_tr = np.load(CACHE_TRAIN_Y)
for c in CAT_FEATURES:
    X_tr[c] = X_tr[c].astype('category')
X_va = pd.read_parquet(CACHE_VAL_X)
y_va = np.load(CACHE_VAL_Y)
stk_va = np.load(CACHE_VAL_STOCK)
sids_va = np.load(CACHE_VAL_SIDS)
pids_va = np.load(CACHE_VAL_PIDS)
for c in CAT_FEATURES:
    X_va[c] = X_va[c].astype('category')

print(f'[{time.time()-T_START:.0f}s] Dataset loaded: train={len(X_tr):,}, val={len(X_va):,}')

train_ds = lgb.Dataset(X_tr, y_tr, categorical_feature=CAT_FEATURES, free_raw_data=False)
val_ds = lgb.Dataset(X_va, y_va, categorical_feature=CAT_FEATURES, reference=train_ds, free_raw_data=False)

# =========================================================================
# 2. WAPE_med per-serie (val, in-stock, min_hours=34)
# =========================================================================
def wape_med_per_series(preds):
    instock = stk_va == 0
    wapes = []
    series_key = sids_va * 100000 + pids_va  # unique series id
    for sk in np.unique(series_key):
        mask = (series_key == sk) & instock
        if mask.sum() < MIN_HOURS_VAL:
            continue
        yt, yp = y_va[mask], preds[mask]
        denom = max(np.abs(yt).sum(), 1e-8)
        wapes.append(np.abs(yt - yp).sum() / denom)
    return float(np.median(wapes)) if wapes else float('nan')

# =========================================================================
# 3. Optuna objective
# =========================================================================
def objective(trial):
    t_trial = time.time()
    hp = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'num_leaves':        trial.suggest_int('num_leaves', 15, 127, log=True),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 100, 2000, log=True),
        'bagging_fraction':  trial.suggest_float('bagging_fraction', 0.2, 0.9, step=0.1),
        'feature_fraction':  trial.suggest_float('feature_fraction', 0.5, 1.0, step=0.1),
        'bagging_freq': 1, 'max_bin': 127, 'verbose': -1, 'num_threads': -1, 'seed': SEED,
    }
    print(f'\n[Trial {trial.number}] HP: {hp}')

    # Train with Optuna pruning callback
    pruning_cb = optuna.integration.LightGBMPruningCallback(trial, metric='l1', valid_name='val')
    model = lgb.train(
        hp, train_ds, num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[train_ds, val_ds], valid_names=['train','val'],
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0), pruning_cb],
    )
    preds = model.predict(X_va, num_iteration=model.best_iteration)
    val_wape = wape_med_per_series(preds)
    elapsed = time.time() - t_trial
    print(f'[Trial {trial.number}] best_iter={model.best_iteration}, val_WAPE_med={val_wape:.4f}, '
          f'elapsed={elapsed:.0f}s')
    return val_wape

# =========================================================================
# 4. Run Optuna
# =========================================================================
print(f'\n[{time.time()-T_START:.0f}s] Creating Optuna study...')
study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=5),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2, interval_steps=1),
    storage=STORAGE, study_name=STUDY_NAME, load_if_exists=True,
)
print(f'  Existing trials: {len(study.trials)}')
remaining = max(0, N_TRIALS - len(study.trials))
print(f'  Remaining to run: {remaining}')

if remaining > 0:
    study.optimize(objective, n_trials=remaining, gc_after_trial=True)

# =========================================================================
# 5. Save results
# =========================================================================
print(f'\n[{time.time()-T_START:.0f}s] Saving results...')
best = study.best_trial
print(f'  Best trial: #{best.number}, val_WAPE_med={best.value:.4f}, params={best.params}')
with open(os.path.join(RESULTS_DIR, 'hpo_lgb_best.json'), 'w') as f:
    json.dump({'best_trial': best.number, 'best_value': best.value,
               'best_params': best.params, 'n_trials': len(study.trials)}, f, indent=2)
study.trials_dataframe().to_parquet(os.path.join(RESULTS_DIR, 'hpo_lgb_trials.parquet'), index=False)
import pickle
with open(os.path.join(RESULTS_DIR, 'hpo_lgb_study.pkl'), 'wb') as f:
    pickle.dump(study, f)
print(f'\n[{time.time()-T_START:.0f}s] DONE in {(time.time()-T_START)/60:.1f} min')
