"""
09_reeval_6_22.py — Ricalcolo metriche solo ore 6-22 (orario operativo)
========================================================================
Ricalcola TUTTE le metriche della matrice restringendo la valutazione
alle ore 6-22 (escluse ore 0-5 e 23, fuori orario operativo).

Include:
  - Naive Fase A: Global Mean, DoW Mean, MA (K=21), Naive Direct
  - ML Fase A: LGB no-lags, LGB M5-lags, MLP no-lags, MLP M5-lags
    (ricarica modelli salvati da results/)
  - Naive Fase B2: tutte le combinazioni imputer × naive forecaster
  - ML B2 NON inclusi (serve retrain, vedi 09b/09c)

Eseguire con: freshnet/bin/python notebooks_final/09_reeval_6_22.py
"""

import sys, os, gc, time, functools
import numpy as np, pandas as pd

print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)

import lightgbm as lgb
import torch
import torch.nn as nn

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
COMPLETED_DIR = os.path.join(DATA_DIR, 'completed_sales')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = 'mps' if torch.backends.mps.is_available() else (
    'cuda' if torch.cuda.is_available() else 'cpu')

CONT_FEATURES = ['discount', 'avg_temperature', 'avg_humidity',
                  'precpt', 'avg_wind_level', 'holiday_flag', 'activity_flag']
CAT_FEATURES_LGB = ['store_id', 'product_id', 'city_id', 'dow', 'hour']
LAG_FEATURE_NAMES = [
    'lag_1d', 'lag_7d', 'lag_14d', 'rmean_7d', 'rmean_14d', 'rstd_7d',
    'lag_dow', 'rmean_dow', 'daily_total_lag1', 'daily_total_rmean7', 'momentum_1d_7d',
]
EMB_DIMS = {'store_id': 32, 'product_id': 32, 'city_id': 8, 'dow': 4}
CARDINALITIES = {'store_id': 898, 'product_id': 865, 'city_id': 18, 'dow': 7}

# RESTRICTION: only evaluate hours 6-22
EVAL_HOURS = set(range(6, 23))
HOUR_MASK_24 = np.array([h in EVAL_HOURS for h in range(24)])  # (24,)
MA_K = 21

IMPUTERS = {'media_cond': 'Media condizionata', 'media_glob': 'Media globale',
            'mediana_cond': 'Mediana condizionata', 'lgb': 'LGB imputer'}

# ===========================================================================
print('=' * 72)
print('  RICALCOLO METRICHE — Solo ore 6-22')
print('=' * 72)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print('\n1. Caricamento dati...')
df_train_hf = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df_eval = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_eval.parquet'))
df_train_hf['dt_parsed'] = pd.to_datetime(df_train_hf['dt'])
df_eval['dt_parsed'] = pd.to_datetime(df_eval['dt'])
df_full = pd.concat([df_train_hf, df_eval], ignore_index=True)
df_full = df_full.sort_values(['store_id', 'product_id', 'dt_parsed']).reset_index(drop=True)
all_dates = sorted(df_full['dt_parsed'].unique())
date_to_day = {d: i + 1 for i, d in enumerate(all_dates)}
df_full['day_num'] = df_full['dt_parsed'].map(date_to_day)
df_full['dow'] = df_full['dt_parsed'].dt.dayofweek

sales_orig = np.array(df_full['hours_sale'].tolist(), dtype=np.float32)
stock_orig = np.array(df_full['hours_stock_status'].tolist(), dtype=np.float32)
print(f'  Full: {len(df_full):,} righe')
print(f'  Eval hours: {sorted(EVAL_HOURS)}')
del df_train_hf, df_eval

# Build series list
print('  Building series list...')
series_list = []
for (sid, pid), grp in df_full.groupby(['store_id', 'product_id'], sort=False):
    grp_s = grp.sort_values('day_num')
    idx = grp_s.index.values
    series_list.append({
        'store_id': sid, 'product_id': pid, 'idx': idx,
        'days': df_full.loc[idx, 'day_num'].values,
        'dows': df_full.loc[idx, 'dow'].values,
    })
print(f'  {len(series_list):,} serie')


# ---------------------------------------------------------------------------
# 2. Evaluation helper (hours 6-22 only)
# ---------------------------------------------------------------------------
def eval_622(pred_24, obs_24, stock_24):
    """Compute metrics on hours 6-22, in-stock only.
    pred_24, obs_24, stock_24: (N_days, 24)
    Returns: per-series dict with hourly/daily WAPE/WPE + accumulators for pooled.
    """
    # Mask: in-stock AND hour 6-22
    eval_mask = (stock_24 == 0) & HOUR_MASK_24[np.newaxis, :]

    p_h, o_h = pred_24[eval_mask], obs_24[eval_mask]
    sae_h = np.abs(p_h - o_h).sum()
    sao_h = np.abs(o_h).sum()
    se_h = (p_h - o_h).sum()
    so_h = o_h.sum()

    n_d = pred_24.shape[0]
    sae_d, sao_d, se_d, so_d, n_vd = 0., 0., 0., 0., 0
    for d in range(n_d):
        m = eval_mask[d]
        if m.any():
            pv, ov = pred_24[d, m].sum(), obs_24[d, m].sum()
            sae_d += abs(pv - ov); sao_d += abs(ov)
            se_d += pv - ov; so_d += ov; n_vd += 1

    return {
        'sae_h': sae_h, 'sao_h': sao_h, 'se_h': se_h, 'so_h': so_h,
        'sae_d': sae_d, 'sao_d': sao_d, 'se_d': se_d, 'so_d': so_d,
        'n_in': int(eval_mask.sum()), 'n_vd': n_vd,
    }


def finalize_metrics(pooled, per_series_records, label):
    """Compute pooled and median metrics from accumulators."""
    p = pooled
    pooled_m = {
        'hourly_wape': p['sae_h'] / p['sao_h'] if p['sao_h'] > 0 else np.nan,
        'hourly_wpe': p['se_h'] / p['so_h'] if p['so_h'] != 0 else np.nan,
        'daily_wape': p['sae_d'] / p['sao_d'] if p['sao_d'] > 0 else np.nan,
        'daily_wpe': p['se_d'] / p['so_d'] if p['so_d'] != 0 else np.nan,
    }
    ps_df = pd.DataFrame(per_series_records)
    median_m = {c: ps_df[c].dropna().median()
                for c in ['hourly_wape', 'hourly_wpe', 'daily_wape', 'daily_wpe']}
    return pooled_m, median_m, ps_df


# ---------------------------------------------------------------------------
# 3. Naive profile functions
# ---------------------------------------------------------------------------
def global_mean_profile(sales, days, max_day):
    m = days <= max_day
    return sales[m].mean(axis=0) if m.any() else np.zeros(24)

def dow_mean_profiles(sales, days, dows, max_day):
    m = days <= max_day
    profs = {}
    for dow in range(7):
        dm = m & (dows == dow)
        profs[dow] = sales[dm].mean(axis=0) if dm.any() else (sales[m].mean(axis=0) if m.any() else np.zeros(24))
    return profs

def ma_profile(sales, days, anchor, K):
    a = days <= anchor
    if not a.any(): return np.zeros(24)
    return sales[a][-min(K, a.sum()):].mean(axis=0)

def naive_direct_profile(sales, days, anchor):
    m = days == anchor
    if m.any(): return sales[m][0]
    a = days <= anchor
    return sales[a][-1] if a.any() else np.zeros(24)


def run_naive_eval(sales_source, label_prefix, forecasters=None):
    """Run all naive forecasters on test set with 6-22 eval.
    sales_source: (N_full, 24) — sales array to compute profiles from (S_obs or completed_sales).
    """
    if forecasters is None:
        forecasters = ['Global Mean', 'DoW Mean', f'MA (K={MA_K})', 'Naive Direct']

    results = {}
    for fc in forecasters:
        pooled = {k: 0. for k in ['sae_h','sao_h','se_h','so_h','sae_d','sao_d','se_d','so_d']}
        ps_recs = []

        for si, ser in enumerate(series_list):
            if (si + 1) % 10000 == 0:
                print(f'      ... {si+1:,}/{len(series_list):,}')
            idx = ser['idx']; days = ser['days']; dows = ser['dows']
            sales = sales_source[idx]; obs_real = sales_orig[idx]; stock = stock_orig[idx]

            eval_mask = (days >= 91) & (days <= 97)
            if not eval_mask.any(): continue
            n_e = eval_mask.sum()
            obs = obs_real[eval_mask]; stk = stock[eval_mask]; ed = dows[eval_mask]

            if fc == 'Global Mean':
                pred = np.tile(global_mean_profile(sales, days, 90), (n_e, 1))
            elif fc == 'DoW Mean':
                profs = dow_mean_profiles(sales, days, dows, 90)
                pred = np.array([profs[d] for d in ed])
            elif fc.startswith('MA'):
                pred = np.tile(ma_profile(sales, days, 90, MA_K), (n_e, 1))
            elif fc == 'Naive Direct':
                pred = np.tile(naive_direct_profile(sales, days, 90), (n_e, 1))

            m = eval_622(pred, obs, stk)
            for k in pooled: pooled[k] += m[k]

            hw = m['sae_h']/m['sao_h'] if m['sao_h']>0 else np.nan
            hwp = m['se_h']/m['so_h'] if m['so_h']!=0 else np.nan
            dw = m['sae_d']/m['sao_d'] if m['sao_d']>0 else np.nan
            dwp = m['se_d']/m['so_d'] if m['so_d']!=0 else np.nan
            ps_recs.append({'store_id': ser['store_id'], 'product_id': ser['product_id'],
                            'hourly_wape': hw, 'hourly_wpe': hwp,
                            'daily_wape': dw, 'daily_wpe': dwp,
                            'n_hours_instock': m['n_in'], 'n_days_valid': m['n_vd']})

        pm, mm, ps_df = finalize_metrics(pooled, ps_recs, fc)
        key = f'{label_prefix}__{fc.lower().replace(" ","_").replace("(","").replace(")","").replace("=","")}'
        ps_df.to_parquet(os.path.join(RESULTS_DIR, f'{key}_622_test.parquet'), index=False)
        results[fc] = {'pooled': pm, 'median': mm}
        print(f'    {label_prefix} × {fc}: WAPE_h pool={pm["hourly_wape"]:.4f}, '
              f'med={mm["hourly_wape"]:.4f}, WPE_h={pm["hourly_wpe"]:.4f}')

    return results


# ===========================================================================
# 4. FASE A — Naive
# ===========================================================================
print('\n' + '=' * 72)
print('  FASE A — NAIVE (ore 6-22)')
print('=' * 72)
t0 = time.time()
naive_a = run_naive_eval(sales_orig, 'no_imp')
print(f'  Tempo: {time.time()-t0:.0f}s')


# ===========================================================================
# 5. FASE A — ML (reload saved models)
# ===========================================================================
print('\n' + '=' * 72)
print('  FASE A — ML (ore 6-22, reload modelli salvati)')
print('=' * 72)

# Helper: build LGB test dataset
def build_lgb_test(use_lags, series_cache_for_lags=None):
    mask = (df_full['day_num'] >= 91) & (df_full['day_num'] <= 97)
    ds = df_full[mask]; idx = np.where(mask.values)[0]; nd = len(ds)
    sids = ds['store_id'].values; pids = ds['product_id'].values
    cids = ds['city_id'].values; dows_d = ds['dow'].values
    conts = ds[CONT_FEATURES].values.astype(np.float32)
    dnums = ds['day_num'].values

    nh = nd * 24
    hrs = np.tile(np.arange(24, dtype=np.int32), nd)
    fd = {'store_id': np.repeat(sids,24), 'product_id': np.repeat(pids,24),
          'city_id': np.repeat(cids,24), 'dow': np.repeat(dows_d,24), 'hour': hrs}
    ch = np.repeat(conts, 24, axis=0)
    for j, c in enumerate(CONT_FEATURES): fd[c] = ch[:, j]

    if use_lags:
        sc = series_cache_for_lags
        la = {n: np.full(nh, np.nan, dtype=np.float32) for n in LAG_FEATURE_NAMES}
        for ri in range(nd):
            sid, pid, d, dv = sids[ri], pids[ri], dnums[ri], dows_d[ri]
            s = sc[(sid, pid)]
            am = s['days'] <= 90; K = int(am.sum()); hs = ri * 24
            if K > 0:
                lg = _compute_lags(s['sales'][am], s['dows'][am], dv, K)
                for n in LAG_FEATURE_NAMES: la[n][hs:hs+24] = lg[n]
        for n in LAG_FEATURE_NAMES: fd[n] = la[n]

    X = pd.DataFrame(fd)
    for c in CAT_FEATURES_LGB: X[c] = X[c].astype('category')
    y = sales_orig[idx].ravel().astype(np.float32)
    stk = stock_orig[idx].ravel().astype(np.float32)
    return X, y, stk, np.repeat(sids, 24), np.repeat(pids, 24)


def _compute_lags(avail_sales, avail_dows, dow, K):
    z = np.float32
    L = {n: np.full(24, np.nan, dtype=z) for n in LAG_FEATURE_NAMES}
    if K == 0: return L
    L['lag_1d'] = avail_sales[-1]
    if K>=7: L['lag_7d'] = avail_sales[-7]
    if K>=14: L['lag_14d'] = avail_sales[-14]
    if K>=7: L['rmean_7d'] = avail_sales[-7:].mean(0)
    if K>=14: L['rmean_14d'] = avail_sales[-14:].mean(0)
    if K>=2: L['rstd_7d'] = avail_sales[-min(7,K):].std(0)
    sd = avail_dows == dow
    if sd.any():
        ds = avail_sales[sd]; L['lag_dow'] = ds[-1]; L['rmean_dow'] = ds.mean(0)
    dt = avail_sales.sum(1)
    L['daily_total_lag1'] = np.full(24, dt[-1], dtype=z)
    if K>=7: L['daily_total_rmean7'] = np.full(24, dt[-7:].mean(), dtype=z)
    r, l = L['rmean_7d'], L['lag_1d']
    if not np.isnan(r).all():
        v = (~np.isnan(l)) & (~np.isnan(r)) & (r > 0)
        if v.any():
            m = np.full(24, np.nan, dtype=z); m[v] = l[v]/r[v]; L['momentum_1d_7d'] = m
    return L


def eval_lgb_622(preds_flat, y_flat, stk_flat, sids_flat, pids_flat, label):
    """Evaluate LGB predictions with 6-22 restriction."""
    nd = len(preds_flat) // 24
    pr = preds_flat.reshape(nd, 24); ob = y_flat.reshape(nd, 24)
    sk = stk_flat.reshape(nd, 24)
    sid_d = sids_flat.reshape(nd, 24)[:, 0]; pid_d = pids_flat.reshape(nd, 24)[:, 0]

    pooled = {k: 0. for k in ['sae_h','sao_h','se_h','so_h','sae_d','sao_d','se_d','so_d']}
    ps_recs = []

    # Per-series via grouping by day-level sid/pid
    sm = {}
    for d in range(nd):
        k = (sid_d[d], pid_d[d])
        if k not in sm: sm[k] = []
        sm[k].append(d)

    for (sid, pid), days_idx in sm.items():
        pred_s = pr[days_idx]; obs_s = ob[days_idx]; stk_s = sk[days_idx]
        m = eval_622(pred_s, obs_s, stk_s)
        for k2 in pooled: pooled[k2] += m[k2]
        hw = m['sae_h']/m['sao_h'] if m['sao_h']>0 else np.nan
        hwp = m['se_h']/m['so_h'] if m['so_h']!=0 else np.nan
        dw = m['sae_d']/m['sao_d'] if m['sao_d']>0 else np.nan
        dwp = m['se_d']/m['so_d'] if m['so_d']!=0 else np.nan
        ps_recs.append({'store_id': sid, 'product_id': pid,
                        'hourly_wape': hw, 'hourly_wpe': hwp,
                        'daily_wape': dw, 'daily_wpe': dwp,
                        'n_hours_instock': m['n_in'], 'n_days_valid': m['n_vd']})

    pm, mm, ps_df = finalize_metrics(pooled, ps_recs, label)
    ps_df.to_parquet(os.path.join(RESULTS_DIR, f'{label}_622_test.parquet'), index=False)
    print(f'    {label}: WAPE_h pool={pm["hourly_wape"]:.4f}, '
          f'med={mm["hourly_wape"]:.4f}, WPE_h={pm["hourly_wpe"]:.4f}')
    return {'pooled': pm, 'median': mm}


# Build series cache for S_obs lags
print('  Building series cache (S_obs)...')
sc_obs = {}
for (sid, pid), grp in df_full.groupby(['store_id', 'product_id'], sort=False):
    gs = grp.sort_values('day_num')
    idx = gs.index.values
    sc_obs[(sid, pid)] = {'days': gs['day_num'].values, 'dows': gs['dow'].values,
                           'sales': sales_orig[idx]}
print(f'  {len(sc_obs):,} serie')

ml_a_results = {}

# LGB no-lags
print('\n  LGB (no lags)...')
model_lgb_nl = lgb.Booster(model_file=os.path.join(RESULTS_DIR, 'lgb_nolags.txt'))
X_te, y_te, stk_te, sid_te, pid_te = build_lgb_test(False)
preds = np.clip(model_lgb_nl.predict(X_te), 0, None)
ml_a_results['LGB (no lags)'] = eval_lgb_622(preds, y_te, stk_te, sid_te, pid_te, 'no_imp__lgb_nolags')
del X_te, preds, model_lgb_nl; gc.collect()

# LGB M5-lags
print('  LGB (M5 lags)...')
model_lgb_m5 = lgb.Booster(model_file=os.path.join(RESULTS_DIR, 'lgb_m5lags.txt'))
X_te, y_te, stk_te, sid_te, pid_te = build_lgb_test(True, sc_obs)
preds = np.clip(model_lgb_m5.predict(X_te), 0, None)
ml_a_results['LGB (M5 lags)'] = eval_lgb_622(preds, y_te, stk_te, sid_te, pid_te, 'no_imp__lgb_m5lags')
del X_te, preds, model_lgb_m5; gc.collect()

# MLP helper
class RetailMLP(nn.Module):
    def __init__(self, n_cont, n_lags):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            n: nn.Embedding(CARDINALITIES[n], EMB_DIMS[n]) for n in EMB_DIMS})
        self.emb_names = ['store_id', 'product_id', 'city_id', 'dow']
        inp = sum(EMB_DIMS.values()) + n_cont + n_lags
        layers = []
        for h in [128, 64]:
            layers += [nn.Linear(inp, h), nn.ReLU()]; inp = h
        layers += [nn.Linear(inp, 24), nn.Softplus()]
        self.mlp = nn.Sequential(*layers)
    def forward(self, cat, cont, lags):
        e = [self.embeddings[n](cat[:, i]) for i, n in enumerate(self.emb_names)]
        x = torch.cat(e + [cont], dim=1)
        if lags.shape[1] > 0: x = torch.cat([x, lags], dim=1)
        return self.mlp(x)


def build_mlp_test(use_lags, sc, cont_mean, cont_std, lag_mean=None, lag_std=None):
    """Build MLP test arrays."""
    cat_l, cont_l, lag_l, tgt_l, stk_l, sid_l, pid_l = [], [], [], [], [], [], []
    for (sid, pid), sd in sc.items():
        days, dows, sales = sd['days'], sd['dows'], sd['sales']
        for idx in range(len(days)):
            d = days[idx]
            if d < 91 or d > 97: continue
            # Get city_id and conts from series_list
            cat_l.append([sid, pid, sd.get('city_id', 0), dows[idx]])
            cont_l.append(sd['conts'][idx] if 'conts' in sd else np.zeros(7, dtype=np.float32))
            tgt_l.append(sales_orig[sd['full_idx'][idx]] if 'full_idx' in sd else sales[idx])
            stk_l.append(stock_orig[sd['full_idx'][idx]] if 'full_idx' in sd else np.zeros(24))
            sid_l.append(sid); pid_l.append(pid)
            if use_lags:
                am = days <= 90; K = int(am.sum())
                ld = _compute_lags(sales[am], dows[am], dows[idx], K) if K > 0 \
                    else {n: np.full(24, np.nan, dtype=np.float32) for n in LAG_FEATURE_NAMES}
                fa, masks = [], np.zeros(11, dtype=np.float32)
                for fi, n in enumerate(LAG_FEATURE_NAMES):
                    arr = ld[n]
                    if not np.isnan(arr).all():
                        masks[fi] = 1.0
                        fa.append(np.where(np.isnan(arr), 0, arr).astype(np.float32))
                    else: fa.append(np.zeros(24, dtype=np.float32))
                fa.append(masks); lag_l.append(np.concatenate(fa))
            else: lag_l.append(np.array([], dtype=np.float32))

    cat_arr = np.array(cat_l, dtype=np.int64)
    cont_arr = (np.array(cont_l, dtype=np.float32) - cont_mean) / cont_std
    tgt_arr = np.array(tgt_l, dtype=np.float32)
    stk_arr = np.array(stk_l, dtype=np.float32)
    lag_arr = np.array(lag_l, dtype=np.float32) if len(lag_l[0]) > 0 else np.zeros((len(cat_l), 0), dtype=np.float32)
    if lag_arr.shape[1] > 0 and lag_mean is not None:
        lag_arr = (lag_arr - lag_mean) / lag_std
    return cat_arr, cont_arr, lag_arr, tgt_arr, stk_arr, np.array(sid_l), np.array(pid_l)


# Build enriched series cache for MLP (needs city_id, conts, full_idx)
print('\n  Building enriched series cache for MLP...')
sc_mlp = {}
for (sid, pid), grp in df_full.groupby(['store_id', 'product_id'], sort=False):
    gs = grp.sort_values('day_num')
    idx = gs.index.values
    sc_mlp[(sid, pid)] = {
        'days': gs['day_num'].values, 'dows': gs['dow'].values,
        'sales': sales_orig[idx], 'city_id': gs['city_id'].values[0],
        'conts': gs[CONT_FEATURES].values.astype(np.float32),
        'full_idx': idx,
    }

# Load normalization params from training (approximate: recompute from train split)
print('  Computing normalization params...')
train_mask = df_full['day_num'].between(2, 83)
cont_train = df_full.loc[train_mask, CONT_FEATURES].values.astype(np.float32)
cont_mean = cont_train.mean(axis=0); cont_std = cont_train.std(axis=0)
cont_std[cont_std < 1e-8] = 1.0
del cont_train

# For lag normalization, we need to build train lag arrays (expensive).
# Instead, load from saved model and use approximate normalization.
# Actually, the .pt file contains model weights but not normalization params.
# We'll recompute from train data.
print('  Building train lag stats (for normalization)...')
lag_vals_train = []
for (sid, pid), sd in sc_mlp.items():
    days, dows, sales = sd['days'], sd['dows'], sd['sales']
    for idx in range(len(days)):
        d = days[idx]
        if d < 2 or d > 83: continue
        am = days <= (d - 1); K = int(am.sum())
        if K > 0:
            ld = _compute_lags(sales[am], dows[am], dows[idx], K)
            fa, masks = [], np.zeros(11, dtype=np.float32)
            for fi, n in enumerate(LAG_FEATURE_NAMES):
                arr = ld[n]
                if not np.isnan(arr).all():
                    masks[fi] = 1.0
                    fa.append(np.where(np.isnan(arr), 0, arr).astype(np.float32))
                else: fa.append(np.zeros(24, dtype=np.float32))
            fa.append(masks)
            lag_vals_train.append(np.concatenate(fa))
    if len(lag_vals_train) > 100000: break  # sample enough for stats

lag_sample = np.array(lag_vals_train[:100000], dtype=np.float32)
lag_mean_m5 = lag_sample.mean(axis=0); lag_std_m5 = lag_sample.std(axis=0)
lag_std_m5[lag_std_m5 < 1e-8] = 1.0
del lag_vals_train, lag_sample

# MLP no-lags
print('\n  MLP (no lags)...')
model_mlp_nl = RetailMLP(7, 0)
model_mlp_nl.load_state_dict(torch.load(os.path.join(RESULTS_DIR, 'mlp_nolags.pt'),
                                         map_location='cpu', weights_only=True))
model_mlp_nl.to(DEVICE).eval()

cat_te, cont_te, lag_te, tgt_te, stk_te, sid_te, pid_te = build_mlp_test(
    False, sc_mlp, cont_mean, cont_std)

with torch.no_grad():
    ap = []
    ct = torch.from_numpy(cat_te).to(DEVICE)
    cot = torch.from_numpy(cont_te).to(DEVICE)
    lt = torch.from_numpy(lag_te).to(DEVICE)
    for s in range(0, len(ct), 10000):
        e = min(s + 10000, len(ct))
        ap.append(model_mlp_nl(ct[s:e], cot[s:e], lt[s:e]).cpu().numpy())
preds_mlp = np.concatenate(ap)

# Eval 6-22
eval_mask = (stk_te == 0) & HOUR_MASK_24[np.newaxis, :]
ph, oh = preds_mlp[eval_mask], tgt_te[eval_mask]
pooled_ph = {'sae_h': np.abs(ph-oh).sum(), 'sao_h': np.abs(oh).sum(),
             'se_h': (ph-oh).sum(), 'so_h': oh.sum()}
# Per-series
sm = {}
for i in range(len(sid_te)):
    k = (sid_te[i], pid_te[i])
    if k not in sm: sm[k] = []
    sm[k].append(i)

pooled_d = {'sae_d': 0., 'sao_d': 0., 'se_d': 0., 'so_d': 0.}
ps_recs = []
for (sid, pid), idxs in sm.items():
    m622 = eval_622(preds_mlp[idxs], tgt_te[idxs], stk_te[idxs])
    for k2 in pooled_d: pooled_d[k2] += m622[k2]
    hw = m622['sae_h']/m622['sao_h'] if m622['sao_h']>0 else np.nan
    hwp = m622['se_h']/m622['so_h'] if m622['so_h']!=0 else np.nan
    dw = m622['sae_d']/m622['sao_d'] if m622['sao_d']>0 else np.nan
    dwp = m622['se_d']/m622['so_d'] if m622['so_d']!=0 else np.nan
    ps_recs.append({'store_id': sid, 'product_id': pid,
                    'hourly_wape': hw, 'hourly_wpe': hwp,
                    'daily_wape': dw, 'daily_wpe': dwp})

pooled_ph.update(pooled_d)
pm, mm, ps_df = finalize_metrics(pooled_ph, ps_recs, 'MLP_nl')
ps_df.to_parquet(os.path.join(RESULTS_DIR, 'no_imp__mlp_nolags_622_test.parquet'), index=False)
ml_a_results['MLP (no lags)'] = {'pooled': pm, 'median': mm}
print(f'    MLP (no lags): WAPE_h pool={pm["hourly_wape"]:.4f}, '
      f'med={mm["hourly_wape"]:.4f}, WPE_h={pm["hourly_wpe"]:.4f}')

del model_mlp_nl, ct, cot, lt, preds_mlp; gc.collect()
if DEVICE == 'mps': torch.mps.empty_cache()

# MLP M5-lags
print('  MLP (M5 lags)...')
model_mlp_m5 = RetailMLP(7, 275)
model_mlp_m5.load_state_dict(torch.load(os.path.join(RESULTS_DIR, 'mlp_m5lags.pt'),
                                         map_location='cpu', weights_only=True))
model_mlp_m5.to(DEVICE).eval()

cat_te, cont_te, lag_te, tgt_te, stk_te, sid_te, pid_te = build_mlp_test(
    True, sc_mlp, cont_mean, cont_std, lag_mean_m5, lag_std_m5)

with torch.no_grad():
    ap = []
    ct = torch.from_numpy(cat_te).to(DEVICE)
    cot = torch.from_numpy(cont_te).to(DEVICE)
    lt = torch.from_numpy(lag_te).to(DEVICE)
    for s in range(0, len(ct), 10000):
        e = min(s + 10000, len(ct))
        ap.append(model_mlp_m5(ct[s:e], cot[s:e], lt[s:e]).cpu().numpy())
preds_mlp = np.concatenate(ap)

sm = {}
for i in range(len(sid_te)):
    k = (sid_te[i], pid_te[i])
    if k not in sm: sm[k] = []
    sm[k].append(i)

pooled_ph2 = {'sae_h': 0., 'sao_h': 0., 'se_h': 0., 'so_h': 0.,
               'sae_d': 0., 'sao_d': 0., 'se_d': 0., 'so_d': 0.}
ps_recs2 = []
for (sid, pid), idxs in sm.items():
    m622 = eval_622(preds_mlp[idxs], tgt_te[idxs], stk_te[idxs])
    for k2 in pooled_ph2: pooled_ph2[k2] += m622[k2]
    hw = m622['sae_h']/m622['sao_h'] if m622['sao_h']>0 else np.nan
    hwp = m622['se_h']/m622['so_h'] if m622['so_h']!=0 else np.nan
    dw = m622['sae_d']/m622['sao_d'] if m622['sao_d']>0 else np.nan
    dwp = m622['se_d']/m622['so_d'] if m622['so_d']!=0 else np.nan
    ps_recs2.append({'store_id': sid, 'product_id': pid,
                     'hourly_wape': hw, 'hourly_wpe': hwp,
                     'daily_wape': dw, 'daily_wpe': dwp})

pm2, mm2, ps_df2 = finalize_metrics(pooled_ph2, ps_recs2, 'MLP_m5')
ps_df2.to_parquet(os.path.join(RESULTS_DIR, 'no_imp__mlp_m5lags_622_test.parquet'), index=False)
ml_a_results['MLP (M5 lags)'] = {'pooled': pm2, 'median': mm2}
print(f'    MLP (M5 lags): WAPE_h pool={pm2["hourly_wape"]:.4f}, '
      f'med={mm2["hourly_wape"]:.4f}, WPE_h={pm2["hourly_wpe"]:.4f}')

del model_mlp_m5, preds_mlp; gc.collect()
if DEVICE == 'mps': torch.mps.empty_cache()


# ===========================================================================
# 6. FASE B2 — Naive su completed_sales
# ===========================================================================
print('\n' + '=' * 72)
print('  FASE B2 — NAIVE su completed_sales (ore 6-22)')
print('=' * 72)

naive_b2 = {}
for imp_key, imp_label in IMPUTERS.items():
    print(f'\n  Imputer: {imp_label}')
    df_cs = pd.read_parquet(os.path.join(COMPLETED_DIR, f'{imp_key}.parquet'))
    cs_sales = np.array(df_cs['hours_sale'].tolist(), dtype=np.float32)
    completed_full = sales_orig.copy()
    cs_keys = (df_cs['store_id'].astype(str)+'_'+df_cs['product_id'].astype(str)+'_'+df_cs['dt']).values
    full_keys = (df_full['store_id'].astype(str)+'_'+df_full['product_id'].astype(str)+'_'+df_full['dt']).values
    km = dict(zip(cs_keys, range(len(df_cs))))
    for i in range(len(df_full)):
        k = full_keys[i]
        if k in km: completed_full[i] = cs_sales[km[k]]
    del df_cs, cs_sales, km

    naive_b2[imp_key] = run_naive_eval(
        completed_full, imp_key,
        forecasters=['Global Mean', 'DoW Mean', f'MA (K={MA_K})'])
    del completed_full; gc.collect()


# ===========================================================================
# 7. Summary
# ===========================================================================
print('\n' + '=' * 72)
print('  RIEPILOGO COMPLETO — Ore 6-22 (test, in-stock)')
print('=' * 72)

print(f'\n  === FASE A ===')
print(f'  {"Model":<20} {"WAPE_h pool":>12} {"WPE_h pool":>11} {"WAPE_h med":>11} {"WPE_h med":>10}')
print('  ' + '-' * 68)

all_naive_a = {**naive_a}
for fc, r in all_naive_a.items():
    p, m = r['pooled'], r['median']
    print(f'  {fc:<20} {p["hourly_wape"]:>12.4f} {p["hourly_wpe"]:>11.4f} '
          f'{m["hourly_wape"]:>11.4f} {m["hourly_wpe"]:>10.4f}')
for fc, r in ml_a_results.items():
    p, m = r['pooled'], r['median']
    print(f'  {fc:<20} {p["hourly_wape"]:>12.4f} {p["hourly_wpe"]:>11.4f} '
          f'{m["hourly_wape"]:>11.4f} {m["hourly_wpe"]:>10.4f}')

print(f'\n  === FASE B2 — Naive su completed_sales ===')
for imp_key, imp_label in IMPUTERS.items():
    for fc, r in naive_b2[imp_key].items():
        p, m = r['pooled'], r['median']
        print(f'  {imp_label:<24} {fc:<16} {p["hourly_wape"]:>10.4f} {p["hourly_wpe"]:>9.4f} '
              f'{m["hourly_wape"]:>10.4f} {m["hourly_wpe"]:>9.4f}')

print('\n  ML B2 (ore 6-22) richiede retrain — vedi 09b/09c.')

print('\n' + '=' * 72)
print('  DONE — 09_reeval_6_22.py')
print('=' * 72)
