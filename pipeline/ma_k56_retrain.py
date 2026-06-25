"""
ma_k56_retrain.py — Re-train MA forecaster with K=56 across all 14 imputers.

Reason: K=56 is the optimum under the per-series median WAPE criterion
(min_hours=34), which is the same criterion used by the Optuna HPO of
LGB_M5/MLP_M5/TFT. The previous K=21 was selected under pooled WAPE
(inconsistent with HPO).

Output: for each imputer in {no_imp, media_cond, media_glob, mediana_cond,
mediana_glob, lgb, dlinear, forward_fill, seasonal_naive, linear_interp,
saits, itransformer, timesnet, imputeformer}, write
  results/{imp}__ma_k56_test_per_series.parquet
plus results/naive_ma_k56_test_per_series.parquet (no_imp) and the val twin.

Window: hours 6-22, anchor day 90, test days 91-97.
"""
import os, functools, numpy as np, pandas as pd
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
COMPLETED_DIR = os.path.join(DATA_DIR, 'completed_sales_622')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

H_START, H_END = 6, 23
N_HOURS = H_END - H_START
MA_K = 56

IMPUTERS = [
    'media_cond', 'media_glob', 'mediana_cond', 'mediana_glob',
    'lgb', 'dlinear', 'forward_fill', 'seasonal_naive', 'linear_interp',
    'saits', 'itransformer', 'timesnet', 'imputeformer',
]

print('=' * 72)
print(f'  MA re-train with K={MA_K}  ({len(IMPUTERS)} imputer + no_imp)')
print('=' * 72)

print('\n1. Loading base data...')
df_train_hf = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df_eval = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_eval.parquet'))
df_train_hf['dt_parsed'] = pd.to_datetime(df_train_hf['dt'])
df_eval['dt_parsed'] = pd.to_datetime(df_eval['dt'])
df_full = pd.concat([df_train_hf, df_eval], ignore_index=True)
df_full = df_full.sort_values(['store_id', 'product_id', 'dt_parsed']).reset_index(drop=True)
all_dates = sorted(df_full['dt_parsed'].unique())
date_to_day = {d: i + 1 for i, d in enumerate(all_dates)}
df_full['day_num'] = df_full['dt_parsed'].map(date_to_day)
sales_orig = np.array(df_full['hours_sale'].tolist(), dtype=np.float64)[:, H_START:H_END]
stock_orig = np.array(df_full['hours_stock_status'].tolist(), dtype=np.int8)[:, H_START:H_END]
del df_train_hf, df_eval

series_list = []
for (sid, pid), grp in df_full.groupby(['store_id', 'product_id'], sort=False):
    gs = grp.sort_values('day_num')
    idx = gs.index.values
    series_list.append({'store_id': sid, 'product_id': pid, 'idx': idx,
                        'days': df_full.loc[idx, 'day_num'].values})
print(f'  {len(series_list):,} series')


def ma(sales, days, anchor, K):
    a = days <= anchor
    if not a.any():
        return np.zeros(N_HOURS)
    return sales[a][-min(K, a.sum()):].mean(axis=0)


def eval_instock(pred, obs, stk):
    instock = stk == 0
    ph, oh = pred[instock], obs[instock]
    sae_h, sao_h = np.abs(ph - oh).sum(), np.abs(oh).sum()
    se_h, so_h = (ph - oh).sum(), oh.sum()
    nd = pred.shape[0]
    sae_d = sao_d = se_d = so_d = 0.0
    nvd = 0
    for d in range(nd):
        m = instock[d]
        if m.any():
            pv = pred[d, m].sum(); ov = obs[d, m].sum()
            sae_d += abs(pv - ov); sao_d += abs(ov)
            se_d += pv - ov; so_d += ov; nvd += 1
    return {'sae_h': sae_h, 'sao_h': sao_h, 'se_h': se_h, 'so_h': so_h,
            'sae_d': sae_d, 'sao_d': sao_d, 'se_d': se_d, 'so_d': so_d,
            'n_in': int(instock.sum()), 'n_vd': nvd}


def run_one(completed_sales, label, eval_min, eval_max, anchor, out_path):
    pooled = {k: 0.0 for k in ['sae_h', 'sao_h', 'se_h', 'so_h',
                                'sae_d', 'sao_d', 'se_d', 'so_d']}
    recs = []
    for si, ser in enumerate(series_list):
        if (si + 1) % 10000 == 0:
            print(f'    ... {label}: {si + 1:,}/{len(series_list):,}')
        idx = ser['idx']; days = ser['days']
        sales_cs = completed_sales[idx]; obs_real = sales_orig[idx]; stock = stock_orig[idx]
        em = (days >= eval_min) & (days <= eval_max)
        if not em.any():
            continue
        pred = np.tile(ma(sales_cs, days, anchor, MA_K), (em.sum(), 1))
        obs = obs_real[em]; stk = stock[em]
        m = eval_instock(pred, obs, stk)
        for k in pooled:
            pooled[k] += m[k]
        hw = m['sae_h'] / m['sao_h'] if m['sao_h'] > 0 else np.nan
        hwp = m['se_h'] / m['so_h'] if m['so_h'] != 0 else np.nan
        dw = m['sae_d'] / m['sao_d'] if m['sao_d'] > 0 else np.nan
        dwp = m['se_d'] / m['so_d'] if m['so_d'] != 0 else np.nan
        recs.append({'store_id': ser['store_id'], 'product_id': ser['product_id'],
                     'hourly_wape': hw, 'hourly_wpe': hwp,
                     'daily_wape': dw, 'daily_wpe': dwp,
                     'n_hours_instock': m['n_in'], 'n_days_valid': m['n_vd']})
    p = pooled
    ps = pd.DataFrame(recs)
    ps.to_parquet(out_path, index=False)
    pm_hw = p['sae_h'] / p['sao_h'] if p['sao_h'] > 0 else np.nan
    pm_wpe = p['se_h'] / p['so_h'] if p['so_h'] != 0 else np.nan
    mm_hw = ps['hourly_wape'].dropna().median() if len(ps) else np.nan
    print(f'  {label}: WAPE_h pool={pm_hw:.4f}, med={mm_hw:.4f}, '
          f'WPE_h pool={pm_wpe:.4f} -> {out_path}')


# no_imp: use sales_orig directly. Match script 01 schema: naive_ma_k56_test/val_per_series.parquet
print('\n2. Re-train MA on no_imp (test + val)...')
run_one(sales_orig, 'no_imp/test', 91, 97, 90,
        os.path.join(RESULTS_DIR, f'naive_ma_k{MA_K}_test_per_series.parquet'))
run_one(sales_orig, 'no_imp/val', 84, 90, 83,
        os.path.join(RESULTS_DIR, f'naive_ma_k{MA_K}_val_per_series.parquet'))

print(f'\n3. Re-train MA across {len(IMPUTERS)} imputers (test)...')
for imp in IMPUTERS:
    cs_path = os.path.join(COMPLETED_DIR, f'{imp}.parquet')
    if not os.path.exists(cs_path):
        print(f'  SKIP {imp}: {cs_path} not found')
        continue
    df_cs = pd.read_parquet(cs_path)
    cs_sales = np.array(df_cs['hours_sale'].tolist(), dtype=np.float64)
    if cs_sales.shape[1] == 24:
        cs_sales = cs_sales[:, H_START:H_END]
    completed = sales_orig.copy()
    cs_keys = (df_cs['store_id'].astype(str) + '_' +
               df_cs['product_id'].astype(str) + '_' +
               df_cs['dt'].astype(str)).values
    full_keys = (df_full['store_id'].astype(str) + '_' +
                 df_full['product_id'].astype(str) + '_' +
                 df_full['dt'].astype(str)).values
    km = dict(zip(cs_keys, range(len(df_cs))))
    for i in range(len(df_full)):
        k = full_keys[i]
        if k in km:
            completed[i] = cs_sales[km[k]]
    del df_cs, cs_sales, km
    out = os.path.join(RESULTS_DIR, f'{imp}__ma_k{MA_K}_test_per_series.parquet')
    run_one(completed, imp, 91, 97, 90, out)

print('\n' + '=' * 72)
print(f'  DONE — MA K={MA_K} retrain complete')
print('=' * 72)
