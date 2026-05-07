"""
03_baseline_naive.py — Fase A1: Baseline Naive (ore 6-22)
==========================================================
Serie temporali ristrette alle ore 6-22 (17 ore, orario operativo).

4 modelli naive su dati sporchi (S_obs con zeri da stockout):
  1. Global Mean  2. DoW Mean  3. Naive Direct  4. MA (K selezionato su val)

Eseguire con: freshnet/bin/python notebooks_622/03_baseline_naive.py
"""
import sys, os, functools, numpy as np, pandas as pd
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

H_START, H_END = 6, 23   # slice [6:23] → ore 6-22 inclusive
N_HOURS = H_END - H_START # 17
MA_K_CANDIDATES = [3, 5, 7, 10, 14, 21, 28, 42, 56, 83]

# ===========================================================================
print('=' * 72)
print('  FASE A1 — BASELINE NAIVE (ore 6-22)')
print('=' * 72)

print('\n1. Caricamento dati...')
df_train = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df_eval = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_eval.parquet'))
df_train['dt_parsed'] = pd.to_datetime(df_train['dt'])
df_eval['dt_parsed'] = pd.to_datetime(df_eval['dt'])
df_full = pd.concat([df_train, df_eval], ignore_index=True)
df_full = df_full.sort_values(['store_id', 'product_id', 'dt_parsed']).reset_index(drop=True)
all_dates = sorted(df_full['dt_parsed'].unique())
date_to_day = {d: i + 1 for i, d in enumerate(all_dates)}
df_full['day_num'] = df_full['dt_parsed'].map(date_to_day)
df_full['dow'] = df_full['dt_parsed'].dt.dayofweek
n_series = df_full.groupby(['store_id', 'product_id']).ngroups
print(f'  Full: {len(df_full):,} righe, {len(all_dates)} giorni, {n_series:,} serie')
del df_train, df_eval

# SLICE TO HOURS 6-22
print(f'  Slicing ore {H_START}-{H_END-1} ({N_HOURS} ore per giorno)...')
sales_all = np.array(df_full['hours_sale'].tolist(), dtype=np.float64)[:, H_START:H_END]
stock_all = np.array(df_full['hours_stock_status'].tolist(), dtype=np.int8)[:, H_START:H_END]
print(f'  Shape: sales={sales_all.shape}, stock={stock_all.shape}')

# Build series list
print('  Building series list...')
series_list = []
for (sid, pid), grp in df_full.groupby(['store_id', 'product_id'], sort=False):
    grp_s = grp.sort_values('day_num')
    idx = grp_s.index.values
    series_list.append({'store_id': sid, 'product_id': pid, 'idx': idx,
                        'days': df_full.loc[idx, 'day_num'].values,
                        'dows': df_full.loc[idx, 'dow'].values})
print(f'  {len(series_list):,} serie')

# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
def eval_instock(pred, obs, stk):
    instock = stk == 0
    ph, oh = pred[instock], obs[instock]
    sae_h, sao_h = np.abs(ph - oh).sum(), np.abs(oh).sum()
    se_h, so_h = (ph - oh).sum(), oh.sum()
    nd = pred.shape[0]
    sae_d, sao_d, se_d, so_d, nvd = 0., 0., 0., 0., 0
    for d in range(nd):
        m = instock[d]
        if m.any():
            pv, ov = pred[d, m].sum(), obs[d, m].sum()
            sae_d += abs(pv-ov); sao_d += abs(ov); se_d += pv-ov; so_d += ov; nvd += 1
    return {'sae_h':sae_h,'sao_h':sao_h,'se_h':se_h,'so_h':so_h,
            'sae_d':sae_d,'sao_d':sao_d,'se_d':se_d,'so_d':so_d,
            'n_in':int(instock.sum()),'n_vd':nvd}

# Profile functions
def global_mean_profile(sales, days, max_day):
    m = days <= max_day
    return sales[m].mean(axis=0) if m.any() else np.zeros(N_HOURS)

def dow_mean_profiles(sales, days, dows, max_day):
    m = days <= max_day
    profs = {}
    for dow in range(7):
        dm = m & (dows == dow)
        profs[dow] = sales[dm].mean(axis=0) if dm.any() else (sales[m].mean(axis=0) if m.any() else np.zeros(N_HOURS))
    return profs

def naive_direct_profile(sales, days, anchor):
    m = days == anchor
    if m.any(): return sales[m][0]
    a = days <= anchor
    return sales[a][-1] if a.any() else np.zeros(N_HOURS)

def ma_profile(sales, days, anchor, K):
    a = days <= anchor
    if not a.any(): return np.zeros(N_HOURS)
    return sales[a][-min(K, a.sum()):].mean(axis=0)

# ---------------------------------------------------------------------------
# MA K selection on val
# ---------------------------------------------------------------------------
print('\n2. Selezione K per MA su validation...')
ma_val = {}
for K in MA_K_CANDIDATES:
    acc = {'sae_h':0.,'sao_h':0.}
    for ser in series_list:
        idx=ser['idx']; days=ser['days']; sales=sales_all[idx]; stock=stock_all[idx]
        em = (days>=84)&(days<=90)
        if not em.any(): continue
        prof = ma_profile(sales, days, 83, K)
        pred = np.tile(prof, (em.sum(), 1))
        instock = stock[em] == 0
        acc['sae_h'] += np.abs(pred[instock]-sales[em][instock]).sum()
        acc['sao_h'] += np.abs(sales[em][instock]).sum()
    w = acc['sae_h']/acc['sao_h'] if acc['sao_h']>0 else np.nan
    ma_val[K] = w
    print(f'  K={K:>3}: WAPE_h={w:.4f}')

BEST_K = min(ma_val, key=ma_val.get)
print(f'  Best K: {BEST_K}')

# ---------------------------------------------------------------------------
# Full evaluation on val + test
# ---------------------------------------------------------------------------
print('\n3. Valutazione completa...')
MODELS = ['Global Mean', 'DoW Mean', 'Naive Direct', f'MA (K={BEST_K})']
SPLITS = {'val': (83, 84, 90, 83), 'test': (90, 91, 97, 90)}
all_results = {m: {} for m in MODELS}

for split_name, (prof_max, d_min, d_max, anchor) in SPLITS.items():
    print(f'\n  --- {split_name} ---')
    pooled = {m: {k:0. for k in ['sae_h','sao_h','se_h','so_h','sae_d','sao_d','se_d','so_d']}
              for m in MODELS}
    per_series = {m: [] for m in MODELS}

    for si, ser in enumerate(series_list):
        if (si+1) % 10000 == 0: print(f'    ... {si+1:,}/{len(series_list):,}')
        idx=ser['idx']; days=ser['days']; dows=ser['dows']
        sales=sales_all[idx]; stock=stock_all[idx]
        em = (days>=d_min)&(days<=d_max)
        if not em.any(): continue
        ne=em.sum(); obs=sales[em]; stk=stock[em]; ed=dows[em]

        profiles = {
            'Global Mean': np.tile(global_mean_profile(sales, days, prof_max), (ne,1)),
            'DoW Mean': np.array([dow_mean_profiles(sales, days, dows, prof_max)[d] for d in ed]),
            'Naive Direct': np.tile(naive_direct_profile(sales, days, anchor), (ne,1)),
            f'MA (K={BEST_K})': np.tile(ma_profile(sales, days, anchor, BEST_K), (ne,1)),
        }

        for mn, pred in profiles.items():
            m = eval_instock(pred, obs, stk)
            for k in pooled[mn]: pooled[mn][k] += m[k]
            hw=m['sae_h']/m['sao_h'] if m['sao_h']>0 else np.nan
            hwp=m['se_h']/m['so_h'] if m['so_h']!=0 else np.nan
            dw=m['sae_d']/m['sao_d'] if m['sao_d']>0 else np.nan
            dwp=m['se_d']/m['so_d'] if m['so_d']!=0 else np.nan
            per_series[mn].append({'store_id':ser['store_id'],'product_id':ser['product_id'],
                                   'hourly_wape':hw,'hourly_wpe':hwp,'daily_wape':dw,'daily_wpe':dwp,
                                   'n_hours_instock':m['n_in'],'n_days_valid':m['n_vd']})

    for mn in MODELS:
        p = pooled[mn]
        pm = {'hourly_wape':p['sae_h']/p['sao_h'] if p['sao_h']>0 else np.nan,
              'hourly_wpe':p['se_h']/p['so_h'] if p['so_h']!=0 else np.nan,
              'daily_wape':p['sae_d']/p['sao_d'] if p['sao_d']>0 else np.nan,
              'daily_wpe':p['se_d']/p['so_d'] if p['so_d']!=0 else np.nan}
        ps = pd.DataFrame(per_series[mn])
        mm = {c: ps[c].dropna().median() for c in ['hourly_wape','hourly_wpe','daily_wape','daily_wpe']}
        all_results[mn][split_name] = {'pooled': pm, 'median': mm, 'ps': ps}
        safe = mn.lower().replace(' ','_').replace('(','').replace(')','').replace('=','')
        ps.to_parquet(os.path.join(RESULTS_DIR, f'naive_{safe}_{split_name}_per_series.parquet'), index=False)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
print('\n' + '=' * 72)
print('  RISULTATI — BASELINE NAIVE (ore 6-22, test, in-stock)')
print('=' * 72)
print(f'\n  {"Model":<20} {"WAPE_h pool":>12} {"WPE_h pool":>11} '
      f'{"WAPE_h med":>11} {"WPE_h med":>10}')
print('  ' + '-' * 68)
for mn in MODELS:
    r = all_results[mn]['test']
    p, m = r['pooled'], r['median']
    print(f'  {mn:<20} {p["hourly_wape"]:>12.4f} {p["hourly_wpe"]:>11.4f} '
          f'{m["hourly_wape"]:>11.4f} {m["hourly_wpe"]:>10.4f}')

print(f'\n  MA K selection: {BEST_K}')
print('\n' + '=' * 72)
print('  DONE — 03_baseline_naive.py (ore 6-22)')
print('=' * 72)
