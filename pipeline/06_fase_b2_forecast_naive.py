"""
08b_naive_completed.py — Fase B2: Naive forecaster su completed_sales (ore 6-22)
=================================================================================
12 celle: 4 imputer × 3 naive forecaster.
Eseguire con: freshnet/bin/python notebooks_622/08b_naive_completed.py
"""
import sys, os, functools, numpy as np, pandas as pd
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
COMPLETED_DIR = os.path.join(DATA_DIR, 'completed_sales_622')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

H_START, H_END = 6, 23; N_HOURS = H_END - H_START; MA_K = 56
IMPUTERS = {'media_cond':'Media condizionata','media_glob':'Media globale',
            'mediana_cond':'Mediana condizionata','lgb':'LGB imputer',
            'itransformer':'iTransformer','timesnet':'TimesNet',
            'imputeformer':'ImputeFormer'}
# Skip cells already computed (parquet exists)
import glob
existing_cells = {os.path.basename(p).replace('_test_per_series.parquet','')
                  for p in glob.glob(os.path.join(RESULTS_DIR, '*_test_per_series.parquet'))}

print('=' * 72)
print('  FASE B2 — NAIVE su completed_sales (ore 6-22)')
print('=' * 72)

# Load base data
print('\n1. Caricamento dati...')
df_train_hf = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df_eval = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_eval.parquet'))
df_train_hf['dt_parsed'] = pd.to_datetime(df_train_hf['dt'])
df_eval['dt_parsed'] = pd.to_datetime(df_eval['dt'])
df_full = pd.concat([df_train_hf, df_eval], ignore_index=True)
df_full = df_full.sort_values(['store_id','product_id','dt_parsed']).reset_index(drop=True)
all_dates = sorted(df_full['dt_parsed'].unique())
date_to_day = {d: i+1 for i, d in enumerate(all_dates)}
df_full['day_num'] = df_full['dt_parsed'].map(date_to_day)
df_full['dow'] = df_full['dt_parsed'].dt.dayofweek
sales_orig = np.array(df_full['hours_sale'].tolist(), dtype=np.float64)[:, H_START:H_END]
stock_orig = np.array(df_full['hours_stock_status'].tolist(), dtype=np.int8)[:, H_START:H_END]
del df_train_hf, df_eval

series_list = []
for (sid,pid), grp in df_full.groupby(['store_id','product_id'], sort=False):
    gs = grp.sort_values('day_num')
    idx = gs.index.values
    series_list.append({'store_id':sid,'product_id':pid,'idx':idx,
                        'days':df_full.loc[idx,'day_num'].values,
                        'dows':df_full.loc[idx,'dow'].values})
print(f'  {len(series_list):,} serie')

def eval_instock(pred, obs, stk):
    instock = stk == 0
    ph, oh = pred[instock], obs[instock]
    sae_h, sao_h = np.abs(ph-oh).sum(), np.abs(oh).sum()
    se_h, so_h = (ph-oh).sum(), oh.sum()
    nd = pred.shape[0]
    sae_d, sao_d, se_d, so_d, nvd = 0., 0., 0., 0., 0
    for d in range(nd):
        m = instock[d]
        if m.any():
            pv, ov = pred[d,m].sum(), obs[d,m].sum()
            sae_d += abs(pv-ov); sao_d += abs(ov); se_d += pv-ov; so_d += ov; nvd += 1
    return {'sae_h':sae_h,'sao_h':sao_h,'se_h':se_h,'so_h':so_h,
            'sae_d':sae_d,'sao_d':sao_d,'se_d':se_d,'so_d':so_d,'n_in':int(instock.sum()),'n_vd':nvd}

def gm(sales,days,mx): m=days<=mx; return sales[m].mean(0) if m.any() else np.zeros(N_HOURS)
def dwm(sales,days,dows,mx):
    m=days<=mx; p={}
    for d in range(7):
        dm=m&(dows==d); p[d]=sales[dm].mean(0) if dm.any() else (sales[m].mean(0) if m.any() else np.zeros(N_HOURS))
    return p
def ma(sales,days,anc,K):
    a=days<=anc
    if not a.any(): return np.zeros(N_HOURS)
    return sales[a][-min(K,a.sum()):].mean(0)

FORECASTERS = ['Global Mean','DoW Mean',f'MA (K={MA_K})']
all_results = {}

for imp_key, imp_label in IMPUTERS.items():
    print(f'\n{"="*72}')
    print(f'  IMPUTER: {imp_label}')
    print(f'{"="*72}')

    df_cs = pd.read_parquet(os.path.join(COMPLETED_DIR, f'{imp_key}.parquet'))
    cs_sales = np.array(df_cs['hours_sale'].tolist(), dtype=np.float64)
    completed_full = sales_orig.copy()
    cs_keys = (df_cs['store_id'].astype(str)+'_'+df_cs['product_id'].astype(str)+'_'+df_cs['dt']).values
    full_keys = (df_full['store_id'].astype(str)+'_'+df_full['product_id'].astype(str)+'_'+df_full['dt']).values
    km = dict(zip(cs_keys, range(len(df_cs))))
    for i in range(len(df_full)):
        k = full_keys[i]
        if k in km: completed_full[i] = cs_sales[km[k]]
    del df_cs, cs_sales, km

    for fc in FORECASTERS:
        fc_safe_skip = fc.lower().replace(' ','_').replace('(','').replace(')','').replace('=','')
        ck_skip = f'{imp_key}__{fc_safe_skip}'
        if ck_skip in existing_cells:
            print(f'    SKIP {imp_label} × {fc}: {ck_skip} exists')
            continue
        pooled = {k:0. for k in ['sae_h','sao_h','se_h','so_h','sae_d','sao_d','se_d','so_d']}
        ps_recs = []
        for si, ser in enumerate(series_list):
            if (si+1)%10000==0: print(f'    ... {si+1:,}/{len(series_list):,}')
            idx=ser['idx']; days=ser['days']; dows=ser['dows']
            sales_cs=completed_full[idx]; obs_real=sales_orig[idx]; stock=stock_orig[idx]
            em=(days>=91)&(days<=97)
            if not em.any(): continue
            ne=em.sum(); obs=obs_real[em]; stk=stock[em]; ed=dows[em]
            if fc=='Global Mean': pred=np.tile(gm(sales_cs,days,90),(ne,1))
            elif fc=='DoW Mean':
                p=dwm(sales_cs,days,dows,90); pred=np.array([p[d] for d in ed])
            else: pred=np.tile(ma(sales_cs,days,90,MA_K),(ne,1))
            m=eval_instock(pred,obs,stk)
            for k2 in pooled: pooled[k2]+=m[k2]
            hw=m['sae_h']/m['sao_h'] if m['sao_h']>0 else np.nan
            hwp=m['se_h']/m['so_h'] if m['so_h']!=0 else np.nan
            dw=m['sae_d']/m['sao_d'] if m['sao_d']>0 else np.nan
            dwp=m['se_d']/m['so_d'] if m['so_d']!=0 else np.nan
            ps_recs.append({'store_id':ser['store_id'],'product_id':ser['product_id'],
                            'hourly_wape':hw,'hourly_wpe':hwp,'daily_wape':dw,'daily_wpe':dwp})
        p=pooled
        pm={'hourly_wape':p['sae_h']/p['sao_h'],'hourly_wpe':p['se_h']/p['so_h'],
            'daily_wape':p['sae_d']/p['sao_d'] if p['sao_d']>0 else np.nan,
            'daily_wpe':p['se_d']/p['so_d'] if p['so_d']!=0 else np.nan}
        ps=pd.DataFrame(ps_recs)
        mm={c:ps[c].dropna().median() for c in ['hourly_wape','hourly_wpe','daily_wape','daily_wpe']}
        fc_safe=fc.lower().replace(' ','_').replace('(','').replace(')','').replace('=','')
        ck=f'{imp_key}__{fc_safe}'
        ps.to_parquet(os.path.join(RESULTS_DIR,f'{ck}_test_per_series.parquet'),index=False)
        all_results[ck]={'imputer':imp_label,'forecaster':fc,'pooled':pm,'median':mm}
        print(f'    {imp_label} × {fc}: WAPE_h pool={pm["hourly_wape"]:.4f}, '
              f'med={mm["hourly_wape"]:.4f}, WPE_h={pm["hourly_wpe"]:.4f}')
    del completed_full

print('\n' + '=' * 72)
print('  RIEPILOGO')
print('=' * 72)
print(f'\n  {"Imputer":<24} {"Forecaster":<16} {"WAPE_h pool":>12} {"WPE_h pool":>11} {"WAPE_h med":>11}')
print('  ' + '-' * 78)
for ck, r in all_results.items():
    p,m=r['pooled'],r['median']
    print(f'  {r["imputer"]:<24} {r["forecaster"]:<16} {p["hourly_wape"]:>12.4f} {p["hourly_wpe"]:>11.4f} {m["hourly_wape"]:>11.4f}')

print('\n' + '=' * 72)
print('  DONE — 08b_naive_completed.py (ore 6-22)')
print('=' * 72)
