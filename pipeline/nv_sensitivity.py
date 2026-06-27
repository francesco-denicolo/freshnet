"""Round-2 minor revisions for the newsvendor analysis.

(1) Reference sensitivity: rebuild the reference completed daily demand y* with
    three cell-independent reference imputers computable on the test horizon
    - R1 conditional median over (store, product, hour)   [the one in the paper]
    - R2 conditional mean   over (store, product, hour)
    - R3 conditional median over (store, product, dow, hour)
    and show the headline conclusions (within-family cost spread, rho(cost,WAPE),
    rho(cost vs y*, cost vs in-stock), forecaster separation) are invariant.

(3) Bootstrap CIs (series-level resampling) for the headline numbers under R1.

Reuses the already-exported daily orders q(s,d) for the 26 lag-based cells; no
retraining. Writes newsvendor_sensitivity.txt.
"""
import os, glob, functools, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr
print = functools.partial(print, flush=True)
t0 = time.time()
PR = os.path.join(os.path.dirname(__file__), '..')
DATA = os.path.join(PR, 'data'); RES = os.path.join(os.path.dirname(__file__), 'results')
H0, H1 = 6, 23; NH = H1 - H0
IMPS = ['dlinear','forward_fill','imputeformer','itransformer','lgb','linear_interp',
        'media_cond','media_glob','mediana_cond','mediana_glob','saits','seasonal_naive','timesnet']
out_lines = []
def emit(s): print(s); out_lines.append(s)

def load(fn):
    d = pd.read_parquet(os.path.join(DATA, fn)); d['dt_parsed'] = pd.to_datetime(d['dt'])
    return d.sort_values(['store_id','product_id','dt_parsed']).reset_index(drop=True)

print('Loading train/eval...')
tr = load('frn50k_train.parquet'); ev = load('frn50k_eval.parquet')
key = tr[['store_id','product_id']].drop_duplicates().reset_index(drop=True)
S = len(key); ND_TR = len(tr)//S; ND_EV = len(ev)//S
keyidx = {(s,p): i for i, (s, p) in enumerate(zip(key.store_id.values, key.product_id.values))}
_cm = tr.drop_duplicates(['store_id','product_id'])[['store_id','product_id','city_id']]
city = key.merge(_cm, on=['store_id','product_id'])['city_id'].values        # aligned to key order
_cities = np.unique(city); city_rows = {c: np.where(city==c)[0] for c in _cities}
print(f'{len(_cities)} cities; largest = {max(len(v) for v in city_rows.values())/len(city)*100:.1f}% of series')

def cube(d, nd):
    sa = np.array(d['hours_sale'].tolist(), dtype=np.float32)[:, H0:H1].reshape(S, nd, NH)
    st = np.array(d['hours_stock_status'].tolist(), dtype=np.float32)[:, H0:H1].reshape(S, nd, NH)
    dow = d['dt_parsed'].dt.dayofweek.values.reshape(S, nd)
    return sa, st, dow

tr_sa, tr_st, tr_dow = cube(tr, ND_TR)
ev_sa, ev_st, ev_dow = cube(ev, ND_EV)
masked = np.where(tr_st == 0, tr_sa, np.nan)             # in-stock train sales

print('Building 3 references...')
with np.errstate(all='ignore'):
    med_sph = np.nanmedian(masked, axis=1)              # R1 (S,17)
    mean_sph = np.nanmean(masked, axis=1)               # R2 (S,17)
    med_spdowh = np.full((S, 7, NH), np.nan, np.float32) # R3 (S,7,17)
    for dw in range(7):
        mk = np.where(tr_dow == dw, 1.0, np.nan)[:, :, None]
        med_spdowh[:, dw, :] = np.nanmedian(masked * mk, axis=1)
glob_h = np.where(np.isnan(np.nanmedian(masked.reshape(-1, NH), axis=0)), 0.0,
                  np.nanmedian(masked.reshape(-1, NH), axis=0))
def fillnan(a): a[np.isnan(a)] = np.broadcast_to(glob_h, a.shape)[np.isnan(a)]; return a
med_sph = fillnan(med_sph); mean_sph = fillnan(mean_sph)
for dw in range(7): med_spdowh[:, dw, :] = fillnan(med_spdowh[:, dw, :])

def ystar(ref, dow_dep=False):
    if dow_dep:                                          # ref is (S,7,17): index dow per test day
        out = np.empty((S, ND_EV, NH), np.float32)
        for j in range(ND_EV):
            out[:, j, :] = ref[np.arange(S), ev_dow[:, j].astype(int), :]
        comp = np.where(ev_st == 0, ev_sa, out)
    else:                                                # ref is (S,17)
        comp = np.where(ev_st == 0, ev_sa, ref[:, None, :])
    return comp.sum(2)                                   # (S,7)

YS = {'R1_med_sph': ystar(med_sph), 'R2_mean_sph': ystar(mean_sph),
      'R3_med_spdowh': ystar(med_spdowh, dow_dep=True)}
y_obs = np.where(ev_st == 0, ev_sa, 0.0).sum(2)          # (S,7) in-stock-only

# ---- load q for the 26 cells, aligned to (S,7) ----
print('Loading 26 q matrices...')
def align_q(cell):
    f = f'{RES}/newsvendor_q_{cell}.parquet'
    q = pd.read_parquet(f)
    arr = np.full((S, ND_EV), np.nan, np.float64)
    rows = np.array([keyidx[(s, p)] for s, p in zip(q.store_id.values, q.product_id.values)])
    arr[rows, q.day_idx.values.astype(int)] = q['q'].values
    return arr

def wape_series(cell):
    ps = pd.read_parquet(f'{RES}/{cell}_test_per_series.parquet')
    v = np.full(S, np.nan)
    rows = np.array([keyidx[(s, p)] for s, p in zip(ps.store_id.values, ps.product_id.values)])
    v[rows] = ps['hourly_wape'].values
    return v

FCS = ['lgb_m5lags', 'mlp_m5lags']
cells = [f'{imp}__{fc}_hpo' for fc in FCS for imp in IMPS]
Q = {c: align_q(c) for c in cells}
W = {c: wape_series(c) for c in cells}

def cost_series(qarr, ystar_arr, r=2.0):
    c = np.clip(qarr - ystar_arr, 0, None) * 1.0 + np.clip(ystar_arr - qarr, 0, None) * r
    return np.nansum(c, axis=1)                          # per-series total (sum over 7 days)

# ---- per reference: headline stats ----
emit('='*70); emit('(1) REFERENCE SENSITIVITY  (median per-series newsvendor cost, r=2)')
percell_cost = {}   # ref -> {cell: per-series cost array}
for ref, ya in YS.items():
    percell_cost[ref] = {c: cost_series(Q[c], ya) for c in cells}
# cost vs in-stock-only
cost_obs = {c: cost_series(Q[c], y_obs) for c in cells}

def med(x): return float(np.nanmedian(x))
def fam(fc): return [f'{imp}__{fc}_hpo' for imp in IMPS]

summary = {}
for ref in YS:
    row = {}
    for fc in FCS:
        cc = fam(fc)
        cost_med = np.array([med(percell_cost[ref][c]) for c in cc])
        wape_med = np.array([med(W[c]) for c in cc])
        row[fc] = dict(spread=cost_med.max()-cost_med.min(),
                       spread_pct=100*(cost_med.max()-cost_med.min())/cost_med.min(),
                       rho_cw=spearmanr(cost_med, wape_med).correlation)
    # rho(cost vs y*, cost vs in-stock) over all 26 cells
    cy = np.array([med(percell_cost[ref][c]) for c in cells])
    co = np.array([med(cost_obs[c]) for c in cells])
    row['rho_ystar_obs'] = spearmanr(cy, co).correlation
    summary[ref] = row
    emit(f'\n  [{ref}]')
    for fc in FCS:
        s = row[fc]; emit(f'    {fc:11s}: spread={s["spread"]:.4f} ({s["spread_pct"]:.1f}%)  rho(cost,WAPE)={s["rho_cw"]:+.3f}')
    emit(f'    rho(cost vs y*, cost vs in-stock demand) over 26 cells = {row["rho_ystar_obs"]:.3f}')

# cross-reference stability of the cell cost ordering
emit('\n  Cross-reference rank correlation of the 26-cell cost ordering:')
refs = list(YS)
ordv = {ref: np.array([med(percell_cost[ref][c]) for c in cells]) for ref in refs}
for i in range(len(refs)):
    for j in range(i+1, len(refs)):
        emit(f'    {refs[i]} vs {refs[j]}: Spearman = {spearmanr(ordv[refs[i]], ordv[refs[j]]).correlation:.4f}')

# ---- (3) bootstrap CIs (series-level), main reference R1 ----
emit('\n' + '='*70); emit('(3) BOOTSTRAP 95% CIs  (series resampling, B=500, reference R1)')
ref = 'R1_med_sph'; B = 500; rng = np.random.default_rng(0)
# pre-stack per-series matrices (S x ncell) for speed
cost_mat = np.column_stack([percell_cost[ref][c] for c in cells])     # (S,26)
costobs_mat = np.column_stack([cost_obs[c] for c in cells])
wape_mat = np.column_stack([W[c] for c in cells])
idx_mlp = [cells.index(c) for c in fam('mlp_m5lags')]
idx_lgb = [cells.index(c) for c in fam('lgb_m5lags')]

def stats_from(rowsel):
    cm = np.nanmedian(cost_mat[rowsel], axis=0)
    com = np.nanmedian(costobs_mat[rowsel], axis=0)
    wm = np.nanmedian(wape_mat[rowsel], axis=0)
    out = {}
    out['rho_ystar_obs'] = spearmanr(cm, com).correlation
    for nm, idx in [('mlp', idx_mlp), ('lgb', idx_lgb)]:
        out[f'rho_cw_{nm}'] = spearmanr(cm[idx], wm[idx]).correlation
        sp = cm[idx]; out[f'spread_pct_{nm}'] = 100*(sp.max()-sp.min())/sp.min()
    return out

point = stats_from(np.arange(S))
boot = {k: [] for k in point}
for b in range(B):
    sel = rng.integers(0, S, S)
    st = stats_from(sel)
    for k in st: boot[k].append(st[k])
def ci(k): a = np.array(boot[k]); return np.nanpercentile(a, 2.5), np.nanpercentile(a, 97.5)

# city block-bootstrap (R1.1): resample whole cities to respect within-city dependence
clist = list(_cities); boot_c = {k: [] for k in point}
for b in range(B):
    samp = rng.choice(len(clist), len(clist), replace=True)
    sel = np.concatenate([city_rows[clist[i]] for i in samp])
    st = stats_from(sel)
    for k in st: boot_c[k].append(st[k])
def ci_c(k): a = np.array(boot_c[k]); return np.nanpercentile(a, 2.5), np.nanpercentile(a, 97.5)
emit('  --- city block-bootstrap (resampling the 18 cities) ---')
for k in ['rho_ystar_obs','rho_cw_mlp','rho_cw_lgb']:
    lo,hi=ci_c(k); emit(f'    {k:18s}: {point[k]:+.3f}  city-block 95% CI [{lo:+.3f}, {hi:+.3f}]')
labels = {'rho_ystar_obs':'rho(cost vs y*, cost vs in-stock) [26 cells]',
          'rho_cw_mlp':'within-MLP-M5 rho(cost,WAPE) [13 imp]',
          'rho_cw_lgb':'within-LGB-M5 rho(cost,WAPE) [13 imp]',
          'spread_pct_mlp':'within-MLP-M5 cost spread %',
          'spread_pct_lgb':'within-LGB-M5 cost spread %'}
for k in ['rho_ystar_obs','rho_cw_mlp','rho_cw_lgb','spread_pct_mlp','spread_pct_lgb']:
    lo, hi = ci(k); emit(f'  {labels[k]:46s}: {point[k]:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]')

with open(f'{RES}/newsvendor_sensitivity.txt', 'w') as f:
    f.write('\n'.join(out_lines) + '\n')
emit(f'\nsaved {RES}/newsvendor_sensitivity.txt  | time={time.time()-t0:.0f}s')
