"""
referee2_holm_seedband.py — Second referee report, Major #6 + minor (seed band in Fig. 6).

(A) Pairwise Wilcoxon signed-rank + Holm correction, replacing Nemenyi where the
    referee asks (Benavoli et al. 2016: a Nemenyi outcome between two cells depends on
    the other k-2 cells; a paired Wilcoxon does not).
      A1. RQ3: Friedman-best cell vs every other matrix cell (global, same series as script 45)
      A2. RQ3 per volume quartile: best cell of each quartile vs every other cell
      A3. Section 4.5: quantile-calibrated direct forecaster (no imputer, tau grid)
          vs the best two-stage cell of the same family and vs the Friedman-best cell
    Effect size: matched-pairs rank-biserial r (Kerby 2014) and Cliff's delta (Romano bands),
    so the decision does not rest on p-values alone at N ~ 50K.

(B) Seed band: 20-seed range of the median hourly WAPE for LGB-M5 and MLP-M5
    (iTransformer cell); flags the imputer cells of each lag-based column whose median
    WAPE lies within one seed range of that column's best cell.

Outputs (results/):
  referee2_wilcoxon_holm_rq3.parquet, referee2_wilcoxon_holm_rq3_quartile.parquet,
  referee2_wilcoxon_holm_tau.parquet, referee2_seedband.parquet
"""
import os, glob, functools
import numpy as np, pandas as pd
from scipy import stats
print = functools.partial(print, flush=True)

RES = os.path.join(os.path.dirname(__file__), 'results')
NON_HPO_FC = {'chronos_bolt', 'timesfm', 'global_mean', 'dow_mean', 'ma_k56', 'croston', 'sba', 'tsb'}
KEY = ['store_id', 'product_id']
ALPHA = 0.05


def parse_name(n):
    return n.split('__', 1) if '__' in n else ('no_imp', n)


def load_matrix_cells():
    """Same cell-loading rule as 45_friedman_nemenyi.py (kept verbatim for identical k and N)."""
    per_series, seen = {}, set()
    for f in sorted(glob.glob(f'{RES}/*_hpo_test_per_series.parquet')):
        if any(x in os.path.basename(f) for x in ('censored', '_hicap')): continue
        name = os.path.basename(f).replace('_hpo_test_per_series.parquet', '')
        per_series[name] = pd.read_parquet(f); seen.add(name)
    for f in sorted(glob.glob(f'{RES}/*_test_per_series.parquet')):
        fn = os.path.basename(f)
        if any(x in fn for x in ('censored', '_hicap')) or '_hpo_test_per_series' in fn: continue
        name = fn.replace('_test_per_series.parquet', '')
        if name in seen: continue
        if name.startswith('naive_'):
            base = name.replace('naive_', '', 1)
            if base in NON_HPO_FC and f'no_imp__{base}' not in seen:
                per_series[f'no_imp__{base}'] = pd.read_parquet(f); seen.add(f'no_imp__{base}')
            continue
        imp, fc = parse_name(name)
        if fc not in NON_HPO_FC: continue
        per_series[name] = pd.read_parquet(f); seen.add(name)
    return per_series


def wide(per_series, col='hourly_wape'):
    names = sorted(per_series)
    idx = None
    for n in names:
        i = per_series[n].set_index(KEY).index
        idx = i if idx is None else idx.intersection(i)
    W = pd.DataFrame({n: per_series[n].set_index(KEY)[col].reindex(idx).values for n in names}, index=idx)
    return W.dropna()


def cliffs_delta_paired(a, b):
    """Cliff's delta on paired per-series WAPE, same convention as the paper:
    share of series where b is lower (better) minus share where a is lower."""
    d = a - b
    return float((d > 0).mean() - (d < 0).mean())


def romano(d):
    d = abs(d)
    return 'negligible' if d < 0.147 else 'small' if d < 0.33 else 'medium' if d < 0.474 else 'large'


def wilcoxon_pair(a, b):
    """a = reference cell, b = challenger. Returns p, rank-biserial r (>0 => b better), delta."""
    d = a - b
    nz = d[d != 0]
    if len(nz) == 0:
        return 1.0, 0.0, 0.0
    p = stats.wilcoxon(nz, zero_method='wilcox', alternative='two-sided', method='approx').pvalue
    rk = stats.rankdata(np.abs(nz))
    r = (rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum()
    return float(p), float(r), cliffs_delta_paired(a, b)


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for rank, i in enumerate(o):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def best_vs_all(W, best, tag):
    rows = []
    for c in W.columns:
        if c == best: continue
        p, r, dl = wilcoxon_pair(W[best].values, W[c].values)
        rows.append(dict(group=tag, reference=best, cell=c, n=len(W),
                         med_ref=W[best].median(), med_cell=W[c].median(),
                         p_raw=p, r_rb=r, cliffs_delta=dl, romano=romano(dl)))
    out = pd.DataFrame(rows)
    out['p_holm'] = holm(out['p_raw'])
    out['holm_indistinguishable'] = out['p_holm'] >= ALPHA
    return out.sort_values('p_raw', ascending=False)


# ======================================================================
# A1 / A2 — RQ3 equivalence set
# ======================================================================
print('A. Loading matrix cells (same rule as script 45)...')
per_series = load_matrix_cells()
fr = pd.read_parquet(f'{RES}/friedman_nemenyi_ranks.parquet')
per_series = {n: d for n, d in per_series.items() if n in set(fr['cell'])}  # drop auxiliary outputs (e.g. safehours_*)
assert len(per_series) == len(fr), (len(per_series), len(fr))
W = wide(per_series)
print(f'   k={W.shape[1]} cells, N={len(W):,} complete-case series')

best = fr.iloc[0]['cell']
nem_set = set(fr[fr.cd_indistinguishable]['cell'])
print(f'   Friedman best: {best} | Nemenyi set: {sorted(nem_set)}')

a1 = best_vs_all(W, best, 'global')
a1['nemenyi_indistinguishable'] = a1['cell'].isin(nem_set)
a1.to_parquet(f'{RES}/referee2_wilcoxon_holm_rq3.parquet', index=False)
print(f'\nA1. Global, {len(a1)} comparisons, Holm at alpha={ALPHA}')
print(f'    Holm-indistinguishable from best: {a1.holm_indistinguishable.sum()} cells')
print(f'    Negligible |delta| (<0.147): {(a1.romano == "negligible").sum()} cells')
print(a1.head(12)[['cell', 'med_cell', 'p_raw', 'p_holm', 'r_rb', 'cliffs_delta', 'romano',
                   'nemenyi_indistinguishable']].to_string(index=False))

print('\nA2. Per volume quartile...')
qmap = pd.read_parquet(f'{RES}/series_quartile_op.parquet').set_index(KEY)['quartile']
frq = pd.read_parquet(f'{RES}/friedman_nemenyi_ranks_per_quartile.parquet')
qcol = 'quartile' if 'quartile' in frq.columns else frq.columns[0]
a2 = []
Wq_all = W.join(qmap.rename('_q'), how='inner')
for q in sorted(Wq_all['_q'].dropna().unique()):
    Wq = Wq_all[Wq_all['_q'] == q].drop(columns='_q')
    sub = frq[frq[qcol] == q].sort_values('mean_rank')
    qbest = sub.iloc[0]['cell']
    res = best_vs_all(Wq, qbest, str(q))
    res['nemenyi_indistinguishable'] = res['cell'].isin(set(sub[sub.cd_indistinguishable]['cell']))
    a2.append(res)
    print(f'    {q}: best={qbest} N={len(Wq):,} | Nemenyi set={int(res.nemenyi_indistinguishable.sum())} '
          f'| Holm-indist={int(res.holm_indistinguishable.sum())} | negligible delta={(res.romano == "negligible").sum()}')
a2 = pd.concat(a2, ignore_index=True)
a2.to_parquet(f'{RES}/referee2_wilcoxon_holm_rq3_quartile.parquet', index=False)

# ======================================================================
# A3 — Section 4.5: tau-direct forecaster vs two-stage cells
# ======================================================================
print('\nA3. Quantile-calibrated direct forecaster vs two-stage (Wilcoxon + Holm)...')
REFS = {'lgb': ['mediana_cond__lgb_m5lags', best],
        'mlp': ['timesnet__mlp_m5lags', best]}
rows = []
for fam, refs in REFS.items():
    for tau in ['0.5', '0.6', '0.7', '0.8', '0.9']:
        f = f'{RES}/censored_{fam}_m5_q{tau}_hpo_test_per_series.parquet'
        if not os.path.exists(f):
            print(f'    missing {f}'); continue
        ch = pd.read_parquet(f).set_index(KEY)['hourly_wape']
        for ref in refs:
            rs = per_series[ref].set_index(KEY)['hourly_wape']
            j = pd.concat([rs.rename('ref'), ch.rename('ch')], axis=1, join='inner').dropna()
            p, r, dl = wilcoxon_pair(j['ref'].values, j['ch'].values)
            rows.append(dict(family=fam, tau=float(tau), reference=ref, n=len(j),
                             med_ref=j['ref'].median(), med_direct=j['ch'].median(),
                             p_raw=p, r_rb=r, cliffs_delta=dl, romano=romano(dl),
                             share_direct_better=float((j['ch'] < j['ref']).mean())))
a3 = pd.DataFrame(rows)
a3['p_holm'] = holm(a3['p_raw'])
a3['holm_indistinguishable'] = a3['p_holm'] >= ALPHA
a3.to_parquet(f'{RES}/referee2_wilcoxon_holm_tau.parquet', index=False)
print(a3[['family', 'tau', 'reference', 'med_ref', 'med_direct', 'p_holm', 'r_rb', 'cliffs_delta',
          'romano', 'share_direct_better']].to_string(index=False))

# ======================================================================
# B — Seed band for Fig. 6
# ======================================================================
print('\nB. Seed band (20 seeds, iTransformer cell)...')
mat = pd.read_parquet(f'{RES}/hpo_matrix_pareto.parquet')
rows = []
for fam in ['lgb', 'mlp']:
    meds = np.array([pd.read_parquet(f)['hourly_wape'].dropna().median() for f in
                     sorted(glob.glob(f'{RES}/seedrun_itransformer__{fam}_m5lags_hpo_seed*_test_per_series.parquet'))])
    band = meds.max() - meds.min()
    fc = f'{fam}_m5lags'
    col = mat[mat.forecaster == fc][['imputer', 'wape_h_med']].sort_values('wape_h_med')
    cbest = col['wape_h_med'].min()
    print(f'    {fc}: {len(meds)} seeds, median WAPE {meds.min():.4f}-{meds.max():.4f} '
          f'(range {band:.4f}, sd {meds.std(ddof=1):.4f}) | column best {cbest:.4f}, '
          f'column spread {col.wape_h_med.max() - cbest:.4f}')
    for _, r in col.iterrows():
        rows.append(dict(forecaster=fc, imputer=r.imputer, wape_h_med=r.wape_h_med,
                         gap_to_col_best=r.wape_h_med - cbest, seed_range=band,
                         seed_sd=meds.std(ddof=1), n_seeds=len(meds),
                         within_seed_band=(r.wape_h_med - cbest) <= band))
sb = pd.DataFrame(rows)
sb.to_parquet(f'{RES}/referee2_seedband.parquet', index=False)
print(sb.groupby('forecaster').agg(n=('imputer', 'size'), within=('within_seed_band', 'sum')).to_string())
print(sb[~sb.within_seed_band].to_string(index=False))

# cross-family: cells of any forecaster within MLP seed band of the global best median
gbest = mat['wape_h_med'].min()
mlp_band = sb[sb.forecaster == 'mlp_m5lags']['seed_range'].iloc[0]
near = mat[mat['wape_h_med'] - gbest <= mlp_band].sort_values('wape_h_med')
print(f'\n    Cells within the MLP-M5 seed range ({mlp_band:.4f}) of the lowest median WAPE ({gbest:.4f}): {len(near)}')
print(near.groupby('forecaster').size().to_string())
