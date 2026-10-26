"""
referee3_mask.py — internal review (Oct 2026), variant (ii): pinball loss MASKED on stock-out hours.

The lag-based forecasters are trained on observed sales, stock-out zeros included, so part of their
under-forecast may come from the censored target rather than from the loss. Variant (ii) gives the
stock-out hours zero weight in the loss (MASK_STOCKOUT=1, RUN_TAG=_mask; run_overnight_target_mask.sh):
  A. no imputation, tau in {0.5,...,0.8}: hourly WAPE/WPE, daily WAPE, validation WAPE, against the
     unmasked single-run cells of Table 6 (RUN_TAG=_nvsel); tau* on the validation criterion (written to
     referee3_mask_tau_star.json for the launcher); newsvendor cost with tau selected on validation cost.
  B. imputer sensitivity under the mask (no imputation, Median global, iTransformer, Linear interp.) at
     tau = 0.5 and tau*: paired delta vs no imputation (seed band 0.11 LGB / 0.13 MLP) and Kendall's W,
     next to the same four imputers unmasked (tau grid of Section 4.6).
Works on partial runs. Outputs: referee3_mask_A.parquet, referee3_mask_B.parquet, referee3_mask_tau_star.json
"""
import os, json, functools
import numpy as np, pandas as pd
from scipy import stats
print = functools.partial(print, flush=True)
RES = os.path.join(os.path.dirname(__file__), 'results'); PRED = os.path.join(RES, 'preds')
K = ['store_id', 'product_id']; RATIOS = [1.0, 2.0, 5.0]; TAUS = ['0.5', '0.6', '0.7', '0.8']
SM = '_SMOKE' if os.getenv('SMOKE') == '1' else ''
SEED = {'lgb': 0.11, 'mlp': 0.13}
IMPS = ['no_imp', 'mediana_glob', 'itransformer', 'linear_interp']
REF = {'test': pd.read_parquet(f'{RES}/newsvendor_yref.parquet'), 'val': pd.read_parquet(f'{RES}/newsvendor_yref_val.parquet')}


def name(fam, tau, imp, tag):
    return f'censored_{fam}_m5_q{tau}_hpo_imp-{imp}{tag}'


def per_series(c, split='test'):
    f = f'{RES}/{c}_{"test" if split == "test" else "VAL"}_per_series.parquet'
    return pd.read_parquet(f) if os.path.exists(f) else None


def cost(npz, split):
    if not os.path.exists(npz):
        return None
    z = np.load(npz); n = len(z['store_id'])
    q = pd.DataFrame({'store_id': z['store_id'], 'product_id': z['product_id'],
                      'day_idx': np.tile(np.arange(7), n // 7), 'q': z['preds'].sum(1).astype(float)})
    m = q.merge(REF[split], on=['store_id', 'product_id', 'day_idx'], how='inner')
    return {r: float((np.clip(m.q - m.y_star, 0, None) + r * np.clip(m.y_star - m.q, 0, None))
                     .groupby([m.store_id, m.product_id]).sum().median()) for r in RATIOS}


# ---------------- A ----------------
rowsA, tau_star = [], {}
for fam in ['lgb', 'mlp']:
    for tag, label in [('_nvsel', 'unmasked'), ('_mask' + SM, 'masked')]:
        for tau in TAUS:
            c = name(fam, tau, 'no_imp', tag)
            t, v = per_series(c), per_series(c, 'val')
            if t is None or v is None:
                continue
            vc, tc = cost(f'{PRED}/{c}_VAL.npz', 'val'), cost(f'{PRED}/{c}.npz', 'test')
            row = dict(family=f'{fam.upper()}-M5', loss=label, tau=float(tau), wape_h=t.hourly_wape.median(),
                       wpe_h=t.hourly_wpe.median(), wape_d=t.daily_wape.median(),
                       val_wape=v[v.n_instock >= 34].val_hourly_wape.median())
            for r in RATIOS:
                row[f'val_cost_r{r:g}'] = vc[r] if vc else np.nan; row[f'test_cost_r{r:g}'] = tc[r] if tc else np.nan
            rowsA.append(row)
A = pd.DataFrame(rowsA)
if len(A):
    A.to_parquet(f'{RES}/referee3_mask_A{SM}.parquet', index=False)
    print('A. no imputation: masked vs unmasked pinball loss')
    for (fam, loss), g in A.groupby(['family', 'loss'], sort=False):
        g = g.sort_values('tau')
        print(f'  {fam} {loss:9s}: ' + ' | '.join(f"tau={x.tau:.1f} WAPE {x.wape_h:.3f} WPE {x.wpe_h:+.2f} dWAPE {x.wape_d:.3f}" for x in g.itertuples()))
        ts = g.loc[g.val_wape.idxmin(), 'tau']; print(f'      tau* on validation WAPE = {ts:.1f}')
        if loss == 'masked':
            tau_star[fam] = f'{ts:.1f}'
        for r in RATIOS:
            gg = g.dropna(subset=[f'val_cost_r{r:g}'])
            if len(gg):
                b = gg.loc[gg[f'val_cost_r{r:g}'].idxmin()]
                print(f'      r={r:g}: tau* on validation cost = {b.tau:.1f}, test cost {b[f"test_cost_r{r:g}"]:.2f}')
    json.dump(tau_star, open(f'{RES}/referee3_mask_tau_star{SM}.json', 'w'))

# ---------------- B ----------------
rowsB = []
taus_B = sorted({'0.5'} | set(tau_star.values()))
for fam in ['lgb', 'mlp']:
    for tau in taus_B:
        for tag, label in [('_mask' + SM, 'masked'), ('', 'unmasked (Sec. 4.6 grid)')]:
            cells = {}
            for imp in IMPS:
                if label == 'masked':
                    c = name(fam, tau, imp, tag)
                else:
                    c = name(fam, tau, imp, '') if imp != 'no_imp' else name(fam, tau, 'no_imp', '')
                d = per_series(c)
                if d is not None:
                    cells[imp] = d.set_index(K)['hourly_wape']
            if 'no_imp' not in cells or len(cells) < 2:
                continue
            H = pd.DataFrame(cells).dropna()
            dl = {i: float(((H['no_imp'] - H[i]) > 0).mean() - ((H['no_imp'] - H[i]) < 0).mean()) for i in H if i != 'no_imp'}
            W = np.nan
            if H.shape[1] >= 3:
                chi2, _ = stats.friedmanchisquare(*[H[c].values for c in H]); W = chi2 / (len(H) * (H.shape[1] - 1))
            rowsB.append(dict(family=f'{fam.upper()}-M5', tau=float(tau), loss=label, k=H.shape[1], W=W,
                              medians=str(H.median().round(3).to_dict()), delta_vs_noimp=str({k: round(v, 2) for k, v in dl.items()}),
                              n_beyond_seed=sum(abs(v) > SEED[fam] for v in dl.values())))
            print(f"B. {fam.upper()}-M5 tau={tau} {label:26s} k={H.shape[1]} W={W:.3f} | delta vs no-imp {rowsB[-1]['delta_vs_noimp']} (seed band {SEED[fam]})")
if rowsB:
    pd.DataFrame(rowsB).to_parquet(f'{RES}/referee3_mask_B{SM}.parquet', index=False)
