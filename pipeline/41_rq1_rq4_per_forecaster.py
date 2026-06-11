"""
41_rq1_rq4_per_forecaster.py — RQ1 (imputer causal effect per forecaster)
                                + RQ4 (per-forecaster equivalence sets)
=========================================================================
For each forecaster column:
  RQ1: paired Wilcoxon + Cliff's δ of every imputer vs no_imp baseline
       → answers "does imputation help over raw S_obs?"
  RQ4: equivalence set within the forecaster column
       (which imputers are statistically indistinguishable from best?)

Outputs:
  - pipeline/results/rq1_imputer_effect_per_forecaster.parquet
  - pipeline/results/rq4_equivalence_per_forecaster.parquet
  - pipeline/figures/fig_rq1_imputer_effect.png   (Cliff δ heatmap)
"""
import os, glob, functools, numpy as np, pandas as pd
from scipy import stats
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EQUIV_TH = 0.147

# ---------------------------------------------------------------------
# Load all cells (paired per-series WAPE)
# ---------------------------------------------------------------------
NON_HPO_FC = {'chronos_bolt','timesfm','global_mean','dow_mean','ma_k21'}

def parse_name(name):
    if '__' in name: return name.split('__', 1)
    return 'no_imp', name

per_series = {}
seen = set()
for f in sorted(glob.glob(f'{RESULTS_DIR}/*_hpo_test_per_series.parquet')):
    name = os.path.basename(f).replace('_hpo_test_per_series.parquet','')
    imp, fc = parse_name(name); per_series[(imp,fc)] = pd.read_parquet(f); seen.add(name)
for f in sorted(glob.glob(f'{RESULTS_DIR}/*_test_per_series.parquet')):
    fn = os.path.basename(f)
    if '_hpo_test_per_series' in fn: continue
    name = fn.replace('_test_per_series.parquet','')
    if name in seen: continue
    if name.startswith('naive_'):
        base = name.replace('naive_','',1)
        if base in NON_HPO_FC:
            imp, fc = 'no_imp', base
            if (imp, fc) in per_series: continue
            per_series[(imp,fc)] = pd.read_parquet(f); seen.add(f'no_imp__{base}')
            continue
    imp, fc = parse_name(name)
    if fc not in NON_HPO_FC: continue
    per_series[(imp,fc)] = pd.read_parquet(f); seen.add(name)

print(f'Loaded {len(per_series)} cells.')

# ---------------------------------------------------------------------
# RQ1: imputer effect vs no_imp baseline, per forecaster
# ---------------------------------------------------------------------
def cliff_delta_paired(a, b):
    a, b = np.asarray(a), np.asarray(b)
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]
    if len(a) == 0: return np.nan
    gt = (a < b).sum(); lt = (a > b).sum()
    return (gt - lt) / len(a)

forecasters = sorted({fc for (_, fc) in per_series.keys()})
print(f'Forecasters: {forecasters}')

rq1_rows = []
for fc in forecasters:
    # Baseline = no_imp for this forecaster (if exists)
    if ('no_imp', fc) not in per_series:
        print(f'  {fc}: no no_imp baseline → skip RQ1')
        continue
    base_df = per_series[('no_imp', fc)]
    base_idx = base_df.set_index(['store_id','product_id'])['hourly_wape']
    for (imp, fc2), df in per_series.items():
        if fc2 != fc: continue
        if imp == 'no_imp': continue
        other_idx = df.set_index(['store_id','product_id'])['hourly_wape']
        common = base_idx.index.intersection(other_idx.index)
        a = base_idx.loc[common].values  # no_imp
        b = other_idx.loc[common].values  # imputer treated
        valid = ~(np.isnan(a) | np.isnan(b))
        a, b = a[valid], b[valid]
        if len(a) == 0: continue
        # Cliff's δ: positive = a < b means imputer worsens (a=no_imp better)
        # We want positive = imputer improves (b < a). Define: δ = (a > b) - (a < b)
        gt_b_lower = (b < a).sum()
        lt_b_higher = (b > a).sum()
        delta = (gt_b_lower - lt_b_higher) / len(a)
        # Wilcoxon paired
        diff = a - b
        nz = diff != 0
        if nz.sum() == 0:
            p = np.nan
        else:
            try:
                res = stats.wilcoxon(a[nz], b[nz], alternative='two-sided')
                p = res.pvalue
            except Exception:
                p = np.nan
        rq1_rows.append({
            'forecaster': fc, 'imputer': imp,
            'wape_no_imp': float(np.nanmedian(a)),
            'wape_imputer': float(np.nanmedian(b)),
            'delta_med': float(np.nanmedian(a) - np.nanmedian(b)),
            'cliffs_delta_improvement': delta,
            'p_value': p,
            'n_paired': int(len(a)),
        })

rq1 = pd.DataFrame(rq1_rows)
rq1.to_parquet(f'{RESULTS_DIR}/rq1_imputer_effect_per_forecaster.parquet', index=False)
print(f'\nRQ1 written: {len(rq1)} (forecaster, imputer) pairs')

# Aggregate per forecaster
print('\nRQ1 summary per forecaster (imputer effect vs no_imp):')
print(f'{"forecaster":<15} {"n_imp":<6} {"mean δ":<8} {"%improves":<10} {"%negligible":<12}')
print('-' * 65)
for fc in forecasters:
    sub = rq1[rq1.forecaster == fc]
    if len(sub) == 0: continue
    mean_delta = sub.cliffs_delta_improvement.mean()
    pct_improve = (sub.cliffs_delta_improvement > 0).mean() * 100
    pct_negligible = (sub.cliffs_delta_improvement.abs() < EQUIV_TH).mean() * 100
    print(f'{fc:<15} {len(sub):<6} {mean_delta:<+8.3f} {pct_improve:<10.1f} {pct_negligible:<12.1f}')

# ---------------------------------------------------------------------
# RQ4: equivalence set per forecaster (best cell + others with |δ| < 0.147)
# ---------------------------------------------------------------------
print('\nRQ4: equivalence set per forecaster')
rq4_rows = []
for fc in forecasters:
    cells_fc = [(imp, df) for (imp, fc2), df in per_series.items() if fc2 == fc]
    # Best cell = lowest WAPE_h median
    cells_fc.sort(key=lambda x: x[1]['hourly_wape'].median())
    if len(cells_fc) == 0: continue
    best_imp, best_df = cells_fc[0]
    best_wape = best_df.hourly_wape.median()
    best_idx = best_df.set_index(['store_id','product_id'])['hourly_wape']
    n_equiv = 1  # best is trivially equivalent to itself
    equiv_list = [best_imp]
    for imp, df in cells_fc[1:]:
        other_idx = df.set_index(['store_id','product_id'])['hourly_wape']
        common = best_idx.index.intersection(other_idx.index)
        d = cliff_delta_paired(best_idx.loc[common].values, other_idx.loc[common].values)
        if abs(d) < EQUIV_TH:
            n_equiv += 1
            equiv_list.append(imp)
        rq4_rows.append({
            'forecaster': fc, 'best_imputer': best_imp, 'best_wape_med': float(best_wape),
            'other_imputer': imp, 'wape_med_other': float(df.hourly_wape.median()),
            'cliffs_delta': d, 'equivalent': abs(d) < EQUIV_TH,
        })
    print(f'  {fc}: best={best_imp} (WAPE={best_wape:.4f}), n_equiv={n_equiv}/{len(cells_fc)}')
    print(f'    equivalent imputers: {equiv_list}')

rq4 = pd.DataFrame(rq4_rows)
rq4.to_parquet(f'{RESULTS_DIR}/rq4_equivalence_per_forecaster.parquet', index=False)

# ---------------------------------------------------------------------
# Figure: Cliff δ heatmap (imputer × forecaster)
# Positive δ (green) = imputer improves over no_imp; negative δ (red) = worsens.
# ---------------------------------------------------------------------
print('\nGenerating RQ1 heatmap...')
imp_order = ['media_glob','media_cond','mediana_glob','mediana_cond',
             'forward_fill','seasonal_naive','linear_interp','lgb',
             'dlinear','saits','itransformer','timesnet','imputeformer']
fc_order = ['lgb_m5lags','mlp_m5lags','tft','chronos_bolt','global_mean','dow_mean','ma_k21']
fc_short = {'lgb_m5lags':'LGB_M5','mlp_m5lags':'MLP_M5','tft':'TFT','chronos_bolt':'Chronos',
            'global_mean':'GM','dow_mean':'DoW','ma_k21':'MA21'}

# Pivot rq1
piv = rq1.pivot(index='imputer', columns='forecaster', values='cliffs_delta_improvement')
piv = piv.reindex(index=imp_order, columns=fc_order)

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(piv.values, cmap='RdYlGn', aspect='auto', vmin=-0.5, vmax=0.5)
for i, imp in enumerate(imp_order):
    for j, fc in enumerate(fc_order):
        v = piv.iloc[i, j]
        if pd.isna(v): continue
        text = f'{v:+.2f}'
        color = 'white' if abs(v) > 0.3 else 'black'
        # Bold if non-negligible
        weight = 'bold' if abs(v) >= EQUIV_TH else 'normal'
        ax.text(j, i, text, ha='center', va='center', fontsize=11,
                color=color, fontweight=weight)
        if abs(v) < EQUIV_TH:
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                       edgecolor='gray', lw=0.8, linestyle=':'))

ax.set_xticks(range(len(fc_order)))
ax.set_xticklabels([fc_short[f] for f in fc_order], fontsize=12)
ax.set_yticks(range(len(imp_order)))
ax.set_yticklabels(imp_order, fontsize=11)
ax.set_title('RQ1 — Cliff\'s δ of imputer treatment vs no_imp baseline\n'
             '(+ = imputer improves; − = worsens; gray dotted = negligible |δ|<0.147)',
             fontsize=13, pad=14)
ax.set_xlabel('Forecaster', fontsize=12)
ax.set_ylabel('Imputer (treatment)', fontsize=12)
plt.colorbar(im, ax=ax, label='Cliff\'s δ (improvement vs no_imp)')

plt.tight_layout()
out_fig = f'{FIG_DIR}/fig_rq1_imputer_effect.png'
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
print(f'Saved: {out_fig}')

print('\nDONE')
