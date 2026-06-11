"""
44_rq1_tost_stratified.py — RQ1 with TOST + per-quartile stratification
========================================================================
Extends script 41 (RQ1 = imputer effect vs no_imp baseline per forecaster):
  - Adds bootstrap-based TOST decision per (imputer, forecaster) cell
  - Stratifies by training-volume quartile (Q1-Q4)

Output:
  - pipeline/results/rq1_tost_stratified.parquet (all comparisons)
  - pipeline/figures/fig_rq1_tost_global.png   (1 heatmap, Cliff δ vs no_imp globally)
  - pipeline/figures/fig_rq1_tost_4grid.png    (4 sub-heatmaps Q1-Q4)

Decisione TOST:
  H0: |Cliff δ| ≥ 0.147 (imputer effect is non-negligible)
  H1: |Cliff δ| < 0.147 (no-op / equivalent to no_imp)
  Reject H0 ⇒ EQUIVALENT (imputer è inutile)
  Otherwise: IMPROVES_SIG (δ̂ > +0.147 e CI escluso negativo)
             WORSENS_SIG (δ̂ < -0.147 e CI escluso positivo)
             INCONCLUSIVE (CI attraversa il margine)
"""
import os, glob, functools, time, numpy as np, pandas as pd
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MARGIN = 0.147
N_BOOT = 300
SEED = 42
np.random.seed(SEED)

NON_HPO_FC = {'chronos_bolt','timesfm','global_mean','dow_mean','ma_k21'}

def parse_name(n):
    return n.split('__', 1) if '__' in n else ('no_imp', n)

# ----------------------------------------------------------------------
# 1. Load cells
# ----------------------------------------------------------------------
print('1. Loading cells...')
per_series = {}; seen = set()
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
            per_series[('no_imp', base)] = pd.read_parquet(f); seen.add(f'no_imp__{base}')
            continue
    imp, fc = parse_name(name)
    if fc not in NON_HPO_FC: continue
    per_series[(imp, fc)] = pd.read_parquet(f); seen.add(name)
print(f'   {len(per_series)} cells')

# Quartile map
print('2. Computing volume quartiles...')
df_tr = pd.read_parquet(os.path.join(DATA_DIR,'frn50k_train.parquet'))
df_tr['dt_parsed'] = pd.to_datetime(df_tr['dt'])
df_tr = df_tr.sort_values(['store_id','product_id','dt_parsed']).reset_index(drop=True)
df_tr['day_num'] = df_tr.groupby(['store_id','product_id']).cumcount() + 1
vol = df_tr[df_tr.day_num<=83].groupby(['store_id','product_id'])['sale_amount'].sum().reset_index()
vol.columns = ['store_id','product_id','volume']
vol['quartile'] = pd.qcut(vol['volume'], q=4, labels=['Q1','Q2','Q3','Q4']).astype(str)
quart_map = vol.set_index(['store_id','product_id'])['quartile']
del df_tr

def cliffs_delta(a, b):
    """δ > 0 ⇔ b < a (imputer migliora)"""
    if len(a) == 0: return np.nan
    return ((b < a).sum() - (b > a).sum()) / len(a)

def bootstrap_cliffs(a, b, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(a); deltas = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        a_b, b_b = a[idx], b[idx]
        deltas[k] = ((b_b < a_b).sum() - (b_b > a_b).sum()) / n
    return deltas

def tost_decision(d_hat, ci_lo, ci_hi, margin=MARGIN):
    # EQUIVALENT: CI ⊂ [-margin, +margin]
    if ci_lo > -margin and ci_hi < +margin:
        return 'EQUIVALENT'
    # IMPROVES_SIG: δ̂ > +margin (imputer migliora significativamente)
    if ci_lo > +margin:
        return 'IMPROVES_SIG'
    # WORSENS_SIG: δ̂ < -margin (imputer peggiora significativamente)
    if ci_hi < -margin:
        return 'WORSENS_SIG'
    return 'INCONCLUSIVE'

# ----------------------------------------------------------------------
# 3. Compute Cliff δ + TOST per (imputer, forecaster) vs no_imp
# ----------------------------------------------------------------------
def run_for_subset(level_label, mask_fn=None):
    """mask_fn: takes (series_idx_df, quartile_map) → boolean mask. None = all."""
    rows = []
    forecasters = sorted({fc for (_, fc) in per_series.keys()})
    print(f'\n   {level_label}: {len(forecasters)} forecasters')
    for fc in forecasters:
        if ('no_imp', fc) not in per_series:
            print(f'     {fc}: no no_imp baseline → skip'); continue
        base_df = per_series[('no_imp', fc)]
        if mask_fn is not None:
            base_df = mask_fn(base_df)
        base_s = base_df.set_index(['store_id','product_id'])['hourly_wape']
        for (imp, fc2), df in per_series.items():
            if fc2 != fc or imp == 'no_imp': continue
            other_df = df if mask_fn is None else mask_fn(df)
            other_s = other_df.set_index(['store_id','product_id'])['hourly_wape']
            common = base_s.index.intersection(other_s.index)
            a = base_s.loc[common].values; b = other_s.loc[common].values
            valid = ~(np.isnan(a) | np.isnan(b))
            a, b = a[valid], b[valid]
            if len(a) < 100: continue
            d_obs = cliffs_delta(a, b)
            boots = bootstrap_cliffs(a, b, n_boot=N_BOOT,
                                     seed=SEED + (hash(imp+fc) & 0xFFFF))
            ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
            decision = tost_decision(d_obs, ci_lo, ci_hi)
            rows.append({
                'level': level_label, 'forecaster': fc, 'imputer': imp,
                'cell': f'{imp}__{fc}', 'n_paired': int(len(a)),
                'cliffs_delta_obs': d_obs, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
                'tost_decision': decision,
                'threshold_equiv': abs(d_obs) < MARGIN,
            })
    return pd.DataFrame(rows)

print('\n3. RQ1 TOST: GLOBAL')
t0 = time.time()
rq1_global = run_for_subset('global', mask_fn=None)
print(f'   done in {time.time()-t0:.0f}s ({len(rq1_global)} comparisons)')

print('\n4. RQ1 TOST: stratificato per quartile')
all_strat = [rq1_global]
for q in ['Q1','Q2','Q3','Q4']:
    print(f'\n   --- {q} ---')
    def mk_mask(qq=q):
        def _f(df):
            tmp = df.merge(quart_map.reset_index(), on=['store_id','product_id'])
            return tmp[tmp.quartile == qq][['store_id','product_id','hourly_wape']]
        return _f
    t0 = time.time()
    r = run_for_subset(q, mask_fn=mk_mask())
    print(f'   done in {time.time()-t0:.0f}s ({len(r)} comparisons)')
    all_strat.append(r)

out = pd.concat(all_strat, ignore_index=True)
out.to_parquet(f'{RESULTS_DIR}/rq1_tost_stratified.parquet', index=False)
print(f'\n5. Saved: rq1_tost_stratified.parquet ({len(out)} rows)')

# ----------------------------------------------------------------------
# Summary tables per (level, forecaster)
# ----------------------------------------------------------------------
print('\n6. Summary per (level × forecaster):')
print(f'{"level":<8} {"forecaster":<15} {"n":<4} {"EQUIV":<6} {"IMPR":<6} {"WORS":<6} {"INC":<6}')
print('-'*60)
for lvl in ['global','Q1','Q2','Q3','Q4']:
    sub_l = out[out.level == lvl]
    for fc in sorted(sub_l.forecaster.unique()):
        sub = sub_l[sub_l.forecaster == fc]
        n_eq = (sub.tost_decision=='EQUIVALENT').sum()
        n_im = (sub.tost_decision=='IMPROVES_SIG').sum()
        n_wo = (sub.tost_decision=='WORSENS_SIG').sum()
        n_in = (sub.tost_decision=='INCONCLUSIVE').sum()
        print(f'{lvl:<8} {fc:<15} {len(sub):<4} {n_eq:<6} {n_im:<6} {n_wo:<6} {n_in:<6}')

# ----------------------------------------------------------------------
# 7. Plot: 4-grid stratified Cliff δ heatmap
# ----------------------------------------------------------------------
print('\n7. Plotting stratified heatmaps...')
IMP_ORDER = ['media_glob','media_cond','mediana_glob','mediana_cond',
             'forward_fill','seasonal_naive','linear_interp','lgb',
             'dlinear','saits','itransformer','timesnet','imputeformer']
FC_ORDER = ['lgb_m5lags','mlp_m5lags','tft','chronos_bolt','global_mean','dow_mean','ma_k21']
FC_SHORT = {'lgb_m5lags':'LGB_M5','mlp_m5lags':'MLP_M5','tft':'TFT','chronos_bolt':'Chronos',
            'global_mean':'GM','dow_mean':'DoW','ma_k21':'MA21'}

def heatmap_panel(ax, level, title):
    sub = out[out.level == level]
    piv = sub.pivot_table(index='imputer', columns='forecaster',
                          values='cliffs_delta_obs', aggfunc='first')
    piv = piv.reindex(index=IMP_ORDER, columns=FC_ORDER)
    decisions = sub.pivot_table(index='imputer', columns='forecaster',
                                values='tost_decision', aggfunc='first')
    decisions = decisions.reindex(index=IMP_ORDER, columns=FC_ORDER)
    im = ax.imshow(piv.values, cmap='RdYlGn', aspect='auto', vmin=-0.5, vmax=0.5)
    for i, imp in enumerate(IMP_ORDER):
        for j, fc in enumerate(FC_ORDER):
            v = piv.iloc[i, j]
            dec = decisions.iloc[i, j] if not pd.isna(decisions.iloc[i, j]) else None
            if pd.isna(v): continue
            color = 'white' if abs(v) > 0.30 else 'black'
            weight = 'bold' if abs(v) >= MARGIN else 'normal'
            ax.text(j, i, f'{v:+.2f}', ha='center', va='center', fontsize=9,
                    color=color, fontweight=weight)
            if dec == 'EQUIVALENT':
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                           edgecolor='blue', lw=2.0, zorder=3))
            elif dec == 'IMPROVES_SIG':
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                           edgecolor='darkgreen', lw=2.0, zorder=3))
            elif dec == 'WORSENS_SIG':
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                           edgecolor='darkred', lw=2.0, zorder=3))
            elif dec == 'INCONCLUSIVE':
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                           edgecolor='gray', lw=1.0, linestyle=':', zorder=2))
    ax.set_xticks(range(len(FC_ORDER)))
    ax.set_xticklabels([FC_SHORT[f] for f in FC_ORDER], fontsize=10, rotation=30)
    ax.set_yticks(range(len(IMP_ORDER)))
    ax.set_yticklabels(IMP_ORDER, fontsize=9)
    ax.set_title(title, fontsize=12, pad=8)
    return im

# Global heatmap (single panel)
fig_g, ax_g = plt.subplots(figsize=(11, 9))
im = heatmap_panel(ax_g, 'global', 'RQ1 — Imputer effect (Cliff\'s δ vs no_imp) — GLOBAL\n'
                   'BLUE box=EQUIVALENT, GREEN=IMPROVES, RED=WORSENS, gray dotted=INCONCLUSIVE')
plt.colorbar(im, ax=ax_g, label='Cliff\'s δ (>0=imputer improves)')
ax_g.set_xlabel('Forecaster', fontsize=11)
ax_g.set_ylabel('Imputer (treatment)', fontsize=11)
plt.tight_layout()
out_g = f'{FIG_DIR}/fig_rq1_tost_global.png'
fig_g.savefig(out_g, dpi=150, bbox_inches='tight')
print(f'   Saved: {out_g}')

# 4-grid stratified
fig_q, axes = plt.subplots(2, 2, figsize=(18, 14))
for ax, q in zip(axes.flat, ['Q1','Q2','Q3','Q4']):
    im = heatmap_panel(ax, q, f'{q}: imputer effect vs no_imp (Cliff\'s δ)')
    plt.colorbar(im, ax=ax, label='Cliff\'s δ')
fig_q.suptitle('RQ1 stratified by volume quartile — TOST decisions',
               fontsize=15, y=1.005)
plt.tight_layout()
out_q = f'{FIG_DIR}/fig_rq1_tost_4grid.png'
fig_q.savefig(out_q, dpi=150, bbox_inches='tight')
print(f'   Saved: {out_q}')

print('\nDONE')
