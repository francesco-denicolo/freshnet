"""
35_pareto_analysis_hpo.py — Pareto frontier + statistical analysis on HPO matrix
================================================================================
Aggrega tutte le 45 celle HPO (output con suffisso _hpo_test_per_series.parquet),
calcola Pareto frontier su (WAPE_h_med, |WPE_h_med|), Wilcoxon paired vs best,
Cliff's δ effect sizes. Salva figura + tabelle.
"""
import os, glob, functools
import numpy as np, pandas as pd
from scipy import stats
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Forecaster colors / markers (all 10 forecasters)
FC_COLORS = {
    'lgb_nolags':'#fdae61', 'lgb_m5lags':'#f46d43',
    'mlp_nolags':'#74add1', 'mlp_m5lags':'#4575b4',
    'tft':'#7b3294',
    'chronos_bolt':'#d73027',
    'timesfm':'#b15928',
    'global_mean':'#2ca02c', 'dow_mean':'#bcbd22', 'ma_k21':'#17becf',
}
FC_MARKERS = {
    'lgb_nolags':'^', 'lgb_m5lags':'^',
    'mlp_nolags':'D', 'mlp_m5lags':'D',
    'tft':'o',
    'chronos_bolt':'P',
    'timesfm':'*',
    'global_mean':'s', 'dow_mean':'X', 'ma_k21':'v',
}
FC_LABELS = {
    'lgb_nolags':'LGB_nolags', 'lgb_m5lags':'LGB_M5',
    'mlp_nolags':'MLP_nolags', 'mlp_m5lags':'MLP_M5',
    'tft':'TFT',
    'chronos_bolt':'Chronos-bolt',
    'timesfm':'TimesFM',
    'global_mean':'Global Mean', 'dow_mean':'DoW Mean', 'ma_k21':'MA (K=21)',
}
# Non-HPO forecasters (use _test_per_series.parquet, not _hpo)
NON_HPO_FC = {'chronos_bolt','timesfm','global_mean','dow_mean','ma_k21'}

# ============================================================================
# 1. Load all cells: HPO for tunable forecasters, baseline for non-HPO
# ============================================================================
print('1. Loading cells (HPO + non-HPO)...')

def parse_name(name):
    if '__' in name:
        return name.split('__', 1)
    # Fase A no_imp cells: lgb_nolags, mlp_nolags, lgb_m5lags, mlp_m5lags
    return 'no_imp', name

rows = []
per_series = {}
seen_cells = set()

# Load HPO files first (for forecasters that have HP to tune)
hpo_files = sorted(glob.glob(f'{RESULTS_DIR}/*_hpo_test_per_series.parquet'))
for f in hpo_files:
    name = os.path.basename(f).replace('_hpo_test_per_series.parquet', '')
    imp, fc = parse_name(name)
    df = pd.read_parquet(f)
    per_series[(imp, fc)] = df
    seen_cells.add(name)
    rows.append({
        'cell': name, 'imputer': imp, 'forecaster': fc, 'hpo': True,
        'wape_h_med': df['hourly_wape'].median(),
        'wpe_h_med': df['hourly_wpe'].median(),
        'abs_wpe_med': abs(df['hourly_wpe'].median()),
    })

# Load non-HPO cells for chronos_bolt and naive forecasters
non_hpo_files = sorted(glob.glob(f'{RESULTS_DIR}/*_test_per_series.parquet'))
for f in non_hpo_files:
    fn = os.path.basename(f)
    if '_hpo_test_per_series' in fn:
        continue
    name = fn.replace('_test_per_series.parquet', '')
    if name in seen_cells:
        continue
    # Map naive_<fc> (no imputer) -> no_imp__<fc>
    if name.startswith('naive_'):
        base = name.replace('naive_', '', 1)
        if base in NON_HPO_FC:
            imp, fc = 'no_imp', base
            name = f'no_imp__{base}'
            if name in seen_cells: continue
        else:
            continue
    else:
        imp, fc = parse_name(name)
    if fc not in NON_HPO_FC:
        continue
    df = pd.read_parquet(f)
    per_series[(imp, fc)] = df
    seen_cells.add(name)
    rows.append({
        'cell': name, 'imputer': imp, 'forecaster': fc, 'hpo': False,
        'wape_h_med': df['hourly_wape'].median(),
        'wpe_h_med': df['hourly_wpe'].median(),
        'abs_wpe_med': abs(df['hourly_wpe'].median()),
    })

mat = pd.DataFrame(rows)
print(f'   Total cells: {len(mat)} ({mat.hpo.sum()} HPO, {(~mat.hpo).sum()} non-HPO)')
print(f'   Forecasters: {sorted(mat.forecaster.unique())}')
print(f'   Imputers: {sorted(mat.imputer.unique())}')

# ============================================================================
# 2. Pareto frontier on (WAPE_h_med, |WPE_h_med|)
# ============================================================================
print('\n2. Pareto frontier (WAPE_h_med vs |WPE_h_med|)...')

def pareto_mask(x, y):
    """Return boolean mask: True if point is non-dominated (lower-left frontier)."""
    n = len(x)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if x[j] <= x[i] and y[j] <= y[i] and (x[j] < x[i] or y[j] < y[i]):
                is_pareto[i] = False
                break
    return is_pareto

x = mat['wape_h_med'].values
y = mat['abs_wpe_med'].values
mat['pareto'] = pareto_mask(x, y)
mat_sorted = mat.sort_values(['pareto','wape_h_med'], ascending=[False, True])
print(f'   Pareto-optimal cells: {mat.pareto.sum()}/{len(mat)}')
print(mat_sorted[mat_sorted.pareto].to_string(index=False))

# ============================================================================
# 3. Wilcoxon paired: best Pareto vs all others
# ============================================================================
print('\n3. Wilcoxon paired vs best cell (lowest WAPE_h_med)...')

best_row = mat.sort_values('wape_h_med').iloc[0]
best_key = (best_row['imputer'], best_row['forecaster'])
print(f'   Best: {best_row.cell} (WAPE_h_med={best_row.wape_h_med:.4f})')

best_df = per_series[best_key]
# Build (store_id, product_id) index for pairing
best_idx = best_df.set_index(['store_id','product_id'])['hourly_wape']

wilcoxon_results = []
for (imp, fc), df in per_series.items():
    if (imp, fc) == best_key: continue
    other_idx = df.set_index(['store_id','product_id'])['hourly_wape']
    common = best_idx.index.intersection(other_idx.index)
    a = best_idx.loc[common].values
    b = other_idx.loc[common].values
    # Drop NaN pairs
    valid = ~(np.isnan(a) | np.isnan(b))
    a = a[valid]; b = b[valid]
    diff = b - a
    nz = diff != 0
    if nz.sum() == 0:
        continue
    res = stats.wilcoxon(a[nz], b[nz], alternative='less')  # H1: best < other
    # Cliff's delta on valid pairs
    n = len(a)
    gt = (a < b).sum(); lt = (a > b).sum()
    cliff_d = (gt - lt) / n
    wilcoxon_results.append({
        'imputer': imp, 'forecaster': fc,
        'cell': f'{imp}__{fc}',
        'n_paired': n,
        'mean_diff': diff.mean(),
        'wape_med_other': df['hourly_wape'].median(),
        'p_value': res.pvalue,
        'cliffs_delta': cliff_d,
    })

wilcoxon_df = pd.DataFrame(wilcoxon_results).sort_values('cliffs_delta', ascending=False)
print(f'\n   Top 10 most-different from best (largest Cliff\'s δ):')
print(wilcoxon_df.head(10).to_string(index=False))

# Equivalence set: Friedman+Nemenyi CD (Demšar 2006 standard).
# Two cells are statistically indistinguishable if |Δ mean_rank| ≤ CD.
# Note: best cell here is by lowest WAPE_h_med, but the equivalence set is
# defined from Friedman ranking (script 45).  If Friedman best differs from
# median best, the cd_set is "indistinguishable from Friedman best",
# which is the Demšar-standard equivalence claim.
fr_path = f'{RESULTS_DIR}/friedman_nemenyi_ranks.parquet'
if os.path.exists(fr_path):
    fr = pd.read_parquet(fr_path)
    cd_equiv = set(fr[fr.cd_indistinguishable]['cell'])
    mat['equiv_to_best'] = mat['cell'].isin(cd_equiv)
    equiv_source = 'Nemenyi CD (Friedman post-hoc, α=0.05)'
    equiv_cells = cd_equiv
    friedman_best_cell = fr.iloc[0]['cell']
    print(f'\n   Friedman best (lowest mean rank): {friedman_best_cell}')
    print(f'   Median-WAPE best:                 {best_row.cell}')
else:
    equiv_cells = set()
    mat['equiv_to_best'] = False
    equiv_source = 'NONE (Friedman output missing)'
    friedman_best_cell = best_row.cell
print(f'\n   Equivalence set ({equiv_source}): {len(equiv_cells)} cells')
for c in mat[mat.equiv_to_best].sort_values('wape_h_med')['cell']:
    print(f'     - {c}')

# Save
wilcoxon_df.to_parquet(f'{RESULTS_DIR}/hpo_wilcoxon_vs_best.parquet', index=False)
mat.to_parquet(f'{RESULTS_DIR}/hpo_matrix_pareto.parquet', index=False)

# ============================================================================
# 4. Plot Pareto frontier — square axes, same limits x/y
# ============================================================================
print('\n4. Generating Pareto figure...')

# Knee point: pre-compute for plot
par_for_knee = mat[mat.pareto].copy()
par_for_knee['x_n'] = (par_for_knee.wape_h_med - par_for_knee.wape_h_med.min()) / (par_for_knee.wape_h_med.max() - par_for_knee.wape_h_med.min() + 1e-9)
par_for_knee['y_n'] = (par_for_knee.abs_wpe_med - par_for_knee.abs_wpe_med.min()) / (par_for_knee.abs_wpe_med.max() - par_for_knee.abs_wpe_med.min() + 1e-9)
par_for_knee['dist'] = np.sqrt(par_for_knee.x_n**2 + par_for_knee.y_n**2)
knee_cell = par_for_knee.sort_values('dist').iloc[0]['cell']

par = mat[mat.pareto].sort_values('wape_h_med')
knee_row = mat[mat.cell == knee_cell].iloc[0]
min_wpe_row = par.sort_values('abs_wpe_med').iloc[0]

def draw_panel(ax, mat_full, par, best, knee, min_wpe,
               xlim, ylim, title, show_labels=True):
    """Draw a Pareto scatter on the given axes."""
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    # Dominated points: colored by forecaster (pale)
    dom = mat_full[~mat_full.pareto]
    for _, r in dom.iterrows():
        ax.scatter([r.wape_h_med], [r.abs_wpe_med],
                   c=FC_COLORS.get(r.forecaster, '#cccccc'),
                   marker=FC_MARKERS.get(r.forecaster, 'o'),
                   s=70, alpha=0.55, edgecolor='gray', linewidth=0.4, zorder=2)
    # Pareto: same colors, full opacity + black edge
    for _, r in par.iterrows():
        ax.scatter([r.wape_h_med], [r.abs_wpe_med],
                   c=FC_COLORS.get(r.forecaster, '#000000'),
                   marker=FC_MARKERS.get(r.forecaster, 'o'),
                   s=200, edgecolor='black', linewidth=1.8, zorder=4)
    # (Equivalence rings omitted on Pareto plot to avoid clutter — see heatmap instead)
    # Frontier line
    ax.plot(par.wape_h_med, par.abs_wpe_med, '-', c='black', lw=1.5, alpha=0.5, zorder=3)
    # Highlights: gold star (best WAPE), green ring (knee), blue ring (min |WPE|)
    ax.scatter([best.wape_h_med], [best.abs_wpe_med], marker='*',
               s=450, c='gold', edgecolor='black', linewidth=1.5, zorder=5)
    ax.scatter([knee.wape_h_med], [knee.abs_wpe_med], marker='o',
               s=380, facecolors='none', edgecolor='#2ca02c', linewidth=3, zorder=6)
    ax.scatter([min_wpe.wape_h_med], [min_wpe.abs_wpe_med], marker='o',
               s=380, facecolors='none', edgecolor='#1f77b4', linewidth=3, zorder=6)
    # (Labels removed — highlights identified via legend)
    ax.set_xlabel('WAPE_h median (lower = better accuracy)', fontsize=15)
    ax.set_ylabel('|WPE_h median| (lower = less bias)', fontsize=15)
    ax.set_title(title, fontsize=16, pad=14)
    ax.grid(True, alpha=0.25, linestyle='--')

# --- Build figure: large single panel ---
fig, ax = plt.subplots(figsize=(20, 12))

xmin, xmax = mat.wape_h_med.min()-0.005, mat.wape_h_med.max()+0.012
ymin, ymax = mat.abs_wpe_med.min()-0.03, mat.abs_wpe_med.max()+0.03
draw_panel(ax, mat, par, best_row, knee_row, min_wpe_row,
           xlim=(xmin, xmax), ylim=(ymin, ymax),
           title=f'Pareto frontier WAPE × |WPE|  —  {len(mat)} cells, {mat.pareto.sum()} Pareto-optimal',
           show_labels=False)

# Custom legend: 1 entry per forecaster + highlights
import matplotlib.lines as mlines
legend_handles = []
# Forecaster entries (only those actually in matrix)
for fc in ['lgb_nolags','lgb_m5lags','mlp_nolags','mlp_m5lags','tft',
           'chronos_bolt','timesfm','global_mean','dow_mean','ma_k21']:
    if fc in mat.forecaster.unique():
        legend_handles.append(
            mlines.Line2D([], [], color=FC_COLORS[fc], marker=FC_MARKERS[fc],
                          linestyle='None', markeredgecolor='black', markersize=10,
                          label=FC_LABELS[fc])
        )
# Highlights
n_equiv = mat.equiv_to_best.sum() if 'equiv_to_best' in mat.columns else 0
legend_handles += [
    mlines.Line2D([], [], color='gold', marker='*', linestyle='None',
                  markeredgecolor='black', markersize=15,
                  label=f'★ Best WAPE: {best_row.imputer}__{FC_LABELS.get(best_row.forecaster, best_row.forecaster)}'),
    mlines.Line2D([], [], color='#ff7f0e', marker='o', linestyle='--', markerfacecolor='none',
                  markeredgewidth=2.5, markersize=12,
                  label=f'⊙ CD-indistinguishable from Friedman best (n={n_equiv})'),
    mlines.Line2D([], [], color='#2ca02c', marker='o', linestyle='None', markerfacecolor='none',
                  markeredgewidth=2.5, markersize=12,
                  label=f'● Knee: {knee_row.imputer}__{FC_LABELS.get(knee_row.forecaster, knee_row.forecaster)}'),
    mlines.Line2D([], [], color='#1f77b4', marker='o', linestyle='None', markerfacecolor='none',
                  markeredgewidth=2.5, markersize=12,
                  label=f'● Min |WPE|: {min_wpe_row.imputer}__{FC_LABELS.get(min_wpe_row.forecaster, min_wpe_row.forecaster)}'),
]
ax.legend(handles=legend_handles, loc='center left',
          bbox_to_anchor=(1.02, 0.5), fontsize=13, framealpha=0.95)
ax.tick_params(axis='both', labelsize=13)
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

plt.tight_layout()
out_fig = f'{FIG_DIR}/fig_pareto_hpo.png'
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
print(f'   Saved: {out_fig}')

# ============================================================================
# 4b. General heatmap WAPE_h_med (matching stratified style)
# ============================================================================
print('\n4b. Generating general heatmap...')
IMP_ORDER = ['no_imp','media_glob','media_cond','mediana_glob','mediana_cond',
             'forward_fill','seasonal_naive','linear_interp','lgb',
             'dlinear','saits','itransformer','timesnet','imputeformer']
FC_ORDER = ['lgb_nolags','lgb_m5lags','mlp_nolags','mlp_m5lags','tft',
            'chronos_bolt','timesfm','global_mean','dow_mean','ma_k21']
FC_SHORT = {'lgb_nolags':'LGB_nl','lgb_m5lags':'LGB_M5','mlp_nolags':'MLP_nl','mlp_m5lags':'MLP_M5',
            'tft':'TFT','chronos_bolt':'Chron','timesfm':'TimesFM',
            'global_mean':'GM','dow_mean':'DoW','ma_k21':'MA21'}

fig2, ax2 = plt.subplots(figsize=(13, 10))
pivot = mat.pivot(index='imputer', columns='forecaster', values='wape_h_med')
pivot = pivot.reindex(index=IMP_ORDER, columns=FC_ORDER)
eq_mat = mat[mat.equiv_to_best] if 'equiv_to_best' in mat.columns else pd.DataFrame(columns=mat.columns)
eq_cells = set(eq_mat.cell)

im = ax2.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto', vmin=0.95, vmax=1.20)
for i, imp in enumerate(IMP_ORDER):
    for j, fc in enumerate(FC_ORDER):
        v = pivot.iloc[i, j]
        if np.isnan(v): continue
        cell = f'{imp}__{fc}'
        is_best = (cell == best_row.cell)
        is_equiv = cell in eq_cells
        text = f'{v:.3f}'
        color = 'white' if v > 1.10 or v < 0.98 else 'black'
        ax2.text(j, i, text, ha='center', va='center', fontsize=11,
                color=color, fontweight='bold' if (is_best or is_equiv) else 'normal')
        if is_best:
            ax2.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                        edgecolor='blue', lw=3.5, zorder=3))
        elif is_equiv:
            ax2.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                        edgecolor='orange', lw=2.2, linestyle='--', zorder=2))

ax2.set_xticks(range(len(FC_ORDER)))
ax2.set_xticklabels([FC_SHORT[c] for c in FC_ORDER], fontsize=12, rotation=15)
ax2.set_yticks(range(len(IMP_ORDER)))
ax2.set_yticklabels(IMP_ORDER, fontsize=12)
ax2.set_title(f'General heatmap WAPE_h_med — blue=best ({best_row.cell}), '
              f'orange dashed=Nemenyi CD-equivalent to Friedman best (n={len(eq_cells)})',
              fontsize=13, pad=14)
ax2.set_xlabel('Forecaster', fontsize=13)
ax2.set_ylabel('Imputer', fontsize=13)
plt.colorbar(im, ax=ax2, label='WAPE_h_med')
plt.tight_layout()
out_fig2 = f'{FIG_DIR}/fig_heatmap_general.png'
plt.savefig(out_fig2, dpi=150, bbox_inches='tight')
print(f'   Saved: {out_fig2}')

# ============================================================================
# 5. Summary
# ============================================================================
print('\n5. Summary')
print(f'   Total cells:  {len(mat)}')
print(f'   Pareto-opt:   {mat.pareto.sum()}')
print(f'   Dominated:    {(~mat.pareto).sum()}')
print(f'   Best WAPE:    {best_row.cell} = {best_row.wape_h_med:.4f} (|WPE|={best_row.abs_wpe_med:.4f})')

# Knee point: closest to origin on (wape, |wpe|) plane after min-max normalization
par_norm = par.copy()
par_norm['x_n'] = (par_norm.wape_h_med - par_norm.wape_h_med.min()) / (par_norm.wape_h_med.max() - par_norm.wape_h_med.min() + 1e-9)
par_norm['y_n'] = (par_norm.abs_wpe_med - par_norm.abs_wpe_med.min()) / (par_norm.abs_wpe_med.max() - par_norm.abs_wpe_med.min() + 1e-9)
par_norm['dist'] = np.sqrt(par_norm.x_n**2 + par_norm.y_n**2)
knee = par_norm.sort_values('dist').iloc[0]
print(f'   Knee point:   {knee.cell} (WAPE={knee.wape_h_med:.4f}, |WPE|={knee.abs_wpe_med:.4f})')

# Min |WPE|
min_wpe = par.sort_values('abs_wpe_med').iloc[0]
print(f'   Min |WPE|:    {min_wpe.cell} = {min_wpe.abs_wpe_med:.4f} (WAPE={min_wpe.wape_h_med:.4f})')

print('\nDONE')
