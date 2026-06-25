"""
49b_slide_rq1_per_quartile.py — Slide-ready figure: RQ1 stratified per volume quartile.

Layout 16:9:
  - Left panel:  heatmap Kendall W per (forecaster × quartile)
                 + bordo blu sui casi in cui no_imp NON è CD-equiv (= imputer aiuta)
                 + bordo rosso sui casi in cui no_imp È CD-equiv (= imputer non aiuta)
  - Right panel: text box con i 3 regimi emersi + take-away

Output: fig_slide_rq1_per_quartile.png (slide 16:9)
"""
import os, functools
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')

# Hardcoded values from PAPER_FINDINGS.md (rq1 stratified analysis)
# Row order: by family (naive → foundation → DL+lag → ML+lag)
data = pd.DataFrame([
    # forecaster      family          Q1      Q2      Q3      Q4    no_imp_helps_q1, q2, q3, q4
    ['Global Mean',  'naive',         0.754,  0.672,  0.423,  0.193, True, True, True, True],
    ['DoW Mean',     'naive',         0.691,  0.628,  0.415,  0.193, True, True, True, True],
    ['MA_K56',       'naive',         0.729,  0.655,  0.425,  0.191, True, True, True, True],
    ['Chronos-bolt', 'foundation',    0.501,  0.415,  0.153,  0.182, True, True, True, True],
    ['TimesFM',      'foundation',    0.447,  0.263,  0.095,  0.080, False, True, True, True],
    ['TFT',          'DL+lag',        0.263,  0.215,  0.160,  0.348, True, True, True, True],
    ['MLP_M5',       'ML+lag',        0.034,  0.033,  0.031,  0.033, False, False, True, True],
    ['LGB_M5',       'ML+lag',        0.004,  0.005,  0.012,  0.033, False, False, True, True],
],
columns=['forecaster', 'family', 'Q1', 'Q2', 'Q3', 'Q4',
         'helps_Q1', 'helps_Q2', 'helps_Q3', 'helps_Q4'])

forecasters = data['forecaster'].tolist()
quartiles = ['Q1', 'Q2', 'Q3', 'Q4']
W_matrix = data[quartiles].values

# Build figure 16:9
fig = plt.figure(figsize=(20, 11.25))
gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.15)

# ============================================================================
# LEFT panel: heatmap (forecaster × quartile) colored by Kendall W
# ============================================================================
ax = fig.add_subplot(gs[0])

# Colormap: white → orange → red → dark red for Kendall W
# Bands: 0-0.1 negligible, 0.1-0.3 small, 0.3-0.5 moderate, ≥0.5 large
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('Wmap',
    [(0.000, '#f5f5f5'),
     (0.125, '#fee08b'),
     (0.375, '#fdae61'),
     (0.625, '#d73027'),
     (1.000, '#67001f')])

im = ax.imshow(W_matrix, cmap=cmap, aspect='auto', vmin=0.0, vmax=0.8)

# Annotate each cell with W value and helps_QX
for i, fc in enumerate(forecasters):
    for j, q in enumerate(quartiles):
        w = W_matrix[i, j]
        helps = data[f'helps_{q}'].iloc[i]

        # text in cell
        color = 'white' if w > 0.45 else 'black'
        ax.text(j, i, f'{w:.3f}', ha='center', va='center',
                fontsize=13, fontweight='bold', color=color)

        # Border: green = helps (no_imp NOT in CD-equiv); red = doesn't help
        if helps:
            border_color = '#2ca02c'  # green
            border_style = '-'
            lw = 0
        else:
            border_color = '#d73027'  # red
            border_style = '-'
            lw = 3.5

        if lw > 0:
            rect = Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, fill=False,
                              edgecolor=border_color, lw=lw, linestyle=border_style,
                              zorder=3)
            ax.add_patch(rect)

# Category labels on right side of each row
category_text = {
    'naive': 'Imputer-\nsensitive\n(forte)',
    'foundation': 'Imputer-\nsensitive\n(medio)',
    'DL+lag': 'Anomalo',
    'ML+lag': 'Imputer-\nirrelevant',
}

ax.set_xticks(range(len(quartiles)))
ax.set_xticklabels([f'{q}\n(N≈12.5K)' for q in quartiles], fontsize=12)
ax.set_yticks(range(len(forecasters)))
ax.set_yticklabels([f'{fc}\n[{fam}]' for fc, fam in zip(forecasters, data['family'].tolist())],
                    fontsize=11)
ax.tick_params(axis='y', length=0)

ax.set_title('Kendall\'s W per (forecaster × quartile)\n'
             '(rosso = no_imp CD-equiv = imputer NON aiuta)', fontsize=13, pad=10)
ax.set_xlabel('Volume quartile', fontsize=12)

# Colorbar
cbar = plt.colorbar(im, ax=ax, pad=0.02)
cbar.set_label('Kendall\'s W (concordance)', fontsize=11)
cbar.ax.text(1.4, 0.05, 'negligible', va='center', fontsize=8, color='gray')
cbar.ax.text(1.4, 0.20, 'small', va='center', fontsize=8, color='gray')
cbar.ax.text(1.4, 0.40, 'moderate', va='center', fontsize=8, color='gray')
cbar.ax.text(1.4, 0.65, 'large', va='center', fontsize=8, color='gray')

# Family separator lines
for sep_idx in [3, 5, 6]:  # after naive, foundation, DL+lag
    ax.axhline(sep_idx - 0.5, color='black', lw=1.2, alpha=0.7)

# ============================================================================
# RIGHT panel: 3 regimi + take-away
# ============================================================================
ax2 = fig.add_subplot(gs[1])
ax2.axis('off')

ax2.text(0.5, 0.97, '3 Regimi di "imputer relevance" emergenti',
         ha='center', va='top', fontsize=15, fontweight='bold',
         transform=ax2.transAxes)

regimes = [
    ('REGIME A — Imputer matters in ANY volume',
     '#d73027',
     ['Naive aggregati (W decresce ma resta ≥ small)',
      'Chronos-bolt (W decresce ma resta small)'],
     'sempre imputer-sensitive'),
    ('REGIME B — Imputer matters only in LOW volume',
     '#fdae61',
     ['TimesFM (W large in Q1, negligible in Q3-Q4)'],
     'sensibilità condizionale al regime'),
    ('REGIME C — Imputer NEVER matters significantly',
     '#2ca02c',
     ['MLP_M5 (W invariante ≈ 0 in tutti i quartili)',
      'LGB_M5 (W invariante ≈ 0)'],
     'invariant insensitivity → lag M5 = buffer strutturale'),
]

box_y_start = 0.91
box_h = 0.21
for i, (title, color, items, note) in enumerate(regimes):
    y_top = box_y_start - i * (box_h + 0.02)

    # Title bar
    box = FancyBboxPatch((0.02, y_top - 0.05), 0.96, 0.05,
                          boxstyle='round,pad=0.01',
                          facecolor=color, alpha=0.55, edgecolor='black',
                          transform=ax2.transAxes, linewidth=1.2)
    ax2.add_patch(box)
    ax2.text(0.5, y_top - 0.025, title, ha='center', va='center',
             fontsize=11.5, fontweight='bold', color='black',
             transform=ax2.transAxes)

    # Content
    for j, item in enumerate(items):
        ax2.text(0.04, y_top - 0.08 - j * 0.025, '   • ' + item,
                 fontsize=10, color='black', transform=ax2.transAxes,
                 family='monospace')

    # Note
    n_items = len(items)
    ax2.text(0.04, y_top - 0.085 - n_items * 0.025 - 0.012,
             f'   → {note}',
             fontsize=10.5, style='italic', color='black',
             transform=ax2.transAxes)

# Anomalia box
ax2.text(0.5, 0.21, 'ANOMALIA: TFT (W = 0.26 → 0.16 → 0.16 → 0.35, non monotono)',
         ha='center', va='center', fontsize=10.5, fontweight='bold',
         color='#7b3294',
         transform=ax2.transAxes,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#f3e6f7',
                    edgecolor='#7b3294', linewidth=1.0))

# Take-away
ax2.text(0.5, 0.10,
         'TAKE-AWAY:\n'
         'La dicotomia ML+lag (irrelevant) vs others (sensitive)\n'
         'REGGE in ogni quartile. La forza dell\'effetto imputer\n'
         'decresce con il volume — eccezione: lag M5 = buffer\n'
         'invariante (W ≈ 0 in tutti i regimi).',
         ha='center', va='center', fontsize=11, fontweight='bold',
         transform=ax2.transAxes,
         bbox=dict(boxstyle='round,pad=0.6', facecolor='#e8f4fd',
                    edgecolor='#1f77b4', linewidth=1.5))

# Suptitle
fig.suptitle('RQ1 — Imputer aiuta? Stratificazione per quartile di volume',
             fontsize=18, fontweight='bold', y=0.985)

plt.tight_layout(rect=[0, 0, 1, 0.965])
out = os.path.join(FIG_DIR, 'fig_slide_rq1_per_quartile.png')
plt.savefig(out, dpi=120, bbox_inches='tight', facecolor='white')
print(f'Saved: {out}')
