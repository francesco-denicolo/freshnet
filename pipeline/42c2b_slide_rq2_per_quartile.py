"""
42c2b_slide_rq2_per_quartile.py — Slide RQ2 per quartile.

Layout 16:9:
  - Left:  heatmap P(concord) forecaster × quartile, diverging colormap centered on 0.5
  - Right: 4 pattern + take-away

Output: fig_slide_rq2_per_quartile.png
"""
import os, functools
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')

# Load actual data from script 42c2 output
summary = pd.read_parquet(os.path.join(RESULTS_DIR,
                                         'rq2_pairwise_concordance_per_quartile.parquet'))
glob_path = os.path.join(RESULTS_DIR, 'rq2_pairwise_concordance.parquet')
glob_df = pd.read_parquet(glob_path)

# Pivot: forecaster × quartile, median P(concord)
piv = summary.pivot(index='forecaster', columns='quartile', values='median')[['Q1','Q2','Q3','Q4']]
piv['Globale'] = glob_df.set_index('forecaster')['median']

# Row order (by family)
fc_order = ['GlobalMean', 'DoWMean', 'MA_K56', 'Chronos-bolt', 'TimesFM',
            'TFT', 'MLP_M5', 'LGB_M5']
fc_families = {'GlobalMean':'naive','DoWMean':'naive','MA_K56':'naive',
               'Chronos-bolt':'foundation','TimesFM':'foundation',
               'TFT':'DL+lag','MLP_M5':'ML+lag','LGB_M5':'ML+lag'}
piv = piv.reindex(fc_order)
piv = piv[['Globale','Q1','Q2','Q3','Q4']]

P_matrix = piv.values

# Diverging colormap centered on 0.5: red (< 0.5 inverse) → white (0.5) → green (> 0.5)
cmap = LinearSegmentedColormap.from_list('Pmap',
    [(0.00, '#67001f'),  # dark red - very inverse
     (0.25, '#d73027'),  # red
     (0.40, '#fee08b'),  # light yellow
     (0.50, '#ffffff'),  # white at 0.5
     (0.60, '#a6dba0'),  # light green
     (0.75, '#5aae61'),  # green
     (1.00, '#1b7837')])  # dark green

# Build figure
fig = plt.figure(figsize=(20, 11.25))
gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.15)

# Left: heatmap
ax = fig.add_subplot(gs[0])
im = ax.imshow(P_matrix, cmap=cmap, aspect='auto', vmin=0.3, vmax=0.95)

for i, fc in enumerate(fc_order):
    for j, col in enumerate(piv.columns):
        v = P_matrix[i, j]
        # color of text
        if v > 0.75 or v < 0.4:
            text_color = 'white'
        else:
            text_color = 'black'
        ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                fontsize=13, fontweight='bold', color=text_color)

# Black rectangle for values < 0.5 (inverse)
for i in range(len(fc_order)):
    for j in range(len(piv.columns)):
        if P_matrix[i,j] < 0.5:
            rect = Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, fill=False,
                              edgecolor='black', lw=2.5, linestyle='--', zorder=3)
            ax.add_patch(rect)

ax.set_xticks(range(len(piv.columns)))
ax.set_xticklabels(list(piv.columns), fontsize=12)
ax.set_yticks(range(len(fc_order)))
ax.set_yticklabels([f'{fc}\n[{fc_families[fc]}]' for fc in fc_order], fontsize=11)
ax.tick_params(axis='y', length=0)

ax.set_title('Mediana P(concord) per (forecaster × quartile)\n'
             'verde = positivo (recovery predice), bianco = random (0.5), rosso = inverso',
             fontsize=12, pad=10)
ax.set_xlabel('Volume regime', fontsize=12)

cbar = plt.colorbar(im, ax=ax, pad=0.02)
cbar.set_label('Mediana P(concord)', fontsize=11)

# Family separators
for sep in [3, 5, 6]:
    ax.axhline(sep - 0.5, color='black', lw=1.2, alpha=0.7)
# Globale | quartili separator
ax.axvline(0.5, color='black', lw=1.5, alpha=0.7)

# Right: pattern boxes
ax2 = fig.add_subplot(gs[1])
ax2.axis('off')

ax2.text(0.5, 0.97, '4 Pattern di recovery -> forecasting',
         ha='center', va='top', fontsize=15, fontweight='bold',
         transform=ax2.transAxes)

patterns = [
    ('PATTERN A — Naive: predice fortemente, decresce con volume',
     '#1b7837',
     'GlobalMean / DoWMean / MA_K56',
     'P da ~0.92 (Q1) a ~0.67 (Q4)',
     'naive usano direttamente valori imputati'),
    ('PATTERN B — Foundation: pattern complesso',
     '#5aae61',
     'Chronos / TimesFM',
     'Chronos: 0.65 -> CROLLO a 0.33 in Q4 (inverso!)\nTimesFM: 0.57 -> 0.46',
     'boundary condition transfer learning in alto volume'),
    ('PATTERN C — ML+lag: indifferenti, P invariante',
     '#fee08b',
     'MLP_M5 / LGB_M5',
     'P ~ 0.50-0.55 in tutti i quartili',
     'lag M5 = buffer strutturale (coerente con RQ1: W ~ 0)'),
    ('PATTERN D — TFT: INVERSO strutturale (anomalia)',
     '#d73027',
     'TFT',
     'P = 0.34-0.43 in ogni Q (sempre < 0.5)',
     'preferenza inversa stabile, picco in Q4'),
]

box_y_start = 0.91
box_h = 0.18
for i, (title, color, fc_text, vals_text, note) in enumerate(patterns):
    y_top = box_y_start - i * (box_h + 0.02)
    # Title bar
    box = FancyBboxPatch((0.02, y_top - 0.045), 0.96, 0.045,
                          boxstyle='round,pad=0.01',
                          facecolor=color, alpha=0.55, edgecolor='black',
                          transform=ax2.transAxes, linewidth=1.2)
    ax2.add_patch(box)
    ax2.text(0.5, y_top - 0.022, title, ha='center', va='center',
             fontsize=10.5, fontweight='bold', color='black',
             transform=ax2.transAxes)

    ax2.text(0.04, y_top - 0.075, f'   • {fc_text}',
             fontsize=10, fontweight='bold', color='black',
             transform=ax2.transAxes)
    ax2.text(0.04, y_top - 0.105, f'     {vals_text}',
             fontsize=9.5, color='#444', family='monospace',
             transform=ax2.transAxes)
    ax2.text(0.04, y_top - 0.15, f'   -> {note}',
             fontsize=10, style='italic', color='black',
             transform=ax2.transAxes)

# Take-away
ax2.text(0.5, 0.07,
         'TAKE-AWAY:\n'
         'Dicotomia naive (predice) vs others (no) e\' massima\n'
         'in basso volume e si attenua in alto. Finding nuovo:\n'
         'Chronos cambia segno (0.65 -> 0.33) in Q4 = boundary\n'
         'condition del transfer learning.',
         ha='center', va='center', fontsize=10.5, fontweight='bold',
         transform=ax2.transAxes,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f4fd',
                    edgecolor='#1f77b4', linewidth=1.5))

fig.suptitle('RQ2 — Recovery quality predice forecasting? Stratificazione per quartile',
             fontsize=18, fontweight='bold', y=0.985)

plt.tight_layout(rect=[0, 0, 1, 0.965])
out = os.path.join(FIG_DIR, 'fig_slide_rq2_per_quartile.png')
plt.savefig(out, dpi=120, bbox_inches='tight', facecolor='white')
print(f'Saved: {out}')
