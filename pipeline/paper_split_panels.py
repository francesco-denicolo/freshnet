"""Internal review (length): split the two-panel RQ1 and RQ2 figures into single panels.
Panel (a) (global) stays in the main article; panel (b) (by volume quartile) moves to the
Supplementary Material. Same data and styling as paper_extra_figs.py; RQ2 uses the held-out
Track-A test-mask recovery (TEST_MASK files), as in the paper."""
import os, functools
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

RES = os.path.join(os.path.dirname(__file__), 'results')
OUT = '/Users/utente/Desktop/MDPI_Overleaf/figures'
ORDER = ['global_mean','dow_mean','ma_k56','croston','sba','tsb','lgb_m5lags','mlp_m5lags','tft','chronos_bolt','timesfm']
LAB = {'global_mean':'Global Mean','dow_mean':'DoW Mean','ma_k56':'MA (K=56)','croston':'Croston','sba':'SBA','tsb':'TSB',
       'lgb_m5lags':'LGB-M5','mlp_m5lags':'MLP-M5','tft':'TFT','chronos_bolt':'Chronos','timesfm':'TimesFM'}
GROUP = {'global_mean':'naive','dow_mean':'naive','ma_k56':'naive','croston':'intermittent','sba':'intermittent','tsb':'intermittent',
         'lgb_m5lags':'lag-ML','mlp_m5lags':'lag-ML','tft':'deep','chronos_bolt':'foundation','timesfm':'foundation'}
GC = {'naive':'#4c72b0','intermittent':'#55a868','lag-ML':'#c44e52','deep':'#8172b3','foundation':'#ccb974'}
T2K = {'GlobalMean':'global_mean','DoWMean':'dow_mean','MA_K56':'ma_k56','Croston':'croston','SBA':'sba','TSB':'tsb',
       'LGB_M5':'lgb_m5lags','MLP_M5':'mlp_m5lags','TFT':'tft','Chronos-bolt':'chronos_bolt','TimesFM':'timesfm'}
QS = ['Q1','Q2','Q3','Q4']
FAMLEG = [Patch(fc=GC[k], label=k) for k in ['naive','intermittent','lag-ML','deep','foundation']]
xs = np.arange(len(ORDER))

# ---- RQ1
fps = pd.read_parquet(f'{RES}/friedman_per_forecaster_summary.parquet')
g = fps[fps.level=='global'].set_index('forecaster')
fig, ax = plt.subplots(figsize=(7.6, 4.4))
ax.bar(xs, [g.loc[f,'kendall_W'] for f in ORDER], color=[GC[GROUP[f]] for f in ORDER])
for y,lab in [(0.1,'negligible'),(0.3,'small'),(0.5,'moderate')]:
    ax.axhline(y, ls=':', c='0.6', lw=0.9); ax.text(len(ORDER)-0.4, y+0.005, lab, fontsize=7, ha='right', color='0.4')
ax.set_xticks(xs); ax.set_xticklabels([LAB[f] for f in ORDER], rotation=40, ha='right', fontsize=9)
ax.set_ylabel("Kendall's $W$"); ax.legend(handles=FAMLEG, fontsize=7.5, loc='upper right')
plt.tight_layout(); plt.savefig(f'{OUT}/fig_rq1_kendallw.png', dpi=200, bbox_inches='tight'); plt.close()
fig, ax = plt.subplots(figsize=(7.6, 4.4)); qcols = ['#c6dbef','#6baed6','#2171b5','#08306b']; bw = 0.2
for j,q in enumerate(QS):
    gq = fps[fps.level==q].set_index('forecaster')
    ax.bar(xs+(j-1.5)*bw, [gq.loc[f,'kendall_W'] for f in ORDER], bw, label=q, color=qcols[j], edgecolor='white', linewidth=0.3)
ax.axhline(0.1, ls=':', c='0.6', lw=0.9); ax.text(len(ORDER)-0.4, 0.105, 'negligible', fontsize=7, ha='right', color='0.4')
ax.set_xticks(xs); ax.set_xticklabels([LAB[f] for f in ORDER], rotation=40, ha='right', fontsize=9)
ax.set_ylabel("Kendall's $W$"); ax.legend(fontsize=8, title='Quartile', ncol=4, columnspacing=0.8, handlelength=1.1)
plt.tight_layout(); plt.savefig(f'{OUT}/fig_rq1_kendallw_quartile.png', dpi=200, bbox_inches='tight'); plt.close()

# ---- RQ2 (test-mask recovery)
s2 = pd.read_parquet(f'{RES}/rq2_pairwise_concordance_test.parquet'); s2['fk'] = s2.forecaster.map(T2K); s2 = s2.set_index('fk')
q2 = pd.read_parquet(f'{RES}/rq2_pairwise_concordance_per_quartile_test.parquet'); q2['fk'] = q2.forecaster.map(T2K)
fig, ax = plt.subplots(figsize=(7.6, 4.4))
for i,f in enumerate(ORDER):
    r = s2.loc[f]; c = GC[GROUP[f]]
    ax.add_patch(Rectangle((i-0.3, r['q25']), 0.6, r['q75']-r['q25'], fc=c, ec='0.3', alpha=0.8))
    ax.plot([i-0.3,i+0.3],[r['median']]*2, c='k', lw=1.4)
    ax.plot([i,i],[r['min'],r['q25']], c='0.4', lw=0.8); ax.plot([i,i],[r['q75'],r['max']], c='0.4', lw=0.8)
ax.axhline(0.5, ls='--', c='r', lw=1, label='$P=0.5$ (no relation)')
ax.set_xlim(-0.6, len(ORDER)-0.4)
ax.set_xticks(xs); ax.set_xticklabels([LAB[f] for f in ORDER], rotation=40, ha='right', fontsize=9)
ax.set_ylabel('Pairwise concordance $P$'); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f'{OUT}/fig_rq2_concordance.png', dpi=200, bbox_inches='tight'); plt.close()
fig, ax = plt.subplots(figsize=(7.6, 4.4)); xq = np.arange(len(QS))
for f in ORDER:
    gg = q2[q2.fk==f].set_index('quartile')
    ax.plot(xq, [gg.loc[q,'median'] for q in QS], marker='o', ms=4, c=GC[GROUP[f]], alpha=0.85)
ax.axhline(0.5, ls='--', c='r', lw=1)
ax.set_xticks(xq); ax.set_xticklabels(QS); ax.set_xlabel('Volume quartile'); ax.set_ylabel('Median concordance $P$')
ax.legend(handles=FAMLEG, fontsize=7.5)
plt.tight_layout(); plt.savefig(f'{OUT}/fig_rq2_concordance_quartile.png', dpi=200, bbox_inches='tight'); plt.close()
print('saved 4 single-panel figures')
