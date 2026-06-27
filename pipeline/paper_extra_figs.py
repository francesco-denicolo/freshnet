"""Regenerate the four paper figures that have no standing generator, now with the
11 forecasters (intermittent-demand family added): fig_rq1_kendallw,
fig_rq2_concordance, fig_rq4_crossover, fig_deploy_intersection.
Writes into the Overleaf figures/ folder."""
import os, functools
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

RES = os.path.join(os.path.dirname(__file__), 'results')
OUT = '/Users/utente/Desktop/MDPI_Overleaf/figures'

ORDER = ['global_mean','dow_mean','ma_k56','croston','sba','tsb',
         'lgb_m5lags','mlp_m5lags','tft','chronos_bolt','timesfm']
LAB = {'global_mean':'Global Mean','dow_mean':'DoW Mean','ma_k56':'MA (K=56)',
       'croston':'Croston','sba':'SBA','tsb':'TSB','lgb_m5lags':'LGB-M5',
       'mlp_m5lags':'MLP-M5','tft':'TFT','chronos_bolt':'Chronos','timesfm':'TimesFM'}
GROUP = {'global_mean':'naive','dow_mean':'naive','ma_k56':'naive',
         'croston':'intermittent','sba':'intermittent','tsb':'intermittent',
         'lgb_m5lags':'lag-ML','mlp_m5lags':'lag-ML','tft':'deep',
         'chronos_bolt':'foundation','timesfm':'foundation'}
GC = {'naive':'#4c72b0','intermittent':'#55a868','lag-ML':'#c44e52',
      'deep':'#8172b3','foundation':'#ccb974'}
# RQ2 summary uses title-case forecaster keys
T2K = {'GlobalMean':'global_mean','DoWMean':'dow_mean','MA_K56':'ma_k56',
       'Croston':'croston','SBA':'sba','TSB':'tsb','LGB_M5':'lgb_m5lags',
       'MLP_M5':'mlp_m5lags','TFT':'tft','Chronos-bolt':'chronos_bolt','TimesFM':'timesfm'}
QS = ['Q1','Q2','Q3','Q4']

# ---------------------------------------------------------------- RQ1
fps = pd.read_parquet(f'{RES}/friedman_per_forecaster_summary.parquet')
g = fps[fps.level=='global'].set_index('forecaster')
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
xs = np.arange(len(ORDER)); cols = [GC[GROUP[f]] for f in ORDER]
ax[0].bar(xs, [g.loc[f,'kendall_W'] for f in ORDER], color=cols)
for y,lab in [(0.1,'negligible'),(0.3,'small'),(0.5,'moderate')]:
    ax[0].axhline(y, ls=':', c='0.6', lw=0.9); ax[0].text(len(ORDER)-0.4, y+0.005, lab, fontsize=7, ha='right', color='0.4')
ax[0].set_xticks(xs); ax[0].set_xticklabels([LAB[f] for f in ORDER], rotation=40, ha='right', fontsize=9)
ax[0].set_ylabel("Kendall's $W$"); ax[0].set_title('(a) Imputer-effect strength per forecaster (global)', fontsize=11)
for q in QS:
    gq = fps[fps.level==q].set_index('forecaster')
    ax[1].plot(xs, [gq.loc[f,'kendall_W'] for f in ORDER], marker='o', ms=4, label=q)
ax[1].set_xticks(xs); ax[1].set_xticklabels([LAB[f] for f in ORDER], rotation=40, ha='right', fontsize=9)
ax[1].set_ylabel("Kendall's $W$"); ax[1].set_title('(b) By volume quartile', fontsize=11); ax[1].legend(fontsize=8, title='Quartile')
ax[1].axhline(0.1, ls=':', c='0.6', lw=0.9)
leg = [Patch(fc=GC[k], label=k) for k in ['naive','intermittent','lag-ML','deep','foundation']]
ax[0].legend(handles=leg, fontsize=7.5, loc='upper right')
plt.tight_layout(); plt.savefig(f'{OUT}/fig_rq1_kendallw.png', dpi=200, bbox_inches='tight'); plt.close()
print('saved fig_rq1_kendallw')

# ---------------------------------------------------------------- RQ2
s2 = pd.read_parquet(f'{RES}/rq2_pairwise_concordance.parquet')
s2['fk'] = s2.forecaster.map(T2K); s2 = s2.set_index('fk')
q2 = pd.read_parquet(f'{RES}/rq2_pairwise_concordance_per_quartile.parquet'); q2['fk'] = q2.forecaster.map(T2K)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for i,f in enumerate(ORDER):
    r = s2.loc[f]; c = GC[GROUP[f]]
    ax[0].add_patch(Rectangle((i-0.3, r['q25']), 0.6, r['q75']-r['q25'], fc=c, ec='0.3', alpha=0.8))
    ax[0].plot([i-0.3,i+0.3],[r['median']]*2, c='k', lw=1.4)
    ax[0].plot([i,i],[r['min'],r['q25']], c='0.4', lw=0.8); ax[0].plot([i,i],[r['q75'],r['max']], c='0.4', lw=0.8)
ax[0].axhline(0.5, ls='--', c='r', lw=1, label='$P=0.5$ (no relation)')
ax[0].set_xticks(range(len(ORDER))); ax[0].set_xticklabels([LAB[f] for f in ORDER], rotation=40, ha='right', fontsize=9)
ax[0].set_ylabel('Pairwise concordance $P$'); ax[0].set_title('(a) Recovery$\\to$forecasting association', fontsize=11); ax[0].legend(fontsize=8)
xs = np.arange(len(QS))
for f in ORDER:
    gg = q2[q2.fk==f].set_index('quartile')
    ax[1].plot(xs, [gg.loc[q,'median'] for q in QS], marker='o', ms=4, c=GC[GROUP[f]], alpha=0.85)
ax[1].axhline(0.5, ls='--', c='r', lw=1)
ax[1].set_xticks(xs); ax[1].set_xticklabels(QS); ax[1].set_xlabel('Volume quartile')
ax[1].set_ylabel('Median concordance $P$'); ax[1].set_title('(b) By volume quartile', fontsize=11)
ax[1].legend(handles=[Patch(fc=GC[k], label=k) for k in ['naive','intermittent','lag-ML','deep','foundation']], fontsize=7.5)
plt.tight_layout(); plt.savefig(f'{OUT}/fig_rq2_concordance.png', dpi=200, bbox_inches='tight'); plt.close()
print('saved fig_rq2_concordance')

# ---------------------------------------------------------------- RQ4 crossover
st = pd.read_parquet(f'{RES}/hpo_stratified_quartile.parquet')
FAM = {'Naive':['global_mean','dow_mean','ma_k56'],'Intermittent':['croston','sba','tsb'],
       'Lag-ML':['lgb_m5lags','mlp_m5lags'],'TFT':['tft'],'Foundation':['chronos_bolt','timesfm']}
FAMC = {'Naive':'#4c72b0','Intermittent':'#55a868','Lag-ML':'#c44e52','TFT':'#8172b3','Foundation':'#ccb974'}
fig, ax = plt.subplots(figsize=(7.6, 5.2)); xs = np.arange(len(QS))
for fam, fcs in FAM.items():
    ys = []
    for q in QS:
        sub = st[(st.quartile==q) & (st.forecaster.isin(fcs))]
        ys.append(sub.wape_h_med.min() if len(sub) else np.nan)
    ax.plot(xs, ys, marker='o', lw=2, label=fam, color=FAMC[fam])
ax.set_xticks(xs); ax.set_xticklabels([f'{q}\n(vol {b})' for q,b in zip(QS,['low','','','high'])])
ax.set_ylabel('Median WAPE (best cell of family)'); ax.set_xlabel('Volume quartile')
ax.set_title('Forecaster-family accuracy across volume regimes', fontsize=12)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUT}/fig_rq4_crossover.png', dpi=200, bbox_inches='tight'); plt.close()
print('saved fig_rq4_crossover')

# ---------------------------------------------------------------- deploy intersection
mat = pd.read_parquet(f'{RES}/hpo_matrix_pareto.parquet')
fr = pd.read_parquet(f'{RES}/friedman_nemenyi_ranks.parquet')[['cell','mean_rank']]
m = mat.merge(fr, on='cell')
def pareto(df,xc,yc):
    x=df[xc].values; y=df[yc].values; k=[]
    for i in range(len(df)):
        d=((x<=x[i])&(y<=y[i])&((x<x[i])|(y<y[i]))); d[i]=False; k.append(not d.any())
    return np.array(k)
m['p_rank']=pareto(m,'mean_rank','abs_wpe_med'); m['p_wape']=m['pareto']
m['both']=m.p_rank & m.p_wape
best=fr.sort_values('mean_rank').iloc[0]['cell']
fig, ax = plt.subplots(figsize=(7.6, 5.6))
ax.scatter(m.mean_rank, m.abs_wpe_med, s=14, c='0.8', label='other cells', zorder=1)
fo=m[m.p_rank & ~m.both]; ax.scatter(fo.mean_rank, fo.abs_wpe_med, s=40, facecolor='none', edgecolor='#c44e52', label='paired-rank frontier only', zorder=2)
bo=m[m.both]; ax.scatter(bo.mean_rank, bo.abs_wpe_med, s=55, c='#55a868', edgecolor='k', lw=0.5, label='doubly Pareto-optimal', zorder=3)
bb=m[m.cell==best]; ax.scatter(bb.mean_rank, bb.abs_wpe_med, s=220, marker='*', c='gold', edgecolor='k', lw=0.8, label='Friedman best', zorder=4)
ax.set_xlabel('Mean rank (lower = better, paired)'); ax.set_ylabel('Median $|\\mathrm{WPE}|$ (bias)')
ax.set_title('Deployment view: paired-rank vs marginal optimality', fontsize=12)
ax.legend(fontsize=8.5, loc='upper right'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUT}/fig_deploy_intersection.png', dpi=200, bbox_inches='tight'); plt.close()
print('saved fig_deploy_intersection')
print(f'doubly-optimal={int(m.both.sum())} paired-frontier={int(m.p_rank.sum())} marginal={int(m.p_wape.sum())}')
