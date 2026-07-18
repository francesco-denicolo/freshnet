"""MAJOR 1 figure: censored-aware DIRECT forecaster (no imputer, quantile/pinball
sweep over tau) vs the two-stage matrix on the accuracy-bias plane. Shows that the
imputation stage is unnecessary -- the loss level tau alone spans/dominates the
two-stage frontier. Writes fig_censored_direct.png into the Overleaf figures/ folder."""
import os, glob, re, functools
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), 'results')
OUT = '/Users/utente/Desktop/MDPI_Overleaf/figures/fig_censored_direct.png'

# ---- two-stage matrix cells (median WAPE, median |WPE|) ----
mat = pd.read_parquet(f'{RES}/hpo_matrix_pareto.parquet')[['cell','wape_h_med','abs_wpe_med']]
def pareto(x, y):
    k=[]
    for i in range(len(x)):
        d=((x<=x[i])&(y<=y[i])&((x<x[i])|(y<y[i]))); d[i]=False; k.append(not d.any())
    return np.array(k)
front = pareto(mat.wape_h_med.values, mat.abs_wpe_med.values)

# ---- censored-aware direct cells ----
def load_curve(model):
    rows=[]
    for f in sorted(glob.glob(f'{RES}/censored_{model}_m5_q*_hpo_test_per_series.parquet')):
        tau=float(re.search(r'_q([0-9.]+)_hpo', f).group(1))
        d=pd.read_parquet(f)
        rows.append((tau, d.hourly_wape.dropna().median(), abs(d.hourly_wpe.dropna().median())))
    return pd.DataFrame(rows, columns=['tau','wape','awpe']).sort_values('tau')
lgb = load_curve('lgb'); mlp = load_curve('mlp')

fig, ax = plt.subplots(figsize=(7.4, 5.6))
# two-stage cloud + frontier
ax.scatter(mat.wape_h_med[~front], mat.abs_wpe_med[~front], s=16, c='0.78', label='two-stage cells (dominated)', zorder=1)
fr = mat[front].sort_values('wape_h_med')
ax.plot(fr.wape_h_med, fr.abs_wpe_med, '-o', color='#4c72b0', ms=5, lw=1.4, label='two-stage Pareto frontier', zorder=2)
# censored-aware direct curves
for df, col, name in [(lgb,'#c44e52','censored-aware LGB-M5 (no imputer)'),
                      (mlp,'#55a868','censored-aware MLP-M5 (no imputer)')]:
    ax.plot(df.wape, df.awpe, '-s', color=col, ms=6, lw=1.8, label=name, zorder=3)
    for _,r in df.iterrows():
        ax.annotate(fr'$\tau$={r.tau:g}', (r.wape, r.awpe), fontsize=7, color=col,
                    xytext=(3,3), textcoords='offset points')
ax.set_xlabel('Median WAPE (accuracy; lower better)')
ax.set_ylabel(r'Median $|\mathrm{WPE}|$ (bias; lower better)')
ax.set_title('Censored-aware direct forecasting vs the two-stage matrix', fontsize=12)
ax.legend(fontsize=8.5, loc='upper right'); ax.grid(alpha=0.3)
ax.set_xlim(0.94, 1.42)
plt.tight_layout(); plt.savefig(OUT, dpi=200, bbox_inches='tight'); plt.close()
print('saved', OUT)
print('\ncensored LGB:\n', lgb.to_string(index=False))
print('\ncensored MLP:\n', mlp.to_string(index=False))
print(f'\ntwo-stage best WAPE={mat.wape_h_med.min():.4f} ; frontier cells={int(front.sum())}')
