"""
Generate publication-quality EDA figures (English) for the paper's Data section.
Outputs to MDPI_Overleaf/figures/eda_*.png and prints the exact statistics cited.
"""
import os, functools, numpy as np, pandas as pd
print = functools.partial(print, flush=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 150, 'font.size': 11,
                     'axes.grid': True, 'grid.alpha': 0.3})

DATA = '/Users/utente/Desktop/FreshNetRetail/data'
OUT = '/Users/utente/Desktop/MDPI_Overleaf/figures'
os.makedirs(OUT, exist_ok=True)
H0, H1 = 6, 23  # operational hours 6-22

print('Loading...')
df = pd.read_parquet(os.path.join(DATA, 'frn50k_train.parquet'),
                     columns=['city_id','store_id','product_id','dt','sale_amount',
                              'hours_sale','hours_stock_status','discount','holiday_flag',
                              'avg_temperature'])
df['dt'] = pd.to_datetime(df['dt'])
df = df.sort_values(['store_id','product_id','dt']).reset_index(drop=True)
dates = sorted(df['dt'].unique()); d2d = {d:i+1 for i,d in enumerate(dates)}
df['day'] = df['dt'].map(d2d); df['dow'] = df['dt'].dt.dayofweek

HS = np.stack(df['hours_sale'].values).astype(np.float32)          # (N,24)
SS = np.stack(df['hours_stock_status'].values).astype(np.int8)     # (N,24)
print('arrays:', HS.shape)

# ---------- Fig 1: hourly sales (in-stock vs stockout) + stockout rate, all 24h ----------
hours = np.arange(24)
instock = (SS == 0)
sales_in = np.where(instock, HS, np.nan)
sales_so = np.where(~instock, HS, np.nan)
mean_in = np.nanmean(sales_in, axis=0)
mean_so = np.nanmean(sales_so, axis=0)
so_rate_h = SS.mean(axis=0)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(hours-0.2, mean_in, width=0.4, label='in stock', color='#2c7fb8')
ax[0].bar(hours+0.2, mean_so, width=0.4, label='stock-out', color='#d95f0e')
ax[0].axvspan(5.5, 22.5, color='green', alpha=0.07)
ax[0].set_xlabel('Hour of day'); ax[0].set_ylabel('Mean sales per hour')
ax[0].set_title('(a) Mean hourly sales: in-stock vs stock-out'); ax[0].legend()
ax[1].bar(hours, so_rate_h, color='#d95f0e')
ax[1].axvspan(5.5, 22.5, color='green', alpha=0.07)
ax[1].set_xlabel('Hour of day'); ax[1].set_ylabel('Stock-out rate')
ax[1].set_title('(b) Stock-out rate by hour of day')
plt.tight_layout(); plt.savefig(f'{OUT}/eda_01_hourly_sales_stockout.png'); plt.close()
print('Fig1 done. operational stockout rate (6-22): %.4f' % SS[:,H0:H1].mean())

# ---------- per-series aggregates (operational window for stockout; sale_amount days 1-83 for volume) ----------
df['vol83'] = df['sale_amount']
vol = (df[df['day']<=83].groupby(['store_id','product_id'])['sale_amount'].sum())
# stockout rate per series over operational hours
so_op = SS[:, H0:H1]
df['so_cnt'] = so_op.sum(axis=1); df['so_tot'] = (H1-H0)
sr = df.groupby(['store_id','product_id'])[['so_cnt','so_tot']].sum()
sr['rate'] = sr['so_cnt']/sr['so_tot']
ser = pd.DataFrame({'volume': vol}).join(sr['rate'])
q = np.quantile(ser['volume'], [0.25,0.5,0.75])
print('Volume (total train sales) quartile cuts:', np.round(q,1))
print('Volume min/median/max: %.1f / %.1f / %.1f' % (ser.volume.min(), ser.volume.median(), ser.volume.max()))
print('Per-series stockout rate mean/median: %.3f / %.3f' % (ser.rate.mean(), ser.rate.median()))

# ---------- Fig 2: volume distribution with quartile cuts ----------
fig, ax = plt.subplots(figsize=(7,4))
ax.hist(np.log10(ser['volume'].clip(lower=1)), bins=80, color='#2c7fb8', alpha=0.85)
for qi,lab in zip(q, ['Q1|Q2','Q2|Q3','Q3|Q4']):
    ax.axvline(np.log10(qi), color='k', ls='--', lw=1)
    ax.text(np.log10(qi), ax.get_ylim()[1]*0.92, lab, rotation=90, va='top', ha='right', fontsize=8)
ax.set_xlabel('Per-series total training sales (log$_{10}$)'); ax.set_ylabel('Number of series')
ax.set_title('Per-series volume distribution with quartile boundaries')
plt.tight_layout(); plt.savefig(f'{OUT}/eda_02_volume_distribution.png'); plt.close(); print('Fig2 done.')

# ---------- Fig 3: stockout-rate distribution ----------
fig, ax = plt.subplots(figsize=(7,4))
ax.hist(ser['rate'], bins=60, color='#d95f0e', alpha=0.85)
ax.axvline(ser['rate'].mean(), color='k', ls='--', lw=1, label=f"mean = {ser['rate'].mean():.2f}")
ax.set_xlabel('Per-series stock-out rate (operational hours)'); ax.set_ylabel('Number of series')
ax.set_title('Per-series stock-out rate distribution'); ax.legend()
plt.tight_layout(); plt.savefig(f'{OUT}/eda_03_stockout_distribution.png'); plt.close(); print('Fig3 done.')

# ---------- Fig 4 & 5: heatmaps DoW x hour (sales, stockout) operational ----------
dows = df['dow'].values
sales_op = HS[:, H0:H1]; so_op_full = SS[:, H0:H1]
op_hours = np.arange(H0, H1)
SHsale = np.zeros((7, H1-H0)); SHso = np.zeros((7, H1-H0))
for d in range(7):
    m = dows == d
    SHsale[d] = sales_op[m].mean(axis=0)
    SHso[d] = so_op_full[m].mean(axis=0)
dow_lab = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
for arr, name, title, cmap in [(SHsale,'eda_04_heatmap_sales','Mean sales by day-of-week and hour','viridis'),
                                (SHso,'eda_05_heatmap_stockout','Stock-out rate by day-of-week and hour','inferno')]:
    fig, ax = plt.subplots(figsize=(8,3.8))
    im = ax.imshow(arr, aspect='auto', cmap=cmap)
    ax.set_xticks(range(H1-H0)); ax.set_xticklabels(op_hours)
    ax.set_yticks(range(7)); ax.set_yticklabels(dow_lab)
    ax.set_xlabel('Hour of day'); ax.set_title(title); ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout(); plt.savefig(f'{OUT}/{name}.png'); plt.close()
print('Fig4,5 done.')

# ---------- Fig 6: volume vs stockout (hexbin) ----------
fig, ax = plt.subplots(figsize=(6.5,4.5))
hb = ax.hexbin(np.log10(ser['volume'].clip(lower=1)), ser['rate'], gridsize=45, cmap='Blues', bins='log')
ax.set_xlabel('Per-series total training sales (log$_{10}$)'); ax.set_ylabel('Stock-out rate')
ax.set_title('Per-series volume vs stock-out rate'); fig.colorbar(hb, ax=ax, label='log$_{10}$ count')
plt.tight_layout(); plt.savefig(f'{OUT}/eda_06_volume_vs_stockout.png'); plt.close(); print('Fig6 done.')

# ---------- Fig 7: series per city ----------
spc = df.drop_duplicates(['store_id','product_id']).groupby('city_id').size().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8,4))
ax.bar(range(len(spc)), spc.values, color='#2c7fb8')
ax.set_xticks(range(len(spc))); ax.set_xticklabels(spc.index, rotation=0, fontsize=8)
ax.set_xlabel('City id (sorted by size)'); ax.set_ylabel('Number of series')
ax.set_title('Geographic distribution of series across cities')
ax.text(0.3, spc.values[0]*0.9, f'City {spc.index[0]}: {spc.values[0]:,} ({100*spc.values[0]/50000:.0f}%)', fontsize=9)
plt.tight_layout(); plt.savefig(f'{OUT}/eda_07_series_per_city.png'); plt.close()
print('Fig7 done. top city %d = %d' % (spc.index[0], spc.values[0]))

# ---------- Fig 8: covariate effects (discount, holiday, temperature) on in-stock daily sales ----------
df['instock_sales'] = sales_op.sum(axis=1)  # daily operational sales
# discount bins with explicit clean labels
edges = [0,0.5,0.7,0.85,0.95,1.0]; dlabs = ['0–0.5','0.5–0.7','0.7–0.85','0.85–0.95','0.95–1.0']
db = pd.cut(df['discount'], bins=edges, include_lowest=True, labels=dlabs)
dd = df.groupby(db, observed=True)['instock_sales'].mean()
# temperature bins relabelled as rounded ranges
_, tedges = pd.qcut(df['avg_temperature'], 6, duplicates='drop', retbins=True)
tlabs = [f'{tedges[i]:.0f}–{tedges[i+1]:.0f}' for i in range(len(tedges)-1)]
tb = pd.qcut(df['avg_temperature'], 6, duplicates='drop', labels=tlabs)
tt = df.groupby(tb, observed=True)['instock_sales'].mean()
hol = df.groupby('holiday_flag')['instock_sales'].mean()
fig, ax = plt.subplots(1, 3, figsize=(13,4.2))
ax[0].bar(range(len(dd)), dd.values, color='#2c7fb8'); ax[0].set_xticks(range(len(dd))); ax[0].set_xticklabels(dd.index, rotation=30, ha='right')
ax[0].set_title('(a) Mean daily sales vs discount'); ax[0].set_xlabel('Discount range'); ax[0].set_ylabel('Mean daily sales')
ax[1].bar(['non-holiday','holiday'], hol.values, color=['#2c7fb8','#d95f0e']); ax[1].set_title('(b) Holiday effect'); ax[1].set_ylabel('Mean daily sales')
ax[2].plot(range(len(tt)), tt.values, 'o-', color='#2c7fb8'); ax[2].set_xticks(range(len(tt))); ax[2].set_xticklabels(tt.index, rotation=30, ha='right')
ax[2].set_title('(c) Mean daily sales vs temperature'); ax[2].set_xlabel('Temperature range (°C)')
plt.tight_layout(); plt.savefig(f'{OUT}/eda_08_covariates.png'); plt.close()
print('Fig8 done.')
print('Discount effect (low vs high):', round(dd.iloc[0],4), 'vs', round(dd.iloc[-1],4))
print('Holiday effect:', round(hol.get(0,np.nan),4), 'vs', round(hol.get(1,np.nan),4))
print('ALL FIGURES SAVED TO', OUT)
