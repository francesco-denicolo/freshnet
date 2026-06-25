"""
Maj-2 probe: are stock-outs endogenous to demand? Compare in-stock sales in
hours that precede a same-day stock-out vs in-stock hours not followed by one,
controlling for hour of day. A ratio > 1 indicates endogeneity (shelves empty
on high-demand days), which makes the MNAR-mask WAPE_rec optimistic.
Operational hours 6-22.
"""
import numpy as np, pandas as pd, functools, os
print = functools.partial(print, flush=True)
H0, H1 = 6, 23; NH = H1 - H0
DATA = os.path.join(os.path.dirname(__file__), '..', 'data')

df = pd.read_parquet(os.path.join(DATA, 'frn50k_train.parquet'),
                     columns=['hours_sale', 'hours_stock_status'])
S = np.stack(df['hours_sale'].values).astype(np.float32)[:, H0:H1]
K = np.stack(df['hours_stock_status'].values).astype(np.int8)[:, H0:H1]
instock = (K == 0)

# future_so[:,h] = True if a stock-out occurs at any hour > h on that day
future_so = np.zeros_like(K, bool); acc = np.zeros(K.shape[0], bool)
for h in range(NH - 1, -1, -1):
    future_so[:, h] = acc; acc = acc | (K[:, h] == 1)

A = instock & future_so       # in-stock now, stock-out later today
B = instock & (~future_so)    # in-stock now, no stock-out later today

print('hour | mean_pre-SO(A) | mean_noSO(B) | ratio | nA')
wnum = wden = 0.0
for h in range(NH):
    a = S[:, h][A[:, h]]; b = S[:, h][B[:, h]]
    if len(a) > 50 and len(b) > 50 and b.mean() > 0:
        print(f'  {h+H0:2d} | {a.mean():.4f} | {b.mean():.4f} | {a.mean()/b.mean():.2f} | {len(a):,}')
        wnum += a.mean() * len(a); wden += b.mean() * len(a)
print(f'\nOverall nA-weighted ratio (pre-SO / no-SO in-stock demand): {wnum/wden:.2f}')
