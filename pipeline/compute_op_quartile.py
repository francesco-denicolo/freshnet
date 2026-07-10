"""Per-series volume quartiles on the OPERATIONAL window (hours 6-22), for full
consistency with the rest of the analysis (everything else is on 6-22).
Volume = sum over training days 1-83 of the operational-hour sales. Saved as
series_quartile_op.parquet (store_id, product_id, volume, quartile)."""
import os, functools
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)
RES = os.path.join(os.path.dirname(__file__), 'results')
H0, H1 = 6, 23
df = pd.read_parquet(os.path.join(os.path.dirname(__file__), '..', 'data', 'frn50k_train.parquet'))
df['day_num'] = df.groupby(['store_id', 'product_id']).cumcount() + 1
dtr = df[df.day_num <= 83].copy()
dtr['opvol'] = np.array(dtr['hours_sale'].tolist(), dtype=np.float64)[:, H0:H1].sum(1)
vol = dtr.groupby(['store_id', 'product_id'])['opvol'].sum().reset_index()
vol.columns = ['store_id', 'product_id', 'volume']
vol['quartile'] = pd.qcut(vol['volume'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4']).astype(str)
vol.to_parquet(os.path.join(RES, 'series_quartile_op.parquet'), index=False)
cuts = vol['volume'].quantile([.25, .5, .75]).values
print(f'{len(vol):,} series. operational-volume quartile cuts: '
      f'{cuts[0]:.1f} / {cuts[1]:.1f} / {cuts[2]:.1f}')
print(f'  median={vol.volume.median():.1f} mean={vol.volume.mean():.1f} '
      f'min={vol.volume.min():.1f} max={vol.volume.max():.1f}')
print('  sizes:', vol.quartile.value_counts().sort_index().to_dict())
print('saved series_quartile_op.parquet')
