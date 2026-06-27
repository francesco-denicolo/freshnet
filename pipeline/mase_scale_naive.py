"""R2.2 robustness check (MASE / RMSSE). Daily, in-stock granularity (M5-style,
less fragile than sparse hourly). Per series:
  scale_MAE = mean_t |Yd(t)-Yd(t-1)|, scale_MSE = mean_t (Yd(t)-Yd(t-1))^2
over training daily in-stock totals (Yd = sum over in-stock operational hours).
  MASE_s = mean_d |q_d - y_d| / scale_MAE ;  RMSSE_s = sqrt(mean_d (q_d-y_d)^2 / scale_MSE)
on test days 91-97. Saves the per-series scale (mase_scale.parquet) and computes
MASE/RMSSE for the naive and intermittent forecasters (best imputer mediana_glob)."""
import os, functools, time
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)
t0=time.time()
PR=os.path.join(os.path.dirname(__file__),'..'); DATA=os.path.join(PR,'data')
COMP=os.path.join(DATA,'completed_sales_622'); RES=os.path.join(os.path.dirname(__file__),'results')
H0,H1=6,23; NH=H1-H0
def load(fn):
    d=pd.read_parquet(os.path.join(DATA,fn)); d['dt_parsed']=pd.to_datetime(d['dt'])
    return d
tr=pd.concat([load('frn50k_train.parquet'),load('frn50k_eval.parquet')],ignore_index=True)
tr=tr.sort_values(['store_id','product_id','dt_parsed']).reset_index(drop=True)
dates=sorted(tr['dt_parsed'].unique()); d2n={d:i+1 for i,d in enumerate(dates)}
tr['day_num']=tr['dt_parsed'].map(d2n); tr['dow']=tr['dt_parsed'].dt.dayofweek
S=tr[['store_id','product_id']].drop_duplicates().shape[0]; ND=len(tr)//S
sid=tr['store_id'].values.reshape(S,ND)[:,0]; pid=tr['product_id'].values.reshape(S,ND)[:,0]
days=tr['day_num'].values.reshape(S,ND)[0]; dows=tr['dow'].values.reshape(S,ND)
sales=np.array(tr['hours_sale'].tolist(),np.float64)[:,H0:H1].reshape(S,ND,NH)
stock=np.array(tr['hours_stock_status'].tolist(),np.int8)[:,H0:H1].reshape(S,ND,NH)
print(f'{S} series x {ND} days, t={time.time()-t0:.0f}s')

instock=stock==0
daily_obs=np.where(instock,sales,0.0).sum(2)              # (S,ND) in-stock daily total
n_in_day=instock.sum(2)                                    # in-stock hours per day
tr_m=days<=90; te_m=(days>=91)&(days<=97)
# in-sample scale on training daily in-stock totals (consecutive-day diff)
Dtr=daily_obs[:,tr_m]
diff=np.abs(np.diff(Dtr,axis=1)); scale_mae=np.where(diff.sum(1)>0,diff.mean(1),np.nan)
diff2=np.diff(Dtr,axis=1)**2; scale_mse=np.where(diff2.sum(1)>0,diff2.mean(1),np.nan)
pd.DataFrame({'store_id':sid,'product_id':pid,'scale_mae':scale_mae,'scale_mse':scale_mse}).to_parquet(f'{RES}/mase_scale.parquet',index=False)

obs_te=daily_obs[:,te_m]                                   # (S,7) in-stock daily obs
def metrics(pred_daily):                                   # pred_daily (S,7) in-stock daily pred
    ae=np.abs(pred_daily-obs_te).mean(1); se=((pred_daily-obs_te)**2).mean(1)
    mase=np.where(scale_mae>0,ae/scale_mae,np.nan)
    rmsse=np.where(scale_mse>0,np.sqrt(se/scale_mse),np.nan)
    return np.nanmedian(mase),np.nanmedian(rmsse)

def completed(imp):
    df=pd.read_parquet(os.path.join(COMP,f'{imp}.parquet'))
    cs=np.array(df['hours_sale'].tolist(),np.float64)
    km={k:i for i,k in enumerate((df.store_id.astype(str)+'_'+df.product_id.astype(str)+'_'+df.dt).values)}
    out=sales.copy(); fk=(np.repeat(sid,ND).astype(str)+'_'+np.repeat(pid,ND).astype(str)+'_'+tr['dt'].values)
    for j,k in enumerate(fk):
        if k in km: out[j//ND,j%ND]=cs[km[k]]
    return out

# in-stock daily PRED total for a forecaster = sum over in-stock test hours of hourly pred
def instock_daily(pred_hourly):                            # pred_hourly (S,7,NH)
    return np.where(instock[:,te_m,:],pred_hourly,0.0).sum(2)

IMP='mediana_glob'; comp=completed(IMP)
# per-series DoW profile (vectorised): mean of comp over training days of each dow
prof=np.zeros((S,7,NH))
for dw in range(7):
    sel=(dows==dw)&(days<=90)                              # (S,ND)
    num=np.where(sel[:,:,None],comp,0.0).sum(1); cnt=sel.sum(1)[:,None]
    prof[:,dw,:]=np.where(cnt>0,num/np.maximum(cnt,1),0.0)
dow_te=dows[:,te_m]                                        # (S,7)
pred_dow=np.stack([prof[np.arange(S),dow_te[:,j].astype(int),:] for j in range(dow_te.shape[1])],axis=1)
mase_dw,rmsse_dw=metrics(instock_daily(pred_dow))

# SBA intermittent: daily rate x intra-day profile (reuse design)
def sba_rate(D,alpha=0.3):
    Sn,T=D.shape; z=np.zeros(Sn); p=np.ones(Sn); q=np.zeros(Sn); seen=np.zeros(Sn,bool)
    for t in range(T):
        d=D[:,t]; nz=d>0; q+=1
        f=nz&~seen; z[f]=d[f]; p[f]=q[f]; seen[f]=True; q[f]=0
        u=nz&seen&~f; z[u]=alpha*d[u]+(1-alpha)*z[u]; p[u]=alpha*q[u]+(1-alpha)*p[u]; q[u]=0
    return np.where(seen,(1-alpha/2)*z/np.maximum(p,1e-9),0.0)
daily_comp=comp.sum(2); r=sba_rate(daily_comp[:,days<=90])
meanh=comp[:,days<=90,:].mean(1); pr=np.where(meanh.sum(1,keepdims=True)>0,meanh/meanh.sum(1,keepdims=True),1.0/NH)
pred_sba=(r[:,None,None]*pr[:,None,:]).repeat(7,1)
mase_sba,rmsse_sba=metrics(instock_daily(pred_sba))

print('\n=== R2.2 MASE / RMSSE (daily in-stock, median per series) ===')
print(f'  DoW Mean x mediana_glob : MASE={mase_dw:.3f}  RMSSE={rmsse_dw:.3f}')
print(f'  SBA      x mediana_glob : MASE={mase_sba:.3f}  RMSSE={rmsse_sba:.3f}')
print(f'saved mase_scale.parquet | t={time.time()-t0:.0f}s')
