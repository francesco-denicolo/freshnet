"""Co-adaptation control (referee minor). The paper tunes each forecaster ONCE on
the raw censored series (Sec 3.7.3). A sceptic could argue that basis rewards
configs insensitive to how stock-outs are filled. Direct check: tune MLP-M5 on an
IMPUTED basis (itransformer) instead, then NATIVELY train the 13 imputer cells
with that config and recompute Kendall's W across imputers. If W stays negligible,
the imputer-irrelevance is not an artefact of the tuning basis.

Tractable subsample (SUB series) so it runs unattended; W on a subsample is ample
for a negligibility check. Small random search over the MLP space on itransformer,
then 13 native retrains with the winning config.
"""
import os, sys, gc, time, json, functools, itertools
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
print = functools.partial(print, flush=True)
t0 = time.time()
PR = os.path.join(os.path.dirname(__file__), '..'); DATA = os.path.join(PR, 'data')
COMP = os.path.join(DATA, 'completed_sales_622'); RES = os.path.join(os.path.dirname(__file__), 'results')
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
H0, H1 = 6, 23; NH = H1 - H0
CONT = ['discount','avg_temperature','avg_humidity','precpt','avg_wind_level','holiday_flag','activity_flag']
LAGN = ['lag_1d','lag_7d','lag_14d','rmean_7d','rmean_14d','rstd_7d','lag_dow','rmean_dow','daily_total_lag1','daily_total_rmean7','momentum_1d_7d']
CARD = {'store_id':898,'product_id':865,'city_id':18,'dow':7}; EMB0 = {'store_id':32,'product_id':32,'city_id':8,'dow':4}
SUB = int(os.getenv('SUB', '20000'))
TRAIN_IMP = 'itransformer'
TEST_IMPS = ['itransformer','mediana_glob','media_glob','saits','dlinear','forward_fill','timesnet',
             'lgb','imputeformer','seasonal_naive','linear_interp','media_cond','mediana_cond']
np.random.seed(42); torch.manual_seed(42)

print('1. Load base + subsample...')
dt = pd.read_parquet(f'{DATA}/frn50k_train.parquet'); de = pd.read_parquet(f'{DATA}/frn50k_eval.parquet')
for d in (dt, de): d['dt_parsed'] = pd.to_datetime(d['dt'])
df = pd.concat([dt, de], ignore_index=True).sort_values(['store_id','product_id','dt_parsed']).reset_index(drop=True)
del dt, de
keys_all = df[['store_id','product_id']].drop_duplicates().reset_index(drop=True)
if SUB <= 0 or SUB >= len(keys_all):
    print(f'  using ALL {len(keys_all)} series (SUB={SUB})')
else:
    rng = np.random.RandomState(0); sel = rng.choice(len(keys_all), SUB, replace=False)
    df = df.merge(keys_all.iloc[sel], on=['store_id','product_id']).reset_index(drop=True)
alld = sorted(df['dt_parsed'].unique()); df['day_num'] = df['dt_parsed'].map({d:i+1 for i,d in enumerate(alld)})
df['dow'] = df['dt_parsed'].dt.dayofweek
sales_o = np.array(df['hours_sale'].tolist(), np.float32)[:, H0:H1]
stock_o = np.array(df['hours_stock_status'].tolist(), np.float32)[:, H0:H1]
fk = (df['store_id'].astype(str)+'_'+df['product_id'].astype(str)+'_'+df['dt']).values
print(f'  {len(ksub):,} series  t={time.time()-t0:.0f}s')

def clags(a, ad, dw, K):
    z=np.float32; L={n:np.full(NH,np.nan,z) for n in LAGN}
    if K==0: return L
    L['lag_1d']=a[-1]
    if K>=7: L['lag_7d']=a[-7]; L['rmean_7d']=a[-7:].mean(0)
    if K>=14: L['lag_14d']=a[-14]; L['rmean_14d']=a[-14:].mean(0)
    if K>=2: L['rstd_7d']=a[-min(7,K):].std(0)
    sd=ad==dw
    if sd.any(): ds=a[sd]; L['lag_dow']=ds[-1]; L['rmean_dow']=ds.mean(0)
    tot=a.sum(1); L['daily_total_lag1']=np.full(NH,tot[-1],z)
    if K>=7: L['daily_total_rmean7']=np.full(NH,tot[-7:].mean(),z)
    r,l=L['rmean_7d'],L['lag_1d']
    if not np.isnan(r).all():
        v=(~np.isnan(l))&(~np.isnan(r))&(r>0)
        if v.any(): m=np.full(NH,np.nan,z); m[v]=l[v]/r[v]; L['momentum_1d_7d']=m
    return L

def cache_for(imp):
    c = pd.read_parquet(f'{COMP}/{imp}.parquet'); c['dt_parsed']=pd.to_datetime(c['dt'])
    c = c.sort_values(['store_id','product_id','dt_parsed']).reset_index(drop=True)
    arr = np.array(c['hours_sale'].tolist(), np.float32)
    if arr.shape[1]==24: arr=arr[:,H0:H1]
    km = dict(zip((c['store_id'].astype(str)+'_'+c['product_id'].astype(str)+'_'+c['dt']).values, range(len(c))))
    comp = sales_o.copy()
    for i in range(len(df)):
        k=fk[i]
        if k in km: comp[i]=arr[km[k]]
    sc={}
    for (sid,pid),g in df.groupby(['store_id','product_id'],sort=False):
        gg=g.sort_values('day_num'); idx=gg.index.values
        sc[(sid,pid)]={'days':gg['day_num'].values,'dows':gg['dow'].values,'sales':comp[idx],
                       'city':gg['city_id'].values[0],'conts':gg[CONT].values.astype(np.float32),
                       'stock':stock_o[idx]}
    del comp; gc.collect(); return sc

def build(split, sc):
    d0,d1 = (2,83) if split=='train' else ((84,90) if split=='val' else (91,97))
    cat,con,lag,tgt,stk,sid_,pid_=[],[],[],[],[],[],[]
    for (sid,pid),s in sc.items():
        days=s['days']
        for j,d in enumerate(days):
            if d<d0 or d>d1: continue
            ad = d-1 if split=='train' else (83 if split=='val' else 90)
            am = days<=ad; K=int(am.sum())
            if K==0: continue
            L=clags(s['sales'][am], s['dows'][am], s['dows'][j], K)
            lagv=np.stack([L[n] for n in LAGN]) # (11,17)
            cat.append([sid,pid,s['city'],s['dows'][j]]); con.append(s['conts'][j])
            lag.append(lagv); tgt.append(s['sales'][j]); stk.append(s['stock'][j]); sid_.append(sid); pid_.append(pid)
    cat=np.array(cat,np.int64); con=np.array(con,np.float32); lag=np.array(lag,np.float32)
    tgt=np.array(tgt,np.float32); stk=np.array(stk,np.float32)
    return cat,con,lag,tgt,stk,np.array(sid_),np.array(pid_)

class MLP(nn.Module):
    def __init__(s, nlag, hid, drop, emb):
        super().__init__(); s.names=list(CARD)
        s.embs=nn.ModuleList([nn.Embedding(CARD[n],emb[n]) for n in s.names])
        inp=sum(emb.values())+len(CONT)+nlag
        ly=[];
        for h in hid: ly+=[nn.Linear(inp,h),nn.ReLU(),nn.Dropout(drop)]; inp=h
        ly+=[nn.Linear(inp,NH),nn.Softplus()]; s.mlp=nn.Sequential(*ly)
    def forward(s,c,co,l):
        e=[s.embs[i](c[:,i]) for i in range(len(s.names))]
        return s.mlp(torch.cat(e+[co,l.reshape(l.shape[0],-1)],1))

def normalize(con,lag,cm,cs,lm,ls):
    con=(con-cm)/cs
    lagf=lag.reshape(lag.shape[0],-1); lm2=lm; lagf=np.nan_to_num((lagf-lm2)/ls);
    return con.astype(np.float32), lagf.reshape(lag.shape).astype(np.float32)

def per_series_wape(pr, ye, ste, si, pi):
    """pr,ye,ste: (N,NH); si,pi: (N,). In-stock per-series WAPE."""
    ins=(ste==0).ravel(); sif=np.repeat(si,NH); pif=np.repeat(pi,NH)
    prf=pr.ravel(); yef=ye.ravel()
    d=pd.DataFrame({'s':sif[ins],'p':pif[ins],'ae':np.abs(prf[ins]-yef[ins]),'ao':np.abs(yef[ins])})
    g=d.groupby(['s','p']).sum()
    return (g['ae']/g['ao']).replace([np.inf,-np.inf],np.nan).dropna()

def train_eval(cfg, tr, va_or_te, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    (ctr,cotr,ltr,ytr)=tr; emb={k:max(2,int(v*cfg['emb'])) for k,v in EMB0.items()}
    model=MLP(ltr.shape[1]*ltr.shape[2], cfg['hid'], cfg['drop'], emb).to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg['wd'])
    dsx=TensorDataset(torch.tensor(ctr),torch.tensor(cotr),torch.tensor(ltr),torch.tensor(ytr))
    ld=DataLoader(dsx,batch_size=cfg['bs'],shuffle=True)
    ce,co2,le,ye,ste,si,pi=va_or_te
    ct=torch.tensor(ce).to(DEVICE); cot=torch.tensor(co2).to(DEVICE); lt=torch.tensor(le).to(DEVICE)
    best=1e9; bstate=None; bad=0
    for ep in range(1,31):
        model.train()
        for c,co,l,y in ld:
            c,co,l,y=c.to(DEVICE),co.to(DEVICE),l.to(DEVICE),y.to(DEVICE)
            loss=nn.functional.l1_loss(model(c,co,l),y); opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pr=[]
            for sidx in range(0,len(ct),20000):
                e=min(sidx+20000,len(ct)); pr.append(model(ct[sidx:e],cot[sidx:e],lt[sidx:e]).cpu().numpy())
        pr=np.concatenate(pr); w=per_series_wape(pr,ye,ste,si,pi).median()
        if w<best-1e-4: best=w; bstate={k:v.cpu().clone() for k,v in model.state_dict().items()}; bad=0
        else: bad+=1
        if bad>=6: break
    if bstate: model.load_state_dict(bstate)
    # final per-series WAPE
    model.eval()
    with torch.no_grad():
        pr=[]
        for sidx in range(0,len(ct),20000):
            e=min(sidx+20000,len(ct)); pr.append(model(ct[sidx:e],cot[sidx:e],lt[sidx:e]).cpu().numpy())
    pr=np.concatenate(pr); w=per_series_wape(pr,ye,ste,si,pi)
    return best, w

# ---- 1. tune on itransformer ----
print('2. Build itransformer train/val + small search...')
sc_it = cache_for(TRAIN_IMP)
ctr,cotr,ltr,ytr,_,_,_ = build('train', sc_it)
cm=cotr.mean(0); cs=cotr.std(0)+1e-6; lf=ltr.reshape(ltr.shape[0],-1)
lm=np.nanmean(lf,0); ls=np.nanstd(lf,0)+1e-6
cotr_n,ltr_n = normalize(cotr,ltr,cm,cs,lm,ls)
va = build('val', sc_it); cov_n,lv_n = normalize(va[1],va[2],cm,cs,lm,ls)
VA=(va[0],cov_n,lv_n,va[3],va[4],va[5],va[6])
SPACE=[{'hid':h,'drop':dr,'lr':lr,'bs':bs,'wd':wd,'emb':em}
       for h,dr,lr,bs,em,wd in [([128,64],0.0,3.5e-3,1024,2.0,1e-6),([256,128],0.1,8e-4,1024,1.5,1e-6),
                              ([128,64],0.1,1e-3,4096,1.0,1e-5),([256,128],0.0,2e-3,1024,2.0,1e-6),
                              ([128],0.0,3e-3,1024,1.5,1e-6),([256,128,64],0.1,1e-3,1024,1.5,1e-5),
                              ([64,32],0.0,3e-3,1024,2.0,1e-6),([256],0.05,1.5e-3,1024,1.5,1e-6),
                              ([128,64],0.2,2e-3,1024,2.0,1e-4),([256,128],0.1,5e-4,4096,1.0,1e-6),
                              ([128,128,64],0.1,1e-3,1024,1.5,1e-5),([128,64],0.0,5e-3,4096,2.0,1e-6)]]
best_cfg=None; best_w=1e9
for i,cfg in enumerate(SPACE):
    w,_=train_eval(cfg,(ctr,cotr_n,ltr_n,ytr),VA)
    print(f'  cfg{i} hid={cfg["hid"]} lr={cfg["lr"]:.1e} bs={cfg["bs"]} -> val_WAPE_med={w:.4f}  t={time.time()-t0:.0f}s')
    if w<best_w: best_w=w; best_cfg=cfg
print(f'  BEST(itransformer-tuned): {best_cfg} val={best_w:.4f}')
del ctr,cotr,ltr,ytr,cotr_n,ltr_n; gc.collect()

# ---- 2. native retrain of 13 cells with the itransformer-tuned config ----
print('3. Native retrain of 13 cells with itransformer-tuned config...')
wape_by_imp = {}
for B in TEST_IMPS:
    scB = sc_it if B==TRAIN_IMP else cache_for(B)
    ctrB,cotrB,ltrB,ytrB,_,_,_ = build('train', scB)
    cmB=cotrB.mean(0); csB=cotrB.std(0)+1e-6; lfB=ltrB.reshape(ltrB.shape[0],-1)
    lmB=np.nanmean(lfB,0); lsB=np.nanstd(lfB,0)+1e-6
    cotrB_n,ltrB_n = normalize(cotrB,ltrB,cmB,csB,lmB,lsB)
    teB = build('test', scB); cote_n,lte_n = normalize(teB[1],teB[2],cmB,csB,lmB,lsB)
    TE=(teB[0],cote_n,lte_n,teB[3],teB[4],teB[5],teB[6])
    _, w = train_eval(best_cfg,(ctrB,cotrB_n,ltrB_n,ytrB),TE)
    wape_by_imp[B] = w
    print(f'  {B:16s} test_WAPE_med={w.median():.4f}  t={time.time()-t0:.0f}s')
    del scB,ctrB,cotrB,ltrB,ytrB; gc.collect()

# ---- 3. Kendall W across the 13 cells ----
common = None
for w in wape_by_imp.values():
    common = w.index if common is None else common.intersection(w.index)
M = pd.DataFrame({B: wape_by_imp[B].reindex(common) for B in TEST_IMPS}).dropna()
ranks = M.rank(axis=1); Rbar = ranks.mean(0); n,k = M.shape
S = ((Rbar - (k+1)/2)**2).sum(); W = 12*S / (k**2*(k**3-k)) * n * (k-1)  # Kendall W via mean-rank var
# standard Kendall's W = 12*sum((Rj - n(k+1)/2)^2) / (n^2 k (k^2-1))
Rsum = ranks.sum(0); Wk = 12*((Rsum - n*(k+1)/2)**2).sum() / (n**2 * k * (k**2-1))
print(f'\n=== RESULT ===  n={n} series, k={k} imputers (config tuned on {TRAIN_IMP})')
print('median test WAPE per imputer:')
print(M.median(0).sort_values().to_string())
print(f'\nKendall W (itransformer-tuned config) = {Wk:.4f}   [paper censored-tuned: ~0.03 (MLP-M5)]')
pd.DataFrame({'imputer':list(wape_by_imp),'wape_med':[wape_by_imp[b].median() for b in wape_by_imp]}).to_parquet(f'{RES}/retune_itransformer_W.parquet',index=False)
print(f'DONE  t={time.time()-t0:.0f}s')
