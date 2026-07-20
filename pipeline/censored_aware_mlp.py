"""
censored_aware_mlp.py — MAJOR 1 (generalised): a censored-aware DIRECT forecaster.
MLP-M5 trained on the raw CENSORED series (no imputer) with a pinball (quantile) loss,
swept over tau, tracing the accuracy-bias trade-off without any imputation stage.
Usage:  [HPO_VARIANT=1] TAUS="0.5,0.7,0.8,0.9" [SMOKE=1] python censored_aware_mlp.py
Outputs censored_mlp_m5_q<tau>_test_per_series.parquet per tau.
"""
import os, gc, time, functools
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
TAUS = [float(x) for x in os.getenv('TAUS', '0.5,0.7,0.8,0.9').split(',')]
SMOKE = os.getenv('SMOKE') == '1'
H_START, H_END = 6, 23; N_HOURS = H_END - H_START
CONT_FEATURES = ['discount','avg_temperature','avg_humidity','precpt','avg_wind_level','holiday_flag','activity_flag']
LAG_NAMES = ['lag_1d','lag_7d','lag_14d','rmean_7d','rmean_14d','rstd_7d',
             'lag_dow','rmean_dow','daily_total_lag1','daily_total_rmean7','momentum_1d_7d']
BATCH_SIZE=4096; LR=1e-3; MAX_EPOCHS=100; PATIENCE=10; HIDDEN=[128,64]
EMB_DIMS={'store_id':32,'product_id':32,'city_id':8,'dow':4}
CARDINALITIES={'store_id':898,'product_id':865,'city_id':18,'dow':7}
WEIGHT_DECAY=0.0
if os.getenv('HPO_VARIANT') == '1':
    import json
    with open(os.path.join(RESULTS_DIR, 'hpo_mlp_best.json')) as f:
        hpo = json.load(f)['best_params']
    HIDDEN=json.loads(hpo['hidden']); LR=float(hpo['lr']); BATCH_SIZE=int(hpo['batch_size'])
    WEIGHT_DECAY=float(hpo['weight_decay']); emb_scale=float(hpo['emb_scale'])
    EMB_DIMS={k:max(2,int(v*emb_scale)) for k,v in EMB_DIMS.items()}
if SMOKE: MAX_EPOCHS=2; PATIENCE=1
SUFFIX='_hpo' if os.getenv('HPO_VARIANT') == '1' else ''
print(f'=== Censored-aware MLP-M5 (no imputer, pinball loss) | taus={TAUS} | device={DEVICE} ===')

# ---- load raw CENSORED series (no imputation) ----
print('\n1. Loading (raw censored series)...')
df_train_hf = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df_eval = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_eval.parquet'))
df_train_hf['dt_parsed']=pd.to_datetime(df_train_hf['dt']); df_eval['dt_parsed']=pd.to_datetime(df_eval['dt'])
df_full=pd.concat([df_train_hf,df_eval],ignore_index=True)
df_full=df_full.sort_values(['store_id','product_id','dt_parsed']).reset_index(drop=True)
all_dates=sorted(df_full['dt_parsed'].unique()); date_to_day={d:i+1 for i,d in enumerate(all_dates)}
df_full['day_num']=df_full['dt_parsed'].map(date_to_day); df_full['dow']=df_full['dt_parsed'].dt.dayofweek
sales_orig=np.array(df_full['hours_sale'].tolist(),dtype=np.float32)[:,H_START:H_END]
stock_orig=np.array(df_full['hours_stock_status'].tolist(),dtype=np.float32)[:,H_START:H_END]
del df_train_hf,df_eval

print('  Building series cache (censored series)...')
sc={}
for (sid,pid),grp in df_full.groupby(['store_id','product_id'],sort=False):
    gs=grp.sort_values('day_num'); idx=gs.index.values
    sc[(sid,pid)]={'days':gs['day_num'].values,'dows':gs['dow'].values,
                   'sales_c':sales_orig[idx],'sales_o':sales_orig[idx],'stock':stock_orig[idx],
                   'city_id':gs['city_id'].values[0],'conts':gs[CONT_FEATURES].values.astype(np.float32)}
print(f'  {len(sc):,} series')
del df_full,sales_orig,stock_orig; gc.collect()

def clags(as_, ad, dw, K):
    z=np.float32; NH=N_HOURS
    L={n:np.full(NH,np.nan,dtype=z) for n in LAG_NAMES}
    if K==0: return L
    L['lag_1d']=as_[-1]
    if K>=7: L['lag_7d']=as_[-7]
    if K>=14: L['lag_14d']=as_[-14]
    if K>=7: L['rmean_7d']=as_[-7:].mean(0)
    if K>=14: L['rmean_14d']=as_[-14:].mean(0)
    if K>=2: L['rstd_7d']=as_[-min(7,K):].std(0)
    sd=ad==dw
    if sd.any(): ds=as_[sd]; L['lag_dow']=ds[-1]; L['rmean_dow']=ds.mean(0)
    dt=as_.sum(1); L['daily_total_lag1']=np.full(NH,dt[-1],dtype=z)
    if K>=7: L['daily_total_rmean7']=np.full(NH,dt[-7:].mean(),dtype=z)
    r,l=L['rmean_7d'],L['lag_1d']
    if not np.isnan(r).all():
        v=(~np.isnan(l))&(~np.isnan(r))&(r>0)
        if v.any(): m=np.full(NH,np.nan,dtype=z); m[v]=l[v]/r[v]; L['momentum_1d_7d']=m
    return L

def build_ds(split, cm=None, cs_=None, lm=None, ls=None):
    d0,d1=(2,83) if split=='train' else ((84,90) if split=='val' else (91,97))
    cl,col,ll,tl,sl,si,pi=[],[],[],[],[],[],[]
    for (sid,pid),sd in sc.items():
        days,dows,scc,so,stk=sd['days'],sd['dows'],sd['sales_c'],sd['sales_o'],sd['stock']
        ci,co=sd['city_id'],sd['conts']
        for idx in range(len(days)):
            d=days[idx]
            if d<d0 or d>d1: continue
            cl.append([sid,pid,ci,dows[idx]]); col.append(co[idx])
            tl.append(so[idx]); sl.append(stk[idx]); si.append(sid); pi.append(pid)
            am=days<=(d-1 if split=='train' else (83 if split=='val' else 90)); K=int(am.sum())
            ld=clags(scc[am],dows[am],dows[idx],K) if K>0 else {n:np.full(N_HOURS,np.nan,dtype=np.float32) for n in LAG_NAMES}
            fa,masks=[],np.zeros(11,dtype=np.float32)
            for fi,n in enumerate(LAG_NAMES):
                arr=ld[n]
                if not np.isnan(arr).all(): masks[fi]=1.0; fa.append(np.where(np.isnan(arr),0,arr).astype(np.float32))
                else: fa.append(np.zeros(N_HOURS,dtype=np.float32))
            fa.append(masks); ll.append(np.concatenate(fa))
    ca=np.array(cl,dtype=np.int64); coa=np.array(col,dtype=np.float32)
    ta=np.array(tl,dtype=np.float32); sa=np.array(sl,dtype=np.float32); la=np.array(ll,dtype=np.float32)
    if cm is None: cm=coa.mean(0); cs_=coa.std(0); cs_[cs_<1e-8]=1.0
    coa=(coa-cm)/cs_
    if lm is None: lm=la.mean(0); ls=la.std(0); ls[ls<1e-8]=1.0
    la=(la-lm)/ls
    return {'cat':ca,'cont':coa,'lags':la,'targets':ta,'stock':sa,
            'store_ids':np.array(si,dtype=np.int64),'product_ids':np.array(pi,dtype=np.int64),
            'cont_mean':cm,'cont_std':cs_,'lag_mean':lm,'lag_std':ls}

print('\n2. Building datasets (once)...')
t0=time.time()
tr=build_ds('train'); va=build_ds('val',tr['cont_mean'],tr['cont_std'],tr['lag_mean'],tr['lag_std'])
te=build_ds('test',tr['cont_mean'],tr['cont_std'],tr['lag_mean'],tr['lag_std'])
del sc; gc.collect()
print(f'  Train {len(tr["targets"]):,} Val {len(va["targets"]):,} Test {len(te["targets"]):,} — built in {time.time()-t0:.0f}s')

class DS2(Dataset):
    def __init__(s,c,co,l,t): s.c,s.co,s.l,s.t=[torch.from_numpy(x) for x in (c,co,l,t)]
    def __len__(s): return len(s.t)
    def __getitem__(s,i): return s.c[i],s.co[i],s.l[i],s.t[i]
class MLP(nn.Module):
    def __init__(s,nc,nl):
        super().__init__()
        s.embs=nn.ModuleDict({n:nn.Embedding(CARDINALITIES[n],EMB_DIMS[n]) for n in EMB_DIMS})
        s.names=['store_id','product_id','city_id','dow']; inp=sum(EMB_DIMS.values())+nc+nl; layers=[]
        for h in HIDDEN: layers+=[nn.Linear(inp,h),nn.ReLU()]; inp=h
        layers+=[nn.Linear(inp,N_HOURS),nn.Softplus()]; s.mlp=nn.Sequential(*layers)
    def forward(s,cat,cont,lags):
        e=[s.embs[n](cat[:,i]) for i,n in enumerate(s.names)]
        return s.mlp(torch.cat(e+[cont,lags],dim=1))

def pinball(pred,target,tau):
    err=target-pred
    return torch.maximum(tau*err,(tau-1.0)*err).mean()

ds=DS2(tr['cat'],tr['cont'],tr['lags'],tr['targets']); nc,nl=tr['cont'].shape[1],tr['lags'].shape[1]; del tr; gc.collect()
vi=va['stock']==0
vc=torch.from_numpy(va['cat']).to(DEVICE); vco=torch.from_numpy(va['cont']).to(DEVICE); vl=torch.from_numpy(va['lags']).to(DEVICE)
tc=torch.from_numpy(te['cat']).to(DEVICE); tco=torch.from_numpy(te['cont']).to(DEVICE); tlg=torch.from_numpy(te['lags']).to(DEVICE)
inst=te['stock']==0

def eval_and_save(model,out_path):
    model.eval(); ap=[]
    with torch.no_grad():
        for s in range(0,len(tc),10000):
            e=min(s+10000,len(tc)); ap.append(model(tc[s:e],tco[s:e],tlg[s:e]).cpu().numpy())
    preds=np.concatenate(ap); sm={}
    for i in range(len(te['store_ids'])): sm.setdefault((te['store_ids'][i],te['product_ids'][i]),[]).append(i)
    recs=[]
    for (sid,pid),idxs in sm.items():
        sh,aoh,eh,oh2,sd2,aod,ed,od=0.,0.,0.,0.,0.,0.,0.,0.
        for i in idxs:
            m=inst[i]
            sh+=np.abs(preds[i,m]-te['targets'][i,m]).sum(); aoh+=np.abs(te['targets'][i,m]).sum()
            eh+=(preds[i,m]-te['targets'][i,m]).sum(); oh2+=te['targets'][i,m].sum()
            if m.any():
                pv,ov=preds[i,m].sum(),te['targets'][i,m].sum(); sd2+=abs(pv-ov);aod+=abs(ov);ed+=pv-ov;od+=ov
        recs.append({'store_id':sid,'product_id':pid,'hourly_wape':sh/aoh if aoh>0 else np.nan,
                     'hourly_wpe':eh/oh2 if oh2!=0 else np.nan,'daily_wape':sd2/aod if aod>0 else np.nan,
                     'daily_wpe':ed/od if od!=0 else np.nan})
    ps=pd.DataFrame(recs); ps.to_parquet(out_path,index=False)
    if os.getenv('EXPORT_Q'):
        cell=os.path.basename(out_path).replace('_test_per_series.parquet','')
        qrows=[(sid,pid,j,float(preds[i].sum())) for (sid,pid),idxs in sm.items() for j,i in enumerate(idxs)]
        qdf=pd.DataFrame(qrows,columns=['store_id','product_id','day_idx','q'])
        qdf.to_parquet(os.path.join(RESULTS_DIR,f'newsvendor_q_{cell}.parquet'),index=False)
        print(f'  newsvendor q saved: newsvendor_q_{cell}.parquet (mean q={qdf.q.mean():.3f})')
    return ps['hourly_wape'].dropna().median(), ps['hourly_wpe'].dropna().median()

print('\n3. Quantile (pinball) sweep, censored-aware direct...')
for tau in TAUS:
    tsx=f'{tau:.2f}'.rstrip('0').rstrip('.')
    out_path=os.path.join(RESULTS_DIR,f'censored_mlp_m5_q{tsx}{SUFFIX}_test_per_series.parquet')
    if os.path.exists(out_path) and not os.getenv('EXPORT_Q'): print(f'  tau={tau}: SKIP'); continue
    t1=time.time(); torch.manual_seed(42); np.random.seed(42)
    model=MLP(nc,nl).to(DEVICE); loader=DataLoader(ds,batch_size=BATCH_SIZE,shuffle=True)
    opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    bw,be,bs,ni=float('inf'),0,None,0
    for ep in range(1,MAX_EPOCHS+1):
        model.train()
        for c,co,l,t in loader:
            c,co,l,t=c.to(DEVICE),co.to(DEVICE),l.to(DEVICE),t.to(DEVICE)
            loss=pinball(model(c,co,l),t,tau); opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            ap=[]
            for s in range(0,len(vc),10000):
                e=min(s+10000,len(vc)); ap.append(model(vc[s:e],vco[s:e],vl[s:e]).cpu().numpy())
            vp=np.concatenate(ap)
        w=np.abs(vp[vi]-va['targets'][vi]).sum()/np.abs(va['targets'][vi]).sum()  # select on WAPE
        if w<bw: bw,be=w,ep; bs={k:v.cpu().clone() for k,v in model.state_dict().items()}; ni=0
        else: ni+=1
        if ni>=PATIENCE: break
    if bs: model.load_state_dict(bs); model.to(DEVICE)
    wm,pm=eval_and_save(model,out_path)
    print(f'  tau={tau}: best_ep={be}  WAPE_med={wm:.4f}  WPE_med={pm:.4f}  ({time.time()-t1:.0f}s)')
    del model,loader,opt,bs; gc.collect()
    if DEVICE=='mps': torch.mps.empty_cache()
print('\nDONE censored-aware MLP sweep')
