"""Tier-2 mechanism control (cross-application). Train ONE MLP-M5 on a single
imputer (itransformer) and, without retraining, run it on the test-time lag
features built from EACH of the 13 imputers' completed series (normalised with
the training imputer's own statistics). Per-series in-stock WAPE (median) tells
us which model-level mechanism produces the imputer-irrelevance:
  * if model_A(features_B) ~ the native cell B for all B  -> intrinsic
    insensitivity (the trained mapping ignores the imputer-induced feature shift);
  * if it is good only near B = A and degrades elsewhere   -> re-calibration
    (each cell's own training is what makes the cells agree).
"""
import os, sys, gc, time, json, functools
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
print = functools.partial(print, flush=True)
t0 = time.time()
PR = os.path.join(os.path.dirname(__file__), '..'); DATA = os.path.join(PR, 'data')
COMP = os.path.join(DATA, 'completed_sales_622'); RES = os.path.join(os.path.dirname(__file__), 'results')
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
H_START, H_END = 6, 23; N_HOURS = H_END - H_START
CONT_FEATURES = ['discount','avg_temperature','avg_humidity','precpt','avg_wind_level','holiday_flag','activity_flag']
LAG_NAMES = ['lag_1d','lag_7d','lag_14d','rmean_7d','rmean_14d','rstd_7d','lag_dow','rmean_dow','daily_total_lag1','daily_total_rmean7','momentum_1d_7d']
EMB_DIMS = {'store_id':32,'product_id':32,'city_id':8,'dow':4}; CARDINALITIES = {'store_id':898,'product_id':865,'city_id':18,'dow':7}
BATCH_SIZE, LR, MAX_EPOCHS, PATIENCE, HIDDEN, WEIGHT_DECAY, DROPOUT = 4096, 1e-3, 100, 10, [128,64], 0.0, 0.0
with open(f'{RES}/hpo_mlp_best.json') as f: hpo = json.load(f)['best_params']
HIDDEN=json.loads(hpo['hidden']); DROPOUT=float(hpo['dropout']); LR=float(hpo['lr']); BATCH_SIZE=int(hpo['batch_size'])
WEIGHT_DECAY=float(hpo['weight_decay']); EMB_DIMS={k:max(2,int(v*float(hpo['emb_scale']))) for k,v in EMB_DIMS.items()}
np.random.seed(42); torch.manual_seed(42)
TRAIN_IMP = 'itransformer'
TEST_IMPS = ['itransformer','mediana_glob','media_glob','saits','dlinear','forward_fill','timesnet',
             'lgb','imputeformer','seasonal_naive','linear_interp','media_cond','mediana_cond']

print('1. Loading base data...')
dt = pd.read_parquet(f'{DATA}/frn50k_train.parquet'); de = pd.read_parquet(f'{DATA}/frn50k_eval.parquet')
for d in (dt, de): d['dt_parsed'] = pd.to_datetime(d['dt'])
df = pd.concat([dt, de], ignore_index=True).sort_values(['store_id','product_id','dt_parsed']).reset_index(drop=True)
del dt, de
alld = sorted(df['dt_parsed'].unique()); df['day_num'] = df['dt_parsed'].map({d:i+1 for i,d in enumerate(alld)})
df['dow'] = df['dt_parsed'].dt.dayofweek
sales_orig = np.array(df['hours_sale'].tolist(), np.float32)[:, H_START:H_END]
stock_orig = np.array(df['hours_stock_status'].tolist(), np.float32)[:, H_START:H_END]
full_keys = (df['store_id'].astype(str)+'_'+df['product_id'].astype(str)+'_'+df['dt']).values

def aligned_completed(imp):
    cs = pd.read_parquet(f'{COMP}/{imp}.parquet')
    arr = np.array(cs['hours_sale'].tolist(), np.float32)
    kmap = pd.Series(np.arange(len(cs)), index=(cs['store_id'].astype(str)+'_'+cs['product_id'].astype(str)+'_'+cs['dt']).values)
    idx = pd.Series(full_keys).map(kmap)
    out = sales_orig.copy(); valid = idx.notna().values
    out[valid] = arr[idx[valid].astype(int).values]
    return out

print('  building series cache (static fields)...')
sc = {}
for (sid,pid), grp in df.groupby(['store_id','product_id'], sort=False):
    gs = grp.sort_values('day_num'); idx = gs.index.values
    sc[(sid,pid)] = {'days':gs['day_num'].values,'dows':gs['dow'].values,'idx':idx,
                     'sales_o':sales_orig[idx],'stock':stock_orig[idx],
                     'city_id':gs['city_id'].values[0],'conts':gs[CONT_FEATURES].values.astype(np.float32),
                     'sales_c':None}
keys = list(sc.keys()); print(f'  {len(sc):,} series, t={time.time()-t0:.0f}s')

def set_completed(imp):
    cf = aligned_completed(imp)
    for k in keys: sc[k]['sales_c'] = cf[sc[k]['idx']]

def clags(as_, ad, dw, K):
    z=np.float32; L={n:np.full(N_HOURS,np.nan,z) for n in LAG_NAMES}
    if K==0: return L
    L['lag_1d']=as_[-1]
    if K>=7: L['lag_7d']=as_[-7]; L['rmean_7d']=as_[-7:].mean(0)
    if K>=14: L['lag_14d']=as_[-14]; L['rmean_14d']=as_[-14:].mean(0)
    if K>=2: L['rstd_7d']=as_[-min(7,K):].std(0)
    sd=ad==dw
    if sd.any(): ds=as_[sd]; L['lag_dow']=ds[-1]; L['rmean_dow']=ds.mean(0)
    d=as_.sum(1); L['daily_total_lag1']=np.full(N_HOURS,d[-1],z)
    if K>=7: L['daily_total_rmean7']=np.full(N_HOURS,d[-7:].mean(),z)
    r,l=L['rmean_7d'],L['lag_1d']
    if not np.isnan(r).all():
        v=(~np.isnan(l))&(~np.isnan(r))&(r>0)
        if v.any(): m=np.full(N_HOURS,np.nan,z); m[v]=l[v]/r[v]; L['momentum_1d_7d']=m
    return L

def build_ds(split, cm=None, cs_=None, lm=None, ls=None):
    d0,d1 = {'train':(2,83),'val':(84,90),'test':(91,97)}[split]
    cl,col,ll,tl,sl,si,pi=[],[],[],[],[],[],[]
    for (sid,pid),sd in sc.items():
        days,dows,scs,so,stk=sd['days'],sd['dows'],sd['sales_c'],sd['sales_o'],sd['stock']
        ci,co=sd['city_id'],sd['conts']
        for ix in range(len(days)):
            dd=days[ix]
            if dd<d0 or dd>d1: continue
            ad=dd-1 if split=='train' else (83 if split=='val' else 90)
            cl.append([sid,pid,ci,dows[ix]]); col.append(co[ix]); tl.append(so[ix]); sl.append(stk[ix]); si.append(sid); pi.append(pid)
            am=days<=ad; K=int(am.sum())
            ld=clags(scs[am],dows[am],dows[ix],K) if K>0 else {n:np.full(N_HOURS,np.nan,np.float32) for n in LAG_NAMES}
            fa,mk=[],np.zeros(11,np.float32)
            for fi,n in enumerate(LAG_NAMES):
                a=ld[n]
                if not np.isnan(a).all(): mk[fi]=1.0; fa.append(np.where(np.isnan(a),0,a).astype(np.float32))
                else: fa.append(np.zeros(N_HOURS,np.float32))
            fa.append(mk); ll.append(np.concatenate(fa))
    ca=np.array(cl,np.int64); coa=np.array(col,np.float32); ta=np.array(tl,np.float32); sa=np.array(sl,np.float32); la=np.array(ll,np.float32)
    if cm is None: cm=coa.mean(0); cs_=coa.std(0); cs_[cs_<1e-8]=1.0
    coa=(coa-cm)/cs_
    if lm is None: lm=la.mean(0); ls=la.std(0); ls[ls<1e-8]=1.0
    la=(la-lm)/ls
    return {'cat':ca,'cont':coa,'lags':la,'targets':ta,'stock':sa,
            'store_ids':np.array(si,np.int64),'product_ids':np.array(pi,np.int64),'cm':cm,'cs':cs_,'lm':lm,'ls':ls}

class MLP(nn.Module):
    def __init__(s,nc,nl):
        super().__init__(); s.embs=nn.ModuleDict({n:nn.Embedding(CARDINALITIES[n],EMB_DIMS[n]) for n in EMB_DIMS})
        s.names=['store_id','product_id','city_id','dow']; inp=sum(EMB_DIMS.values())+nc+nl; ly=[]
        for h in HIDDEN: ly+=[nn.Linear(inp,h),nn.ReLU()]; inp=h
        ly+=[nn.Linear(inp,N_HOURS),nn.Softplus()]; s.mlp=nn.Sequential(*ly)
    def forward(s,c,co,l):
        e=[s.embs[n](c[:,i]) for i,n in enumerate(s.names)]; return s.mlp(torch.cat(e+[co,l],1))

print(f'\n2. Train on {TRAIN_IMP}...'); set_completed(TRAIN_IMP)
tr=build_ds('train'); va=build_ds('val',tr['cm'],tr['cs'],tr['lm'],tr['ls'])
print(f'  built {len(tr["targets"]):,}/{len(va["targets"]):,}, t={time.time()-t0:.0f}s')
class DS(Dataset):
    def __init__(s,a,b,c,d): s.a,s.b,s.c,s.d=[torch.from_numpy(x) for x in (a,b,c,d)]
    def __len__(s): return len(s.d)
    def __getitem__(s,i): return s.a[i],s.b[i],s.c[i],s.d[i]
model=MLP(tr['cont'].shape[1],tr['lags'].shape[1]).to(DEVICE)
loader=DataLoader(DS(tr['cat'],tr['cont'],tr['lags'],tr['targets']),batch_size=BATCH_SIZE,shuffle=True)
opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
vi=va['stock']==0; vc=torch.from_numpy(va['cat']).to(DEVICE); vco=torch.from_numpy(va['cont']).to(DEVICE); vl=torch.from_numpy(va['lags']).to(DEVICE)
cm,csd,lm,ls=tr['cm'],tr['cs'],tr['lm'],tr['ls']; bw,bs,ni=1e9,None,0
for ep in range(1,MAX_EPOCHS+1):
    model.train()
    for c,co,l,t in loader:
        c,co,l,t=[x.to(DEVICE) for x in (c,co,l,t)]
        loss=nn.functional.l1_loss(model(c,co,l),t); opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        vp=np.concatenate([model(vc[s:s+10000],vco[s:s+10000],vl[s:s+10000]).cpu().numpy() for s in range(0,len(vc),10000)])
    w=np.abs(vp[vi]-va['targets'][vi]).sum()/np.abs(va['targets'][vi]).sum()
    if w<bw: bw,ni=w,0; bs={k:v.cpu().clone() for k,v in model.state_dict().items()}
    else: ni+=1
    print(f'  ep{ep} valWAPE={w:.5f}')
    if ni>=PATIENCE: break
model.load_state_dict(bs); model.eval()
del tr,va,loader; gc.collect()

def wape_med(te):
    tc=torch.from_numpy(te['cat']).to(DEVICE); tco=torch.from_numpy(te['cont']).to(DEVICE); tlg=torch.from_numpy(te['lags']).to(DEVICE)
    with torch.no_grad():
        pr=np.concatenate([model(tc[s:s+10000],tco[s:s+10000],tlg[s:s+10000]).cpu().numpy() for s in range(0,len(tc),10000)])
    ins=te['stock']==0
    df2=pd.DataFrame({'s':te['store_ids'],'p':te['product_ids'],
                      'ae':np.where(ins,np.abs(pr-te['targets']),0).sum(1),'ao':np.where(ins,np.abs(te['targets']),0).sum(1)})
    g=df2.groupby(['s','p']).sum(); w=np.where(g.ao>0,g.ae/g.ao,np.nan)
    return float(np.nanmedian(w))

print(f'\n3. Cross-apply model[{TRAIN_IMP}] to each imputer\'s test features...')
mat=pd.read_parquet(f'{RES}/hpo_matrix_pareto.parquet').set_index('cell')
res=[]
for B in TEST_IMPS:
    set_completed(B); te=build_ds('test',cm,csd,lm,ls); w=wape_med(te)
    native=float(mat.loc[f'{B}__mlp_m5lags','wape_h_med']) if f'{B}__mlp_m5lags' in mat.index else np.nan
    res.append((B,w,native)); print(f'  features={B:15s} model[{TRAIN_IMP}]_WAPE={w:.4f}  native(model[{B}])={native:.4f}  diff={w-native:+.4f}')
r=pd.DataFrame(res,columns=['features','crossapply_wape','native_wape'])
print(f'\ncross-apply WAPE: min={r.crossapply_wape.min():.4f} max={r.crossapply_wape.max():.4f} spread={r.crossapply_wape.max()-r.crossapply_wape.min():.4f}')
print(f'mean |crossapply - native| = {(r.crossapply_wape-r.native_wape).abs().mean():.4f}')
r.to_parquet(f'{RES}/tier2_crossapply.parquet',index=False); print(f'saved, t={time.time()-t0:.0f}s')
