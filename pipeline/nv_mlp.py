"""
08c_mlp_single.py — MLP M5-lags per un singolo imputer (ore 6-22)
==================================================================
Usage: freshnet/bin/python notebooks_622/08c_mlp_single.py <imputer_key>
"""
import sys, os, gc, time, functools
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
COMPLETED_DIR = os.path.join(DATA_DIR, 'completed_sales_622')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

H_START, H_END = 6, 23; N_HOURS = H_END - H_START
CONT_FEATURES = ['discount','avg_temperature','avg_humidity',
                  'precpt','avg_wind_level','holiday_flag','activity_flag']
LAG_NAMES = ['lag_1d','lag_7d','lag_14d','rmean_7d','rmean_14d','rstd_7d',
             'lag_dow','rmean_dow','daily_total_lag1','daily_total_rmean7','momentum_1d_7d']
BATCH_SIZE=4096; LR=1e-3; MAX_EPOCHS=100; PATIENCE=10; HIDDEN=[128,64]
EMB_DIMS={'store_id':32,'product_id':32,'city_id':8,'dow':4}
CARDINALITIES={'store_id':898,'product_id':865,'city_id':18,'dow':7}
WEIGHT_DECAY=0.0; DROPOUT=0.0

if os.getenv('HPO_VARIANT') == '1':
    import json
    with open(os.path.join(RESULTS_DIR, 'hpo_mlp_best.json')) as f:
        hpo = json.load(f)['best_params']
    HIDDEN = json.loads(hpo['hidden'])
    DROPOUT = float(hpo['dropout'])
    LR = float(hpo['lr'])
    BATCH_SIZE = int(hpo['batch_size'])
    WEIGHT_DECAY = float(hpo['weight_decay'])
    emb_scale = float(hpo['emb_scale'])
    EMB_DIMS = {k: max(2, int(v * emb_scale)) for k, v in EMB_DIMS.items()}
    print(f'[HPO] hidden={HIDDEN} dropout={DROPOUT} lr={LR:.3e} bs={BATCH_SIZE} '
          f'wd={WEIGHT_DECAY:.2e} emb_scale={emb_scale} -> emb_dims={EMB_DIMS}')

IMP_KEY = sys.argv[1]
IMP_LABELS = {'media_cond':'Media condizionata','media_glob':'Media globale',
              'mediana_cond':'Mediana condizionata','mediana_glob':'Mediana globale',
              'lgb':'LGB imputer',
              'dlinear':'DLinear',
              'forward_fill':'Forward Fill',
              'seasonal_naive':'Seasonal Naive',
              'linear_interp':'Linear Interp',
              'saits':'SAITS',
              'itransformer':'iTransformer',
              'timesnet':'TimesNet',
              'csdi':'CSDI',
              'imputeformer':'ImputeFormer'}
cell_key = f'{IMP_KEY}__mlp_m5lags' + ('_hpo' if os.getenv('HPO_VARIANT') == '1' else '')
out_path = os.path.join(RESULTS_DIR, f'newsvendor_q_{cell_key}.parquet')
if os.path.exists(out_path): print(f'SKIP: {out_path} exists'); sys.exit(0)

print(f'=== MLP M5 × {IMP_LABELS[IMP_KEY]} (ore 6-22) ===')

# Load data
print('\n1. Loading...')
df_train_hf = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df_eval = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_eval.parquet'))
df_train_hf['dt_parsed'] = pd.to_datetime(df_train_hf['dt'])
df_eval['dt_parsed'] = pd.to_datetime(df_eval['dt'])
df_full = pd.concat([df_train_hf, df_eval], ignore_index=True)
df_full = df_full.sort_values(['store_id','product_id','dt_parsed']).reset_index(drop=True)
all_dates = sorted(df_full['dt_parsed'].unique())
date_to_day = {d: i+1 for i, d in enumerate(all_dates)}
df_full['day_num'] = df_full['dt_parsed'].map(date_to_day)
df_full['dow'] = df_full['dt_parsed'].dt.dayofweek
sales_orig = np.array(df_full['hours_sale'].tolist(), dtype=np.float32)[:, H_START:H_END]
stock_orig = np.array(df_full['hours_stock_status'].tolist(), dtype=np.float32)[:, H_START:H_END]
del df_train_hf, df_eval

# Align completed_sales
df_cs = pd.read_parquet(os.path.join(COMPLETED_DIR, f'{IMP_KEY}.parquet'))
cs_sales = np.array(df_cs['hours_sale'].tolist(), dtype=np.float32)
completed_full = sales_orig.copy()
cs_keys = (df_cs['store_id'].astype(str)+'_'+df_cs['product_id'].astype(str)+'_'+df_cs['dt']).values
full_keys = (df_full['store_id'].astype(str)+'_'+df_full['product_id'].astype(str)+'_'+df_full['dt']).values
km = dict(zip(cs_keys, range(len(df_cs))))
for i in range(len(df_full)):
    k = full_keys[i]
    if k in km: completed_full[i] = cs_sales[km[k]]
del df_cs, cs_sales, cs_keys, full_keys, km; gc.collect()

# Series cache
print('  Building series cache...')
sc = {}
for (sid,pid), grp in df_full.groupby(['store_id','product_id'], sort=False):
    gs = grp.sort_values('day_num'); idx = gs.index.values
    sc[(sid,pid)] = {'days':gs['day_num'].values,'dows':gs['dow'].values,
                     'sales_c':completed_full[idx],'sales_o':sales_orig[idx],
                     'stock':stock_orig[idx],'city_id':gs['city_id'].values[0],
                     'conts':gs[CONT_FEATURES].values.astype(np.float32)}
print(f'  {len(sc):,} series')
if os.getenv('SMOKE')=='1':
    _k=list(sc.keys())[:500]; sc={k:sc[k] for k in _k}; print(f'  SMOKE: trimmed to {len(sc)} series')
del df_full, sales_orig, stock_orig, completed_full; gc.collect()

# Lags
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

def build_ds(split, cm=None, cs=None, lm=None, ls=None):
    if split=='train': d0,d1=2,83
    elif split=='val': d0,d1=84,90
    else: d0,d1=91,97
    cl,col,ll,tl,sl,si,pi=[],[],[],[],[],[],[]
    nd=0
    for (sid,pid),sd in sc.items():
        nd+=1
        if nd%10000==0: print(f'    ... {nd:,}/{len(sc):,}')
        days,dows,sc_,so,stk=sd['days'],sd['dows'],sd['sales_c'],sd['sales_o'],sd['stock']
        ci,co=sd['city_id'],sd['conts']
        for idx in range(len(days)):
            d=days[idx]
            if d<d0 or d>d1: continue
            ad=d-1 if split=='train' else (83 if split=='val' else 90)
            cl.append([sid,pid,ci,dows[idx]]); col.append(co[idx])
            tl.append(so[idx]); sl.append(stk[idx]); si.append(sid); pi.append(pid)
            am=days<=ad; K=int(am.sum())
            ld=clags(sc_[am],dows[am],dows[idx],K) if K>0 \
                else {n:np.full(N_HOURS,np.nan,dtype=np.float32) for n in LAG_NAMES}
            fa,masks=[],np.zeros(11,dtype=np.float32)
            for fi,n in enumerate(LAG_NAMES):
                arr=ld[n]
                if not np.isnan(arr).all(): masks[fi]=1.0; fa.append(np.where(np.isnan(arr),0,arr).astype(np.float32))
                else: fa.append(np.zeros(N_HOURS,dtype=np.float32))
            fa.append(masks); ll.append(np.concatenate(fa))
    ca=np.array(cl,dtype=np.int64); coa=np.array(col,dtype=np.float32)
    ta=np.array(tl,dtype=np.float32); sa=np.array(sl,dtype=np.float32)
    la=np.array(ll,dtype=np.float32)
    if cm is None: cm=coa.mean(0); cs=coa.std(0); cs[cs<1e-8]=1.0
    coa=(coa-cm)/cs
    if lm is None: lm=la.mean(0); ls=la.std(0); ls[ls<1e-8]=1.0
    la=(la-lm)/ls
    return {'cat':ca,'cont':coa,'lags':la,'targets':ta,'stock':sa,
            'store_ids':np.array(si,dtype=np.int64),'product_ids':np.array(pi,dtype=np.int64),
            'cont_mean':cm,'cont_std':cs,'lag_mean':lm,'lag_std':ls}

# Build all datasets, then free series_cache
print('\n2. Building datasets...')
t0=time.time()
tr=build_ds('train'); print(f'  Train: {len(tr["targets"]):,}')
va=build_ds('val',tr['cont_mean'],tr['cont_std'],tr['lag_mean'],tr['lag_std']); print(f'  Val: {len(va["targets"]):,}')
te=build_ds('test',tr['cont_mean'],tr['cont_std'],tr['lag_mean'],tr['lag_std']); print(f'  Test: {len(te["targets"]):,}')
del sc; gc.collect()
print(f'  Built in {time.time()-t0:.0f}s')

# Model
class DS2(Dataset):
    def __init__(s,c,co,l,t):
        s.c,s.co,s.l,s.t=torch.from_numpy(c),torch.from_numpy(co),torch.from_numpy(l),torch.from_numpy(t)
    def __len__(s): return len(s.t)
    def __getitem__(s,i): return s.c[i],s.co[i],s.l[i],s.t[i]

class MLP(nn.Module):
    def __init__(s,nc,nl):
        super().__init__()
        s.embs=nn.ModuleDict({n:nn.Embedding(CARDINALITIES[n],EMB_DIMS[n]) for n in EMB_DIMS})
        s.names=['store_id','product_id','city_id','dow']
        inp=sum(EMB_DIMS.values())+nc+nl
        layers=[]
        for h in HIDDEN: layers+=[nn.Linear(inp,h),nn.ReLU()]; inp=h
        layers+=[nn.Linear(inp,N_HOURS),nn.Softplus()]
        s.mlp=nn.Sequential(*layers)
    def forward(s,cat,cont,lags):
        e=[s.embs[n](cat[:,i]) for i,n in enumerate(s.names)]
        x=torch.cat(e+[cont,lags],dim=1); return s.mlp(x)

# Train
print('\n3. Training...')
model=MLP(tr['cont'].shape[1],tr['lags'].shape[1]).to(DEVICE)
print(f'  {sum(p.numel() for p in model.parameters()):,} params')
ds=DS2(tr['cat'],tr['cont'],tr['lags'],tr['targets'])
loader=DataLoader(ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
del tr; gc.collect()

opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
vi=va['stock']==0
vc=torch.from_numpy(va['cat']).to(DEVICE)
vco=torch.from_numpy(va['cont']).to(DEVICE)
vl=torch.from_numpy(va['lags']).to(DEVICE)
bw,be,bs=float('inf'),0,None; ni=0

for ep in range(1,MAX_EPOCHS+1):
    model.train(); tl,nb=0.,0
    for c,co,l,t in loader:
        c,co,l,t=c.to(DEVICE),co.to(DEVICE),l.to(DEVICE),t.to(DEVICE)
        p=model(c,co,l); loss=nn.functional.l1_loss(p,t)
        opt.zero_grad(); loss.backward(); opt.step(); tl+=loss.item(); nb+=1
    model.eval()
    with torch.no_grad():
        ap=[]
        for s in range(0,len(vc),10000):
            e=min(s+10000,len(vc)); ap.append(model(vc[s:e],vco[s:e],vl[s:e]).cpu().numpy())
        vp=np.concatenate(ap)
    w=np.abs(vp[vi]-va['targets'][vi]).sum()/np.abs(va['targets'][vi]).sum()
    print(f'  Epoch {ep:3d}: loss={tl/nb:.6f}, val_WAPE={w:.6f}')
    if w<bw: bw,be=w,ep; bs={k:v.cpu().clone() for k,v in model.state_dict().items()}; ni=0
    else: ni+=1
    if ni>=PATIENCE: print(f'  Early stop (best={be}, WAPE={bw:.6f})'); break

if bs: model.load_state_dict(bs)
model.to(DEVICE)

# Eval test
print('\n4. Test...')
model.eval()
tc=torch.from_numpy(te['cat']).to(DEVICE)
tco=torch.from_numpy(te['cont']).to(DEVICE)
tlg=torch.from_numpy(te['lags']).to(DEVICE)
ap=[]
with torch.no_grad():
    for s in range(0,len(tc),10000):
        e=min(s+10000,len(tc)); ap.append(model(tc[s:e],tco[s:e],tlg[s:e]).cpu().numpy())
preds=np.concatenate(ap)

# NEWSVENDOR: export daily order q(series, test-day) = sum of hourly predictions
qd=pd.DataFrame({'store_id':te['store_ids'],'product_id':te['product_ids'],
                 'q':preds.sum(1).astype(np.float64)})
qd['day_idx']=qd.groupby(['store_id','product_id'],sort=False).cumcount()
qd=qd[['store_id','product_id','day_idx','q']]
qd.to_parquet(out_path,index=False)
print(f'  NEWSVENDOR q saved: {out_path} ({len(qd):,} rows, mean q={qd.q.mean():.3f})')
sys.exit(0)

inst=te['stock']==0
ph,oh=preds[inst],te['targets'][inst]
pooled={'hourly_wape':np.abs(ph-oh).sum()/np.abs(oh).sum(),
        'hourly_wpe':(ph-oh).sum()/oh.sum()}

sm={}
for i in range(len(te['store_ids'])):
    k=(te['store_ids'][i],te['product_ids'][i])
    if k not in sm: sm[k]=[]
    sm[k].append(i)
recs=[]
for (sid,pid),idxs in sm.items():
    sh,aoh,eh,oh2,sd2,aod,ed,od,nvd,nin=0.,0.,0.,0.,0.,0.,0.,0.,0,0
    for i in idxs:
        m=inst[i]; nin+=int(m.sum())
        sh+=np.abs(preds[i,m]-te['targets'][i,m]).sum(); aoh+=np.abs(te['targets'][i,m]).sum()
        eh+=(preds[i,m]-te['targets'][i,m]).sum(); oh2+=te['targets'][i,m].sum()
        if m.any():
            pv,ov=preds[i,m].sum(),te['targets'][i,m].sum()
            sd2+=abs(pv-ov);aod+=abs(ov);ed+=pv-ov;od+=ov;nvd+=1
    recs.append({'store_id':sid,'product_id':pid,
                 'hourly_wape':sh/aoh if aoh>0 else np.nan,'hourly_wpe':eh/oh2 if oh2!=0 else np.nan,
                 'daily_wape':sd2/aod if aod>0 else np.nan,'daily_wpe':ed/od if od!=0 else np.nan})
ps=pd.DataFrame(recs); ps.to_parquet(out_path,index=False)
med={c:ps[c].dropna().median() for c in ['hourly_wape','hourly_wpe']}

print(f'\n  WAPE_h pool={pooled["hourly_wape"]:.4f}, med={med["hourly_wape"]:.4f}')
print(f'  WPE_h pool={pooled["hourly_wpe"]:.4f}')
print(f'  Salvato: {out_path}')
print('DONE')
