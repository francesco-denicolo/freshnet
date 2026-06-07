"""
04b_baseline_mlp.py — Fase A2: MLP Baseline (ore 6-22)
=======================================================
Serie ristrette a ore 6-22 (17 ore/giorno).
2 varianti: no lags, M5 lags. Output: 17 valori.

Eseguire con: freshnet/bin/python notebooks_622/04b_baseline_mlp.py
"""
import sys, os, gc, time, functools
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

H_START, H_END = 6, 23; N_HOURS = H_END - H_START  # 17
CONT_FEATURES = ['discount','avg_temperature','avg_humidity',
                  'precpt','avg_wind_level','holiday_flag','activity_flag']
LAG_NAMES = ['lag_1d','lag_7d','lag_14d','rmean_7d','rmean_14d','rstd_7d',
             'lag_dow','rmean_dow','daily_total_lag1','daily_total_rmean7','momentum_1d_7d']
N_LAGS_PER_FEAT = N_HOURS  # 17 values per lag feature
N_LAG_DIM_M5 = len(LAG_NAMES) * N_LAGS_PER_FEAT + len(LAG_NAMES)  # 11*17 + 11 = 198

BATCH_SIZE = 4096; LR = 1e-3; MAX_EPOCHS = 100; PATIENCE = 10
HIDDEN = [128, 64]
EMB_DIMS = {'store_id':32,'product_id':32,'city_id':8,'dow':4}
DROPOUT = 0.0; WEIGHT_DECAY = 0.0
CARDINALITIES = {'store_id':898,'product_id':865,'city_id':18,'dow':7}

print('=' * 72)
print('  FASE A2 — MLP BASELINE (ore 6-22)')
print(f'  Output: {N_HOURS} ore, Lag dim (M5): {N_LAG_DIM_M5}')
print('=' * 72)

# Load data
print('\n1. Caricamento dati...')
df_train = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df_eval = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_eval.parquet'))
df_train['dt_parsed'] = pd.to_datetime(df_train['dt'])
df_eval['dt_parsed'] = pd.to_datetime(df_eval['dt'])
df_full = pd.concat([df_train, df_eval], ignore_index=True)
df_full = df_full.sort_values(['store_id','product_id','dt_parsed']).reset_index(drop=True)
all_dates = sorted(df_full['dt_parsed'].unique())
date_to_day = {d: i+1 for i, d in enumerate(all_dates)}
df_full['day_num'] = df_full['dt_parsed'].map(date_to_day)
df_full['dow'] = df_full['dt_parsed'].dt.dayofweek
print(f'  Full: {len(df_full):,} righe, Device: {DEVICE}')
del df_train, df_eval

# SLICE TO 6-22
sales_all = np.array(df_full['hours_sale'].tolist(), dtype=np.float32)[:, H_START:H_END]
stock_all = np.array(df_full['hours_stock_status'].tolist(), dtype=np.float32)[:, H_START:H_END]

# Series cache
print('  Building series cache...')
series_cache = {}
for (sid,pid), grp in df_full.groupby(['store_id','product_id'], sort=False):
    gs = grp.sort_values('day_num')
    idx = gs.index.values
    series_cache[(sid,pid)] = {
        'days':gs['day_num'].values, 'dows':gs['dow'].values,
        'sales':sales_all[idx], 'stock':stock_all[idx],
        'city_id':gs['city_id'].values[0],
        'conts':gs[CONT_FEATURES].values.astype(np.float32)}
print(f'  {len(series_cache):,} serie')
del df_full, sales_all, stock_all; gc.collect()

# Lag computation (17h)
def compute_lags(avail_sales, avail_dows, dow, K):
    z = np.float32; NH = N_HOURS
    L = {n: np.full(NH, np.nan, dtype=z) for n in LAG_NAMES}
    if K==0: return L
    L['lag_1d']=avail_sales[-1]
    if K>=7: L['lag_7d']=avail_sales[-7]
    if K>=14: L['lag_14d']=avail_sales[-14]
    if K>=7: L['rmean_7d']=avail_sales[-7:].mean(0)
    if K>=14: L['rmean_14d']=avail_sales[-14:].mean(0)
    if K>=2: L['rstd_7d']=avail_sales[-min(7,K):].std(0)
    sd=avail_dows==dow
    if sd.any():
        ds=avail_sales[sd]; L['lag_dow']=ds[-1]; L['rmean_dow']=ds.mean(0)
    dt=avail_sales.sum(1)
    L['daily_total_lag1']=np.full(NH,dt[-1],dtype=z)
    if K>=7: L['daily_total_rmean7']=np.full(NH,dt[-7:].mean(),dtype=z)
    r,l=L['rmean_7d'],L['lag_1d']
    if not np.isnan(r).all():
        v=(~np.isnan(l))&(~np.isnan(r))&(r>0)
        if v.any():
            m=np.full(NH,np.nan,dtype=z); m[v]=l[v]/r[v]; L['momentum_1d_7d']=m
    return L

# Dataset builder
def build_dataset(split, use_lags, cont_mean=None, cont_std=None, lag_mean=None, lag_std=None):
    if split=='train': d_min,d_max=2,83
    elif split=='val': d_min,d_max=84,90
    else: d_min,d_max=91,97

    cat_l,cont_l,lag_l,tgt_l,stk_l,sid_l,pid_l=[],[],[],[],[],[],[]
    nd=0
    for (sid,pid),sd in series_cache.items():
        nd+=1
        if nd%10000==0: print(f'      ... {nd:,}/{len(series_cache):,}')
        days,dows,sales,stock=sd['days'],sd['dows'],sd['sales'],sd['stock']
        city,conts=sd['city_id'],sd['conts']
        for idx in range(len(days)):
            d=days[idx]
            if d<d_min or d>d_max: continue
            a_day=d-1 if split=='train' else (83 if split=='val' else 90)
            cat_l.append([sid,pid,city,dows[idx]])
            cont_l.append(conts[idx])
            tgt_l.append(sales[idx]); stk_l.append(stock[idx])
            sid_l.append(sid); pid_l.append(pid)
            if use_lags:
                am=days<=a_day; K=int(am.sum())
                ld=compute_lags(sales[am],dows[am],dows[idx],K) if K>0 \
                    else {n:np.full(N_HOURS,np.nan,dtype=np.float32) for n in LAG_NAMES}
                fa,masks=[],np.zeros(len(LAG_NAMES),dtype=np.float32)
                for fi,n in enumerate(LAG_NAMES):
                    arr=ld[n]
                    if not np.isnan(arr).all():
                        masks[fi]=1.0; fa.append(np.where(np.isnan(arr),0,arr).astype(np.float32))
                    else: fa.append(np.zeros(N_HOURS,dtype=np.float32))
                fa.append(masks); lag_l.append(np.concatenate(fa))
            else: lag_l.append(np.array([],dtype=np.float32))

    cat_arr=np.array(cat_l,dtype=np.int64); cont_arr=np.array(cont_l,dtype=np.float32)
    tgt_arr=np.array(tgt_l,dtype=np.float32); stk_arr=np.array(stk_l,dtype=np.float32)
    lag_arr=np.array(lag_l,dtype=np.float32) if len(lag_l[0])>0 else np.zeros((len(cat_l),0),dtype=np.float32)

    if cont_mean is None:
        cont_mean=cont_arr.mean(0); cont_std=cont_arr.std(0); cont_std[cont_std<1e-8]=1.0
    cont_arr=(cont_arr-cont_mean)/cont_std
    if lag_arr.shape[1]>0:
        if lag_mean is None:
            lag_mean=lag_arr.mean(0); lag_std=lag_arr.std(0); lag_std[lag_std<1e-8]=1.0
        lag_arr=(lag_arr-lag_mean)/lag_std

    return {'cat':cat_arr,'cont':cont_arr,'lags':lag_arr,'targets':tgt_arr,'stock':stk_arr,
            'store_ids':np.array(sid_l,dtype=np.int64),'product_ids':np.array(pid_l,dtype=np.int64),
            'cont_mean':cont_mean,'cont_std':cont_std,'lag_mean':lag_mean,'lag_std':lag_std}

# Model
class DS(Dataset):
    def __init__(s,c,co,l,t):
        s.c,s.co,s.l,s.t=torch.from_numpy(c),torch.from_numpy(co),torch.from_numpy(l),torch.from_numpy(t)
    def __len__(s): return len(s.t)
    def __getitem__(s,i): return s.c[i],s.co[i],s.l[i],s.t[i]

class MLP(nn.Module):
    def __init__(s,n_cont,n_lags):
        super().__init__()
        s.embs=nn.ModuleDict({n:nn.Embedding(CARDINALITIES[n],EMB_DIMS[n]) for n in EMB_DIMS})
        s.names=['store_id','product_id','city_id','dow']
        inp=sum(EMB_DIMS.values())+n_cont+n_lags
        layers=[]
        for h in HIDDEN:
            layers+=[nn.Linear(inp,h),nn.ReLU()]
            if DROPOUT > 0: layers.append(nn.Dropout(DROPOUT))
            inp=h
        layers+=[nn.Linear(inp,N_HOURS),nn.Softplus()]  # OUTPUT = 17
        s.mlp=nn.Sequential(*layers)
    def forward(s,cat,cont,lags):
        e=[s.embs[n](cat[:,i]) for i,n in enumerate(s.names)]
        x=torch.cat(e+[cont],dim=1)
        if lags.shape[1]>0: x=torch.cat([x,lags],dim=1)
        return s.mlp(x)

def predict(model, data):
    model.eval()
    ct=torch.from_numpy(data['cat']).to(DEVICE)
    cot=torch.from_numpy(data['cont']).to(DEVICE)
    lt=torch.from_numpy(data['lags']).to(DEVICE)
    ap=[]
    with torch.no_grad():
        for s in range(0,len(ct),10000):
            e=min(s+10000,len(ct))
            ap.append(model(ct[s:e],cot[s:e],lt[s:e]).cpu().numpy())
    return np.concatenate(ap)

# Eval
def eval_mlp(preds, data, label):
    tgt,stk=data['targets'],data['stock']; sids,pids=data['store_ids'],data['product_ids']
    instock=stk==0
    ph,oh=preds[instock],tgt[instock]
    sae_h,sao_h=np.abs(ph-oh).sum(),np.abs(oh).sum()
    se_h,so_h=(ph-oh).sum(),oh.sum()
    ns=preds.shape[0]
    sae_d,sao_d,se_d,so_d=0.,0.,0.,0.
    for d in range(ns):
        m=instock[d]
        if m.any():
            pv,ov=preds[d,m].sum(),tgt[d,m].sum()
            sae_d+=abs(pv-ov);sao_d+=abs(ov);se_d+=pv-ov;so_d+=ov
    pooled={'hourly_wape':sae_h/sao_h,'hourly_wpe':se_h/so_h,
            'daily_wape':sae_d/sao_d if sao_d>0 else np.nan,'daily_wpe':se_d/so_d if so_d!=0 else np.nan}
    sm={}
    for i in range(ns):
        k=(sids[i],pids[i])
        if k not in sm: sm[k]=[]
        sm[k].append(i)
    recs=[]
    for (sid,pid),idxs in sm.items():
        sh,aoh,eh,oh2=0.,0.,0.,0.
        sd2,aod,ed,od,nvd,ni=0.,0.,0.,0.,0,0
        for i in idxs:
            m=instock[i]; ni+=int(m.sum())
            sh+=np.abs(preds[i,m]-tgt[i,m]).sum(); aoh+=np.abs(tgt[i,m]).sum()
            eh+=(preds[i,m]-tgt[i,m]).sum(); oh2+=tgt[i,m].sum()
            if m.any():
                pv,ov=preds[i,m].sum(),tgt[i,m].sum()
                sd2+=abs(pv-ov);aod+=abs(ov);ed+=pv-ov;od+=ov;nvd+=1
        recs.append({'store_id':sid,'product_id':pid,
                     'hourly_wape':sh/aoh if aoh>0 else np.nan,'hourly_wpe':eh/oh2 if oh2!=0 else np.nan,
                     'daily_wape':sd2/aod if aod>0 else np.nan,'daily_wpe':ed/od if od!=0 else np.nan,
                     'n_hours_instock':ni,'n_days_valid':nvd})
    ps=pd.DataFrame(recs)
    suffix = '_hpo' if os.getenv('HPO_VARIANT') == '1' else ''
    ps.to_parquet(os.path.join(RESULTS_DIR,f'{label}{suffix}_test_per_series.parquet'),index=False)
    med={c:ps[c].dropna().median() for c in ['hourly_wape','hourly_wpe']}
    print(f'  {label}: WAPE_h pool={pooled["hourly_wape"]:.4f}, med={med["hourly_wape"]:.4f}, '
          f'WPE_h={pooled["hourly_wpe"]:.4f}')
    return pooled, med

# ===========================================================================
# Train both variants
# ===========================================================================
for use_lags, label in [(False,'mlp_nolags'),(True,'mlp_m5lags')]:
    vl='M5 lags' if use_lags else 'no lags'
    print(f'\n{"="*72}')
    print(f'  === MLP ({vl}) ===')
    print(f'{"="*72}')
    t0=time.time()

    if os.getenv('HPO_VARIANT') == '1':
        import json
        hpo_file = 'hpo_mlp_best.json' if use_lags else 'hpo_mlp_nolags_best.json'
        with open(os.path.join(RESULTS_DIR, hpo_file)) as f:
            hpo = json.load(f)['best_params']
        HIDDEN = json.loads(hpo['hidden'])
        DROPOUT = float(hpo['dropout'])
        LR = float(hpo['lr'])
        BATCH_SIZE = int(hpo['batch_size'])
        WEIGHT_DECAY = float(hpo['weight_decay'])
        emb_scale = float(hpo['emb_scale'])
        EMB_DIMS = {'store_id':max(2,int(32*emb_scale)),'product_id':max(2,int(32*emb_scale)),
                    'city_id':max(2,int(8*emb_scale)),'dow':max(2,int(4*emb_scale))}
        print(f'  [HPO] hidden={HIDDEN} dropout={DROPOUT} lr={LR:.3e} bs={BATCH_SIZE} '
              f'wd={WEIGHT_DECAY:.2e} emb_scale={emb_scale} -> emb_dims={EMB_DIMS}')

    print('  Building train...')
    tr=build_dataset('train',use_lags)
    print(f'  Train: {len(tr["targets"]):,}, cont={tr["cont"].shape[1]}, lags={tr["lags"].shape[1]}')
    print('  Building val...')
    va=build_dataset('val',use_lags,tr['cont_mean'],tr['cont_std'],tr['lag_mean'],tr['lag_std'])
    print(f'  Val: {len(va["targets"]):,}')
    print('  Building test...')
    te=build_dataset('test',use_lags,tr['cont_mean'],tr['cont_std'],tr['lag_mean'],tr['lag_std'])
    print(f'  Test: {len(te["targets"]):,}')

    # Free series_cache if M5 (memory)
    if use_lags:
        del series_cache; gc.collect()

    model=MLP(tr['cont'].shape[1],tr['lags'].shape[1]).to(DEVICE)
    np_params=sum(p.numel() for p in model.parameters())
    print(f'  Model: {np_params:,} params, output={N_HOURS}')

    ds=DS(tr['cat'],tr['cont'],tr['lags'],tr['targets'])
    loader=DataLoader(ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
    del tr; gc.collect()

    optimizer=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    val_instock=va['stock']==0
    vc=torch.from_numpy(va['cat']).to(DEVICE)
    vco=torch.from_numpy(va['cont']).to(DEVICE)
    vl_t=torch.from_numpy(va['lags']).to(DEVICE)

    best_w,best_ep,best_st=float('inf'),0,None; no_imp=0

    print('  Training...')
    for epoch in range(1,MAX_EPOCHS+1):
        model.train(); tl,nb=0.,0
        for c,co,l,t in loader:
            c,co,l,t=c.to(DEVICE),co.to(DEVICE),l.to(DEVICE),t.to(DEVICE)
            p=model(c,co,l); loss=nn.functional.l1_loss(p,t)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tl+=loss.item(); nb+=1
        model.eval()
        with torch.no_grad():
            ap=[]
            for s in range(0,len(vc),10000):
                e=min(s+10000,len(vc))
                ap.append(model(vc[s:e],vco[s:e],vl_t[s:e]).cpu().numpy())
            vp=np.concatenate(ap)
        w=np.abs(vp[val_instock]-va['targets'][val_instock]).sum()/np.abs(va['targets'][val_instock]).sum()
        print(f'    Epoch {epoch:3d}: loss={tl/nb:.6f}, val_WAPE={w:.6f}')
        if w<best_w:
            best_w,best_ep=w,epoch
            best_st={k:v.cpu().clone() for k,v in model.state_dict().items()}; no_imp=0
        else: no_imp+=1
        if no_imp>=PATIENCE:
            print(f'    Early stop (best={best_ep}, WAPE={best_w:.6f})'); break

    if best_st: model.load_state_dict(best_st)
    model.to(DEVICE)
    torch.save(model.state_dict(),os.path.join(RESULTS_DIR,f'{label}.pt'))

    preds=predict(model,te)
    eval_mlp(preds,te,label)
    print(f'  Time: {time.time()-t0:.0f}s')

    del va,te,model,ds,loader,vc,vco,vl_t; gc.collect()
    if DEVICE=='mps': torch.mps.empty_cache()

print('\n' + '=' * 72)
print('  DONE — 04b_baseline_mlp.py (ore 6-22)')
print('=' * 72)
