import warnings; warnings.filterwarnings('ignore')
import time, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import lightgbm as lgb
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

SEED=42; N_FOLDS=5; t0=time.time(); np.random.seed(SEED)
DATA_DIR=Path('data')
train_raw=pd.read_csv(DATA_DIR/'train.csv'); test_raw=pd.read_csv(DATA_DIR/'test.csv')

def add_time(df):
    p=df['timestamp'].str.split(':',expand=True).astype(float); df=df.copy()
    df['hour']=p[0].values; df['minute']=p[1].values
    df['ts_minutes']=df['hour']*60+df['minute']; df['time_slot']=(df['ts_minutes']//15).astype(int)
    df['hour_sin']=np.sin(2*np.pi*df['hour']/24); df['hour_cos']=np.cos(2*np.pi*df['hour']/24)
    df['day_sin']=np.sin(2*np.pi*df['day']/7);    df['day_cos']=np.cos(2*np.pi*df['day']/7)
    return df

train_raw=add_time(train_raw); test_raw=add_time(test_raw)
for df in [train_raw,test_raw]:
    df['miss_RoadType']=df['RoadType'].isna().astype(int)
    df['miss_Temperature']=df['Temperature'].isna().astype(int)
    df['miss_Weather']=df['Weather'].isna().astype(int)

_B32='0123456789bcdefghjkmnpqrstuvwxyz'
def dec(gh):
    lr=[-90.,90.]; nr=[-180.,180.]; il=True
    for c in gh:
        cd=_B32.index(c)
        for b in [16,8,4,2,1]:
            if il: mid=(nr[0]+nr[1])/2; (nr.__setitem__(0,mid) if cd&b else nr.__setitem__(1,mid))
            else:  mid=(lr[0]+lr[1])/2; (lr.__setitem__(0,mid) if cd&b else lr.__setitem__(1,mid))
            il=not il
    return (lr[0]+lr[1])/2,(nr[0]+nr[1])/2
def enc(lat,lon,p=6):
    lr=[-90.,90.]; nr=[-180.,180.]; il=True; bits=0; bit=0; r=''
    while len(r)<p:
        if il: mid=(nr[0]+nr[1])/2; (nr.__setitem__(0,mid) or setattr(type('',(),{}),'x',None)) if lon>=mid else None; bit=(bit<<1)|(1 if lon>=mid else 0); (nr.__setitem__(0,mid) if lon>=mid else nr.__setitem__(1,mid))
        else:  mid=(lr[0]+lr[1])/2; bit=(bit<<1)|(1 if lat>=mid else 0); (lr.__setitem__(0,mid) if lat>=mid else lr.__setitem__(1,mid))
        il=not il; bits+=1
        if bits==5: r+=_B32[bit]; bits=0; bit=0
    return r

# Simpler encode
def encode_geohash(lat, lon, precision=6):
    lat_r=[-90.,90.]; lon_r=[-180.,180.]
    is_lon=True; bits=0; bit=0; result=''
    while len(result)<precision:
        if is_lon:
            mid=(lon_r[0]+lon_r[1])/2
            if lon>=mid: lon_r[0]=mid; bit=(bit<<1)|1
            else: lon_r[1]=mid; bit=bit<<1
        else:
            mid=(lat_r[0]+lat_r[1])/2
            if lat>=mid: lat_r[0]=mid; bit=(bit<<1)|1
            else: lat_r[1]=mid; bit=bit<<1
        is_lon=not is_lon; bits+=1
        if bits==5: result+=_B32[bit]; bits=0; bit=0
    return result

_gc={gh:dec(gh) for gh in pd.concat([train_raw['geohash'],test_raw['geohash']]).unique()}
for df in [train_raw,test_raw]:
    df['lat']=df['geohash'].map(lambda g:_gc[g][0]); df['lon']=df['geohash'].map(lambda g:_gc[g][1])
    df['gh4']=df['geohash'].str[:4]; df['gh5']=df['geohash'].str[:5]

lag_d48=(train_raw[train_raw['day']==48][['geohash','timestamp','demand']].rename(columns={'demand':'demand_lag'}))
train_raw=train_raw.merge(lag_d48,on=['geohash','timestamp'],how='left')
train_raw['demand_lag']=np.where(train_raw['day']==48,np.nan,train_raw['demand_lag'])
test_raw=test_raw.merge(lag_d48,on=['geohash','timestamp'],how='left')

_d48=train_raw[train_raw['day']==48][['geohash','gh5','time_slot','demand']]
_gts=_d48.groupby(['geohash','time_slot'])['demand'].mean()
_g5ts=_d48.groupby(['gh5','time_slot'])['demand'].mean()
_gm=_d48.groupby('geohash')['demand'].mean(); _glm=float(_d48['demand'].mean())

def _smooth(gts,gm,glm,n=96):
    p={}
    for gh,grp in gts.reset_index().groupby('geohash'):
        v=np.full(n,np.nan); v[grp['time_slot'].values]=grp['demand'].values
        f=float(gm.get(gh,glm)); v=np.where(np.isnan(v),f,v)
        s=(np.roll(v,1)+2*v+np.roll(v,-1))/4.0
        for i in range(n): p[(gh,i)]=s[i]
    return p

_gts_s=_smooth(_gts,_gm,_glm)
def _lag_fb(df,mask):
    i1=df.set_index(['geohash','time_slot']).index; i2=df.set_index(['gh5','time_slot']).index
    v1=pd.Series(i1.map(_gts_s),index=df.index); v2=pd.Series(i2.map(_g5ts),index=df.index)
    vm=df['geohash'].map(_gm); fb=v1.fillna(v2).fillna(vm).fillna(_glm)
    lg=df['demand_lag'].copy(); lg[mask&lg.isna()]=fb[mask&lg.isna()]
    df['demand_lag']=lg; df['d48_profile_smooth']=v1.fillna(_glm); return df

train_raw=_lag_fb(train_raw,train_raw['day']!=48); test_raw=_lag_fb(test_raw,test_raw['day']==49)
train_raw.loc[train_raw['day']==48,'d48_profile_smooth']=np.nan

_d48g=train_raw[train_raw['day']==48].groupby('geohash')['demand'].mean()
_d49g=train_raw[train_raw['day']==49].groupby('geohash')['demand'].mean()
_tr=(_d49g/(_d48g+1e-6)); _gtr=float(_tr.mean())
for df in [train_raw,test_raw]: df['geo_day_trend']=df['geohash'].map(_tr).fillna(_gtr)

_gdm=train_raw.groupby('geohash')['demand'].mean(); _ggm=float(_gdm.mean())
def _nbr(gh):
    lat,lon=_gc[gh]; lr=[-90.,90.]; nr=[-180.,180.]; il=True
    for ch in gh:
        cd=_B32.index(ch)
        for b in [16,8,4,2,1]:
            if il: mid=(nr[0]+nr[1])/2; (nr.__setitem__(0,mid) if cd&b else nr.__setitem__(1,mid))
            else: mid=(lr[0]+lr[1])/2; (lr.__setitem__(0,mid) if cd&b else lr.__setitem__(1,mid))
            il=not il
    ls=lr[1]-lr[0]; ms=nr[1]-nr[0]
    ds=[_gdm[encode_geohash(lat+dl*ls,lon+dm*ms,len(gh))] for dl in [-1,0,1] for dm in [-1,0,1]
        if not(dl==0 and dm==0) and encode_geohash(lat+dl*ls,lon+dm*ms,len(gh)) in _gdm]
    return float(np.mean(ds)) if ds else _ggm
_nm={gh:_nbr(gh) for gh in _gc}
for df in [train_raw,test_raw]: df['neighbor_demand_mean']=df['geohash'].map(_nm).fillna(_ggm)

# Safe daytime aggregates (corrected v3 — no window features)
_tts=set(test_raw['timestamp'].unique())
_d48dt=train_raw[(train_raw['day']==48)&train_raw['timestamp'].isin(_tts)]
_ghm=_d48dt.groupby('geohash')['demand'].mean()
_ghs=_d48dt.groupby('geohash')['demand'].std().fillna(0)
_ghx=_d48dt.groupby('geohash')['demand'].max(); _ghdm=float(_ghm.mean())
for df in [train_raw,test_raw]:
    df['d48_daytime_mean']=df['geohash'].map(_ghm).fillna(_ghdm)
    df['d48_daytime_std']=df['geohash'].map(_ghs).fillna(0)
    df['d48_daytime_max']=df['geohash'].map(_ghx).fillna(_ghdm)
    df['lag_x_trend']=df['demand_lag']*df['geo_day_trend']
    df['lag_x_hour_sin']=df['demand_lag']*df['hour_sin']
    df['lag_normalized']=df['demand_lag']/(df['d48_daytime_mean']+1e-6)

y_tr=train_raw['demand'].values; n_tr=len(train_raw)
CAT=['geohash','gh4','gh5','RoadType','LargeVehicles','Landmarks','Weather']
NUM=['day','hour','minute','hour_sin','hour_cos','day_sin','day_cos',
     'NumberofLanes','Temperature','lat','lon','geo_day_trend','neighbor_demand_mean',
     'miss_RoadType','miss_Temperature','miss_Weather','d48_profile_smooth',
     'd48_daytime_mean','d48_daytime_std','d48_daytime_max',
     'lag_x_trend','lag_x_hour_sin','lag_normalized']
CFE=CAT+NUM+['time_slot','demand_lag']

trl=train_raw.copy(); tel=test_raw.copy()
for c in ['RoadType','Weather']:
    mv=trl[c].mode()[0]; trl[c]=trl[c].fillna(mv); tel[c]=tel[c].fillna(mv)
tm=trl['Temperature'].mean(); trl['Temperature']=trl['Temperature'].fillna(tm); tel['Temperature']=tel['Temperature'].fillna(tm)
trl['d48_profile_smooth']=trl['d48_profile_smooth'].fillna(_glm); tel['d48_profile_smooth']=tel['d48_profile_smooth'].fillna(_glm)

enc_map={}; cb=pd.concat([trl[CAT],tel[CAT]],ignore_index=True)
for c in CAT:
    le=LabelEncoder(); le.fit(cb[c].dropna().unique()); enc_map[c]=le
    trl[c]=le.transform(trl[c].astype(str)); tel[c]=le.transform(tel[c].astype(str))

trc=train_raw.copy(); tec=test_raw.copy()
for c in CAT: trc[c]=trc[c].fillna('Missing').astype(str); tec[c]=tec[c].fillna('Missing').astype(str)

def ofe(tr,te,gc,y,kf,sm=10.):
    gm=y.mean(); te_=np.full(len(tr),gm); tmp=tr[gc].copy(); tmp['__y']=y
    for ti,vi in kf.split(tr):
        st=(tmp.iloc[ti].groupby(gc)['__y'].agg(['mean','count']).reset_index().rename(columns={'mean':'__m','count':'__n'}))
        st['__b']=(st['__m']*st['__n']+gm*sm)/(st['__n']+sm)
        mg=tmp.iloc[vi][gc].merge(st[gc+['__b']],on=gc,how='left'); te_[vi]=mg['__b'].fillna(gm).values
    fl=(tmp.groupby(gc)['__y'].agg(['mean','count']).reset_index().rename(columns={'mean':'__m','count':'__n'}))
    fl['__b']=(fl['__m']*fl['__n']+gm*sm)/(fl['__n']+sm)
    return te_, te[gc].merge(fl[gc+['__b']],on=gc,how='left')['__b'].fillna(gm).values

trl['time_slot']=train_raw['time_slot'].values; tel['time_slot']=test_raw['time_slot'].values
kf=KFold(n_splits=N_FOLDS,shuffle=True,random_state=SEED)
print('OOF encodings...')
tr1,te1=ofe(trl,tel,['geohash','hour'],y_tr,kf)
tr2,te2=ofe(trl,tel,['gh5','hour'],y_tr,kf)
tr3,te3=ofe(trl,tel,['geohash'],y_tr,kf)
tr4,te4=ofe(trl,tel,['RoadType','hour'],y_tr,kf)
tr5,te5=ofe(trl,tel,['geohash','time_slot'],y_tr,kf,sm=5.)
tr6,te6=ofe(trl,tel,['gh5','time_slot'],y_tr,kf,sm=5.)

def bX(df,e1,e2,e3,e4,e5,e6):
    o=df[CAT+NUM].copy().values.astype(float); lg=df['demand_lag'].fillna(df['demand_lag'].median()).values
    return np.column_stack([o,lg,e1,e2,e3,e4,e5,e6])

Xtr=bX(trl,tr1,tr2,tr3,tr4,tr5,tr6); Xte=bX(tel,te1,te2,te3,te4,te5,te6)
for df in [trc,tec]:
    df['time_slot']=train_raw['time_slot'].values if df is trc else test_raw['time_slot'].values
    df['demand_lag']=train_raw['demand_lag'].values if df is trc else test_raw['demand_lag'].values
cfi=list(range(len(CAT)))

# Load pseudo-labels
ps=np.load('preds_v3c_multiseed.npy')
print(f'Pseudo-labels: mean={ps.mean():.4f}  std={ps.std():.4f}')

# Extended dataset
pst=test_raw.copy(); pstl=tel.copy(); pstc=tec.copy()
pst['demand']=ps; pstl['demand']=ps; pstc['demand']=ps
ext_l=pd.concat([trl,pstl],ignore_index=True)
ext_c=pd.concat([trc,pstc],ignore_index=True)
ext_r=pd.concat([train_raw,pst],ignore_index=True)
y_ext=ext_r['demand'].values; n_ext=len(ext_l)
print(f'Extended: {n_ext:,} rows')

def full_enc(tr,te,gc,y,sm=10.):
    gm=y.mean(); tmp=tr[gc].copy(); tmp['__y']=y
    fl=(tmp.groupby(gc)['__y'].agg(['mean','count']).reset_index().rename(columns={'mean':'__m','count':'__n'}))
    fl['__b']=(fl['__m']*fl['__n']+gm*sm)/(fl['__n']+sm)
    return te[gc].merge(fl[gc+['__b']],on=gc,how='left')['__b'].fillna(gm).values

te1p=full_enc(ext_l,tel,['geohash','hour'],y_ext)
te2p=full_enc(ext_l,tel,['gh5','hour'],y_ext)
te3p=full_enc(ext_l,tel,['geohash'],y_ext)
te4p=full_enc(ext_l,tel,['RoadType','hour'],y_ext)
te5p=full_enc(ext_l,tel,['geohash','time_slot'],y_ext,sm=5.)
te6p=full_enc(ext_l,tel,['gh5','time_slot'],y_ext,sm=5.)
Xtep=bX(tel,te1p,te2p,te3p,te4p,te5p,te6p)

kfe=KFold(n_splits=N_FOLDS,shuffle=True,random_state=SEED)
ec=ext_l[CAT+NUM].copy().values.astype(float); el=ext_l['demand_lag'].fillna(ext_l['demand_lag'].median()).values

_,tr1e=ofe(ext_l,tel,['geohash','hour'],y_ext,kfe)
_,tr2e=ofe(ext_l,tel,['gh5','hour'],y_ext,kfe)
_,tr3e=ofe(ext_l,tel,['geohash'],y_ext,kfe)
_,tr4e=ofe(ext_l,tel,['RoadType','hour'],y_ext,kfe)
_,tr5e=ofe(ext_l,tel,['geohash','time_slot'],y_ext,kfe,sm=5.)
_,tr6e=ofe(ext_l,tel,['gh5','time_slot'],y_ext,kfe,sm=5.)
tr1e2,_=ofe(ext_l,tel,['geohash','hour'],y_ext,kfe)
tr2e2,_=ofe(ext_l,tel,['gh5','hour'],y_ext,kfe)
tr3e2,_=ofe(ext_l,tel,['geohash'],y_ext,kfe)
tr4e2,_=ofe(ext_l,tel,['RoadType','hour'],y_ext,kfe)
tr5e2,_=ofe(ext_l,tel,['geohash','time_slot'],y_ext,kfe,sm=5.)
tr6e2,_=ofe(ext_l,tel,['gh5','time_slot'],y_ext,kfe,sm=5.)
Xtre=np.column_stack([ec,el,tr1e2,tr2e2,tr3e2,tr4e2,tr5e2,tr6e2])

# Pseudo-labeled LGB
print('\nPseudo LGB...'); ps_lgb=np.zeros(len(test_raw))
for fold,(ti,vi) in enumerate(kfe.split(Xtre)):
    m=lgb.LGBMRegressor(objective='regression',metric='rmse',n_estimators=2000,learning_rate=0.02,
                         num_leaves=127,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,
                         reg_alpha=0.1,reg_lambda=0.1,n_jobs=2,verbose=-1,random_state=SEED)
    m.fit(Xtre[ti],y_ext[ti],eval_set=[(Xtre[vi],y_ext[vi])],
          callbacks=[lgb.early_stopping(100,verbose=False),lgb.log_evaluation(9999)])
    ps_lgb+=np.clip(m.predict(Xtep),0,1)/N_FOLDS; print(f'  F{fold+1} iter={m.best_iteration_}')

# Pseudo-labeled CatBoost
print('Pseudo CatBoost...'); ps_cat=np.zeros(len(test_raw))
for fold,(ti,vi) in enumerate(kfe.split(ext_c)):
    cb2=CatBoostRegressor(iterations=6000,learning_rate=0.03,depth=8,
                          loss_function='RMSE',random_seed=SEED,verbose=0,early_stopping_rounds=150)
    cb2.fit(ext_c.iloc[ti][CFE],y_ext[ti],cat_features=cfi,
            eval_set=(ext_c.iloc[vi][CFE],y_ext[vi]),use_best_model=True)
    ps_cat+=np.clip(cb2.predict(tec[CFE]),0,1)/N_FOLDS; print(f'  F{fold+1} iter={cb2.best_iteration_}')

# Pseudo-labeled XGB
print('Pseudo XGB...'); ps_xgb=np.zeros(len(test_raw))
for fold,(ti,vi) in enumerate(kfe.split(Xtre)):
    xgb=XGBRegressor(objective='reg:squarederror',n_estimators=2000,learning_rate=0.02,
                     max_depth=6,min_child_weight=20,subsample=0.8,colsample_bytree=0.8,
                     reg_alpha=0.1,reg_lambda=0.1,early_stopping_rounds=100,
                     n_jobs=2,random_state=SEED,verbosity=0)
    xgb.fit(Xtre[ti],y_ext[ti],eval_set=[(Xtre[vi],y_ext[vi])],verbose=False)
    ps_xgb+=np.clip(xgb.predict(Xtep),0,1)/N_FOLDS; print(f'  F{fold+1} iter={xgb.best_iteration}')

final=np.clip(0.20*ps_lgb+0.15*ps_xgb+0.65*ps_cat,0,1)
best=pd.read_csv('submissions/best/submission_best_91798.csv')['demand'].values
print(f'\nvs 91.798: mean={final.mean()-best.mean():+.5f}  corr={np.corrcoef(final,best)[0,1]:.4f}')
np.save('preds_v3c_pseudo.npy',final)
pd.DataFrame({'Index':test_raw['Index'],'demand':final}).to_csv('submission_v3c_pseudo.csv',index=False)
print(f'Saved submission_v3c_pseudo.csv  [{(time.time()-t0)/60:.1f} min]')
