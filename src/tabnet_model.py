"""
TabNet — attention-based tabular neural network.
Trained on full 77k rows WITH demand_lag. 5-fold OOF.
TabNet learns feature selection per sample via sparse attention.

Usage:
    /usr/bin/python3 tabnet_model.py
Saves: preds_tabnet.npy, submission_tabnet.csv, submission_blend_tabnet*.csv
"""
import warnings, time
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from pytorch_tabnet.tab_model import TabNetRegressor

SEED = 42
N_FOLDS = 5
t0 = time.time()
print(f'TabNet  SEED={SEED}')

# ── Data & features (same pipeline as multiseed.py) ───────────────────────────
DATA_DIR = next((p for p in [Path('data'), Path('dataset')] if p.exists()), Path('data'))
train_raw = pd.read_csv(DATA_DIR / 'train.csv')
test_raw  = pd.read_csv(DATA_DIR / 'test.csv')

def add_time_features(df):
    parts = df['timestamp'].str.split(':', expand=True).astype(float)
    df = df.copy()
    df['hour']      = parts[0].values
    df['minute']    = parts[1].values
    df['ts_minutes']= df['hour'] * 60 + df['minute']
    df['time_slot'] = (df['ts_minutes'] // 15).astype(int)
    df['hour_sin']  = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']  = np.cos(2 * np.pi * df['hour'] / 24)
    df['ts_sin']    = np.sin(2 * np.pi * df['time_slot'] / 96)
    df['ts_cos']    = np.cos(2 * np.pi * df['time_slot'] / 96)
    df['day_sin']   = np.sin(2 * np.pi * df['day'] / 7)
    df['day_cos']   = np.cos(2 * np.pi * df['day'] / 7)
    return df

train_raw = add_time_features(train_raw)
test_raw  = add_time_features(test_raw)

# ── Missingness flags ──────────────────────────────────────────────────────────
for df in [train_raw, test_raw]:
    df['miss_RoadType'] = df['RoadType'].isna().astype(int)
    df['miss_Temperature'] = df['Temperature'].isna().astype(int)
    df['miss_Weather'] = df['Weather'].isna().astype(int)

_BASE32 = '0123456789bcdefghjkmnpqrstuvwxyz'
def decode_geohash(gh):
    lat_r=[-90.,90.]; lon_r=[-180.,180.]; is_lon=True
    for c in gh:
        cd=_BASE32.index(c)
        for bits in [16,8,4,2,1]:
            if is_lon:
                mid=(lon_r[0]+lon_r[1])/2
                if cd&bits: lon_r[0]=mid
                else: lon_r[1]=mid
            else:
                mid=(lat_r[0]+lat_r[1])/2
                if cd&bits: lat_r[0]=mid
                else: lat_r[1]=mid
            is_lon=not is_lon
    return (lat_r[0]+lat_r[1])/2,(lon_r[0]+lon_r[1])/2

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
        if bits==5: result+=_BASE32[bit]; bits=0; bit=0
    return result

_gh_coords = {gh: decode_geohash(gh)
              for gh in pd.concat([train_raw['geohash'], test_raw['geohash']]).unique()}
for df in [train_raw, test_raw]:
    df['lat'] = df['geohash'].map(lambda g: _gh_coords[g][0])
    df['lon'] = df['geohash'].map(lambda g: _gh_coords[g][1])
    df['gh4'] = df['geohash'].str[:4]
    df['gh5'] = df['geohash'].str[:5]

# geo_day_trend
_d48g = train_raw[train_raw['day']==48].groupby('geohash')['demand'].mean()
_d49g = train_raw[train_raw['day']==49].groupby('geohash')['demand'].mean()
_trend = (_d49g / (_d48g + 1e-6)); _gtrend = float(_trend.mean())
for df in [train_raw, test_raw]:
    df['geo_day_trend'] = df['geohash'].map(_trend).fillna(_gtrend)

# neighbor_demand_mean
_geo_mean = train_raw.groupby('geohash')['demand'].mean()
_global_mean = float(_geo_mean.mean())

def _nbr(gh):
    lat,lon=_gh_coords[gh]
    lat_r=[-90.,90.]; lon_r=[-180.,180.]; is_lon=True
    for ch in gh:
        cd=_BASE32.index(ch)
        for b in [16,8,4,2,1]:
            if is_lon:
                mid=(lon_r[0]+lon_r[1])/2
                if cd&b: lon_r[0]=mid
                else: lon_r[1]=mid
            else:
                mid=(lat_r[0]+lat_r[1])/2
                if cd&b: lat_r[0]=mid
                else: lat_r[1]=mid
            is_lon=not is_lon
    ls=lat_r[1]-lat_r[0]; ms=lon_r[1]-lon_r[0]
    ds=[_geo_mean[encode_geohash(lat+dl*ls,lon+dm*ms,len(gh))]
        for dl in [-1,0,1] for dm in [-1,0,1]
        if not(dl==0 and dm==0)
        and encode_geohash(lat+dl*ls,lon+dm*ms,len(gh)) in _geo_mean]
    return float(np.mean(ds)) if ds else _global_mean

_nbr_map = {gh: _nbr(gh) for gh in _gh_coords}
for df in [train_raw, test_raw]:
    df['neighbor_demand_mean'] = df['geohash'].map(_nbr_map).fillna(_global_mean)

# demand_lag fallback hierarchy + smoothed day48 profile
lag_d48 = (train_raw[train_raw['day']==48][['geohash','timestamp','demand']]
           .rename(columns={'demand':'demand_lag'}))
train_raw = train_raw.merge(lag_d48, on=['geohash','timestamp'], how='left')
train_raw['demand_lag'] = np.where(train_raw['day']==48, np.nan, train_raw['demand_lag'])
test_raw  = test_raw.merge(lag_d48, on=['geohash','timestamp'], how='left')

_d48 = train_raw[train_raw['day']==48][['geohash','gh5','time_slot','demand']]
_geo_ts = _d48.groupby(['geohash','time_slot'])['demand'].mean()
_gh5_ts = _d48.groupby(['gh5','time_slot'])['demand'].mean()
_geo_mean_d48 = _d48.groupby('geohash')['demand'].mean()
_global_mean_d48 = float(_d48['demand'].mean())

def _build_smoothed_profile(geo_ts, geo_mean, global_mean, n_slots=96):
    prof = {}
    for gh, grp in geo_ts.reset_index().groupby('geohash'):
        vals = np.full(n_slots, np.nan)
        slots = grp['time_slot'].values
        vals[slots] = grp['demand'].values
        fill = float(geo_mean.get(gh, global_mean))
        vals = np.where(np.isnan(vals), fill, vals)
        smooth = (np.roll(vals, 1) + 2 * vals + np.roll(vals, -1)) / 4.0
        for s in range(n_slots):
            prof[(gh, s)] = smooth[s]
    return prof

_geo_ts_smooth = _build_smoothed_profile(_geo_ts, _geo_mean_d48, _global_mean_d48)

def _apply_lag_fallback(df, fill_mask):
    idx_geo_ts = df.set_index(['geohash','time_slot']).index
    idx_gh5_ts = df.set_index(['gh5','time_slot']).index
    geo_ts_vals = pd.Series(idx_geo_ts.map(_geo_ts_smooth), index=df.index)
    gh5_ts_vals = pd.Series(idx_gh5_ts.map(_gh5_ts), index=df.index)
    geo_mean_vals = df['geohash'].map(_geo_mean_d48)
    fallback = geo_ts_vals.fillna(gh5_ts_vals).fillna(geo_mean_vals).fillna(_global_mean_d48)
    lag = df['demand_lag'].copy()
    to_fill = fill_mask & lag.isna()
    lag[to_fill] = fallback[to_fill]
    df['demand_lag'] = lag
    df['d48_profile_smooth'] = geo_ts_vals.fillna(_global_mean_d48)
    return df

train_raw = _apply_lag_fallback(train_raw, train_raw['day'] != 48)
test_raw  = _apply_lag_fallback(test_raw,  test_raw['day'] == 49)
train_raw.loc[train_raw['day']==48, 'd48_profile_smooth'] = np.nan
lag_med = float(train_raw['demand_lag'].median())
train_raw['demand_lag_filled'] = train_raw['demand_lag'].fillna(lag_med)
test_raw['demand_lag_filled']  = test_raw['demand_lag'].fillna(lag_med)

# ── Encode features ───────────────────────────────────────────────────────────
CAT_COLS = ['geohash', 'gh4', 'gh5', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
all_data = pd.concat([train_raw, test_raw], ignore_index=True)
cat_idxs, cat_dims = [], []
for i, col in enumerate(CAT_COLS):
    le = LabelEncoder()
    vals = all_data[col].fillna('Missing').astype(str)
    le.fit(vals)
    train_raw[col+'_enc'] = le.transform(train_raw[col].fillna('Missing').astype(str))
    test_raw[col+'_enc']  = le.transform(test_raw[col].fillna('Missing').astype(str))
    cat_idxs.append(i)
    cat_dims.append(len(le.classes_) + 1)

for df in [train_raw, test_raw]:
    df['Temperature'] = df['Temperature'].fillna(df['Temperature'].mean())
    df['d48_profile_smooth'] = df['d48_profile_smooth'].fillna(_global_mean_d48)
for col in ['RoadType', 'Weather']:
    mode = train_raw[col+'_enc'].mode()[0]
    train_raw[col+'_enc'] = train_raw[col+'_enc'].fillna(mode)
    test_raw[col+'_enc']  = test_raw[col+'_enc'].fillna(mode)

NUM_COLS = ['hour', 'minute', 'hour_sin', 'hour_cos', 'ts_sin', 'ts_cos',
            'day_sin', 'day_cos', 'time_slot', 'day',
            'NumberofLanes', 'Temperature', 'lat', 'lon',
            'geo_day_trend', 'neighbor_demand_mean', 'demand_lag_filled',
            'miss_RoadType', 'miss_Temperature', 'miss_Weather', 'd48_profile_smooth']

# Build feature matrix: categoricals first, then numerics
X_cat_tr = train_raw[[c+'_enc' for c in CAT_COLS]].values.astype(int)
X_num_tr = train_raw[NUM_COLS].values.astype(float)
X_cat_te = test_raw[[c+'_enc' for c in CAT_COLS]].values.astype(int)
X_num_te = test_raw[NUM_COLS].values.astype(float)

scaler = StandardScaler()
X_num_tr = scaler.fit_transform(X_num_tr)
X_num_te = scaler.transform(X_num_te)

X_tr = np.hstack([X_cat_tr, X_num_tr]).astype(np.float32)
X_te = np.hstack([X_cat_te, X_num_te]).astype(np.float32)
y_train = train_raw['demand'].values.astype(np.float32).reshape(-1, 1)

print(f'Features: {X_tr.shape[1]} ({len(CAT_COLS)} cat + {len(NUM_COLS)} num)')
print(f'cat_idxs: {cat_idxs}  cat_dims: {cat_dims}')

# ── 5-fold TabNet ─────────────────────────────────────────────────────────────
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_tabnet  = np.zeros(len(train_raw))
test_tabnet = np.zeros(len(test_raw))

print('\nTraining TabNet (5-fold)...')
for fold, (ti, vi) in enumerate(kf.split(X_tr)):
    model = TabNetRegressor(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=4,
        n_d=32, n_a=32,
        n_steps=5,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        lambda_sparse=1e-4,
        optimizer_fn=__import__('torch').optim.Adam,
        optimizer_params=dict(lr=2e-3, weight_decay=1e-5),
        scheduler_fn=__import__('torch').optim.lr_scheduler.StepLR,
        scheduler_params=dict(step_size=10, gamma=0.9),
        mask_type='entmax',
        seed=SEED + fold,
        verbose=0,
        device_name='cuda',
    )
    model.fit(
        X_tr[ti], y_train[ti],
        eval_set=[(X_tr[vi], y_train[vi])],
        eval_name=['val'],
        eval_metric=['rmse'],
        max_epochs=200,
        patience=20,
        batch_size=4096,
        virtual_batch_size=512,
    )
    oof_tabnet[vi] = np.clip(model.predict(X_tr[vi]).flatten(), 0, 1)
    test_tabnet   += np.clip(model.predict(X_te).flatten(), 0, 1) / N_FOLDS
    print(f'  Fold {fold+1}  R²={r2_score(y_train[vi].flatten(), oof_tabnet[vi]):.4f}  '
          f'epochs={model.best_epoch}')

print(f'\nTabNet OOF R²={r2_score(y_train.flatten(), oof_tabnet):.4f}')
print(f'TabNet preds: mean={test_tabnet.mean():.4f}  std={test_tabnet.std():.4f}')

np.save('preds_tabnet.npy', test_tabnet)
pd.DataFrame({'Index': test_raw['Index'].values, 'demand': test_tabnet}).to_csv('submission_tabnet.csv', index=False)

best98 = pd.read_csv('submissions/best/submission_best_91798.csv')['demand'].values
print(f'Corr with best: {np.corrcoef(test_tabnet, best98)[0,1]:.4f}')
for alpha in [0.05, 0.08, 0.10, 0.15, 0.20]:
    blend = np.clip((1-alpha)*best98 + alpha*test_tabnet, 0, 1)
    fname = f'submission_blend_tabnet{int(alpha*100):02d}.csv'
    pd.DataFrame({'Index': test_raw['Index'].values, 'demand': blend}).to_csv(fname, index=False)
    delta = blend - best98
    print(f'  alpha={alpha:.2f}: shift mean={delta.mean():+.5f}  std={delta.std():.5f} → {fname}')

print(f'\nTotal time: {(time.time()-t0)/60:.1f} min')
