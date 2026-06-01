"""
PyTorch MLP with categorical embeddings — uses demand_lag + d48_profile_smooth.
Provides architectural diversity vs tree-based ensemble models.

Usage:
    python3 nn_model.py
Saves: submission_nn.csv, preds_nn.npy
"""
import warnings, time
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
N_FOLDS = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}  SEED={SEED}')
torch.manual_seed(SEED)
np.random.seed(SEED)
t0 = time.time()

# ── Data ──────────────────────────────────────────────────────────────────────
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

# ── Geohash decode ────────────────────────────────────────────────────────────
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

# ── Encode categoricals ───────────────────────────────────────────────────────
CAT_COLS = ['geohash', 'gh4', 'gh5', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
all_data = pd.concat([train_raw, test_raw], ignore_index=True)
encoders, vocab_sizes = {}, {}
for col in CAT_COLS:
    le = LabelEncoder()
    vals = all_data[col].fillna('Missing').astype(str)
    le.fit(vals)
    encoders[col] = le
    vocab_sizes[col] = len(le.classes_)
    train_raw[col+'_enc'] = le.transform(train_raw[col].fillna('Missing').astype(str))
    test_raw[col+'_enc']  = le.transform(test_raw[col].fillna('Missing').astype(str))

# demand_lag feature with fallback hierarchy + smoothed day48 profile
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
lag_median = train_raw['demand_lag'].median()
train_raw['demand_lag_filled'] = train_raw['demand_lag'].fillna(lag_median)
test_raw['demand_lag_filled']  = test_raw['demand_lag'].fillna(lag_median)

# Numeric features — WITH demand_lag
NUM_COLS = ['hour', 'minute', 'hour_sin', 'hour_cos', 'ts_sin', 'ts_cos',
            'day_sin', 'day_cos', 'time_slot',
            'NumberofLanes', 'Temperature', 'lat', 'lon',
            'geo_day_trend', 'neighbor_demand_mean', 'demand_lag_filled',
            'miss_RoadType', 'miss_Temperature', 'miss_Weather', 'd48_profile_smooth']

for df in [train_raw, test_raw]:
    df['Temperature'] = df['Temperature'].fillna(df['Temperature'].mean())
    df['d48_profile_smooth'] = df['d48_profile_smooth'].fillna(_global_mean_d48)

scaler = StandardScaler()
X_num_tr = scaler.fit_transform(train_raw[NUM_COLS].values)
X_num_te = scaler.transform(test_raw[NUM_COLS].values)
X_cat_tr = train_raw[[c+'_enc' for c in CAT_COLS]].values
X_cat_te = test_raw[[c+'_enc' for c in CAT_COLS]].values
y_train  = train_raw['demand'].values

# ── Model definition ──────────────────────────────────────────────────────────
EMB_DIM = {col: min(50, (vocab_sizes[col]+1)//2) for col in CAT_COLS}

class TrafficMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_sizes[col]+1, EMB_DIM[col]) for col in CAT_COLS
        ])
        emb_total = sum(EMB_DIM.values())
        inp_dim = emb_total + len(NUM_COLS)
        self.net = nn.Sequential(
            nn.Linear(inp_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256),    nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),    nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64),     nn.ReLU(),
            nn.Linear(64, 1),       nn.Sigmoid()
        )
    def forward(self, x_cat, x_num):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embs + [x_num], dim=1)
        return self.net(x).squeeze(1)

# ── Training function ─────────────────────────────────────────────────────────
def train_fold(X_cat_tr, X_num_tr, y_tr, X_cat_val, X_num_val, y_val, seed=42):
    torch.manual_seed(seed)
    model = TrafficMLP().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
    loss_fn = nn.MSELoss()

    ds_tr  = TensorDataset(torch.LongTensor(X_cat_tr), torch.FloatTensor(X_num_tr), torch.FloatTensor(y_tr))
    ds_val = TensorDataset(torch.LongTensor(X_cat_val), torch.FloatTensor(X_num_val), torch.FloatTensor(y_val))
    dl_tr  = DataLoader(ds_tr, batch_size=2048, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=4096, shuffle=False)

    best_val, best_state, patience, no_imp = 1e9, None, 15, 0
    for epoch in range(150):
        model.train()
        for xc, xn, yb in dl_tr:
            xc,xn,yb = xc.to(DEVICE), xn.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss_fn(model(xc, xn), yb).backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            val_preds = torch.cat([model(xc.to(DEVICE), xn.to(DEVICE)) for xc,xn,_ in dl_val]).cpu().numpy()
        val_loss = ((val_preds - y_val)**2).mean()
        if val_loss < best_val:
            best_val = val_loss; best_state = {k: v.clone() for k,v in model.state_dict().items()}; no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience: break

    model.load_state_dict(best_state)
    return model

def predict(model, X_cat, X_num):
    model.eval()
    ds = TensorDataset(torch.LongTensor(X_cat), torch.FloatTensor(X_num))
    dl = DataLoader(ds, batch_size=4096, shuffle=False)
    with torch.no_grad():
        return np.clip(torch.cat([model(xc.to(DEVICE), xn.to(DEVICE)) for xc,xn in dl]).cpu().numpy(), 0, 1)

# ── 5-fold OOF ────────────────────────────────────────────────────────────────
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_nn = np.zeros(len(y_train))
test_nn = np.zeros(len(test_raw))

print('Training PyTorch MLP (5-fold, no demand_lag)...')
for fold, (ti, vi) in enumerate(kf.split(X_num_tr)):
    model = train_fold(X_cat_tr[ti], X_num_tr[ti], y_train[ti],
                       X_cat_tr[vi], X_num_tr[vi], y_train[vi], seed=SEED+fold)
    oof_nn[vi] = predict(model, X_cat_tr[vi], X_num_tr[vi])
    test_nn   += predict(model, X_cat_te, X_num_te) / N_FOLDS
    print(f'  Fold {fold+1}  R²={r2_score(y_train[vi], oof_nn[vi]):.4f}')

print(f'\nNN OOF R²={r2_score(y_train, oof_nn):.4f}')
print(f'NN preds: mean={test_nn.mean():.4f}  std={test_nn.std():.4f}')

np.save('preds_nn.npy', test_nn)
pd.DataFrame({'Index': test_raw['Index'].values, 'demand': test_nn}).to_csv('submission_nn.csv', index=False)

best98 = pd.read_csv('submissions/best/submission_best_91798.csv')['demand'].values
for alpha in [0.05, 0.08, 0.10, 0.15, 0.20]:
    blend = np.clip((1-alpha)*best98 + alpha*test_nn, 0, 1)
    fname = f'submission_blend_nn{int(alpha*100):02d}.csv'
    pd.DataFrame({'Index': test_raw['Index'].values, 'demand': blend}).to_csv(fname, index=False)
    delta = blend - best98
    print(f'  alpha={alpha:.2f}: shift mean={delta.mean():+.5f}  std={delta.std():.5f} → {fname}')

print(f'\nTotal time: {(time.time()-t0)/60:.1f} min')
