"""
Proxy-split Optuna hyperparameter search.

Proxy train:  d49 rows (7,872) + d48 early rows (7,115) = 14,987
Proxy val:    d48 daytime rows (41,851) — exact distribution match to test

Trains FINAL models on full 77k rows with best params, saves preds.

Usage:
    python3 optuna_proxy.py lgb        # tune LGB only
    python3 optuna_proxy.py xgb        # tune XGB only
    python3 optuna_proxy.py cat        # tune CatBoost only
    python3 optuna_proxy.py all        # tune all three
"""
import sys, warnings, time, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import optuna
import lightgbm as lgb
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODE  = sys.argv[1] if len(sys.argv) > 1 else 'all'
SEED  = 42
N_FOLDS = 5
N_TRIALS_LGB = 50
N_TRIALS_XGB = 30
N_TRIALS_CAT = 30
t0 = time.time()

print(f'MODE={MODE}  SEED={SEED}')

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_DIR = next((p for p in [Path('data'), Path('dataset')] if p.exists()), Path('data'))
train_raw = pd.read_csv(DATA_DIR / 'train.csv')
test_raw  = pd.read_csv(DATA_DIR / 'test.csv')
print(f'train: {train_raw.shape}   test: {test_raw.shape}')

def add_time_features(df):
    parts = df['timestamp'].str.split(':', expand=True).astype(float)
    df = df.copy()
    df['hour']      = parts[0].values
    df['minute']    = parts[1].values
    df['ts_minutes']= df['hour'] * 60 + df['minute']
    df['time_slot'] = (df['ts_minutes'] // 15).astype(int)
    df['hour_sin']  = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']  = np.cos(2 * np.pi * df['hour'] / 24)
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

# ── Geohash decode + encode ────────────────────────────────────────────────────
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

# ── demand_lag ────────────────────────────────────────────────────────────────
lag_d48 = (train_raw[train_raw['day']==48][['geohash','timestamp','demand']]
           .rename(columns={'demand':'demand_lag'}))
train_raw = train_raw.merge(lag_d48, on=['geohash','timestamp'], how='left')
train_raw['demand_lag'] = np.where(train_raw['day']==48, np.nan, train_raw['demand_lag'])
test_raw  = test_raw.merge(lag_d48, on=['geohash','timestamp'], how='left')

# demand_lag fallback hierarchy + smoothed day48 profile
_d48 = train_raw[train_raw['day']==48][['geohash','gh5','time_slot','demand']]
_geo_ts = _d48.groupby(['geohash','time_slot'])['demand'].mean()
_gh5_ts = _d48.groupby(['gh5','time_slot'])['demand'].mean()
_geo_mean = _d48.groupby('geohash')['demand'].mean()
_global_mean = float(_d48['demand'].mean())

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

_geo_ts_smooth = _build_smoothed_profile(_geo_ts, _geo_mean, _global_mean)

def _apply_lag_fallback(df, fill_mask):
    idx_geo_ts = df.set_index(['geohash','time_slot']).index
    idx_gh5_ts = df.set_index(['gh5','time_slot']).index
    geo_ts_vals = pd.Series(idx_geo_ts.map(_geo_ts_smooth), index=df.index)
    gh5_ts_vals = pd.Series(idx_gh5_ts.map(_gh5_ts), index=df.index)
    geo_mean_vals = df['geohash'].map(_geo_mean)
    fallback = geo_ts_vals.fillna(gh5_ts_vals).fillna(geo_mean_vals).fillna(_global_mean)
    lag = df['demand_lag'].copy()
    to_fill = fill_mask & lag.isna()
    lag[to_fill] = fallback[to_fill]
    df['demand_lag'] = lag
    df['d48_profile_smooth'] = geo_ts_vals.fillna(_global_mean)
    return df

train_raw = _apply_lag_fallback(train_raw, train_raw['day'] != 48)
test_raw  = _apply_lag_fallback(test_raw,  test_raw['day'] == 49)
train_raw.loc[train_raw['day']==48, 'd48_profile_smooth'] = np.nan

# ── geo_day_trend ─────────────────────────────────────────────────────────────
_d48g = train_raw[train_raw['day']==48].groupby('geohash')['demand'].mean()
_d49g = train_raw[train_raw['day']==49].groupby('geohash')['demand'].mean()
_trend = (_d49g / (_d48g + 1e-6)); _gtrend = float(_trend.mean())
for df in [train_raw, test_raw]:
    df['geo_day_trend'] = df['geohash'].map(_trend).fillna(_gtrend)

# ── neighbor_demand_mean ──────────────────────────────────────────────────────
_geo_demand_mean   = train_raw.groupby('geohash')['demand'].mean()
_global_demand_mean = float(_geo_demand_mean.mean())

def _get_neighbor_mean(gh):
    lat, lon = _gh_coords[gh]
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
    lat_step=lat_r[1]-lat_r[0]; lon_step=lon_r[1]-lon_r[0]
    demands=[_geo_demand_mean[encode_geohash(lat+dlat*lat_step, lon+dlon*lon_step, len(gh))]
             for dlat in [-1,0,1] for dlon in [-1,0,1]
             if not(dlat==0 and dlon==0)
             and encode_geohash(lat+dlat*lat_step, lon+dlon*lon_step, len(gh)) in _geo_demand_mean]
    return float(np.mean(demands)) if demands else _global_demand_mean

_gh_neighbor_mean = {gh: _get_neighbor_mean(gh) for gh in _gh_coords}
for df in [train_raw, test_raw]:
    df['neighbor_demand_mean'] = df['geohash'].map(_gh_neighbor_mean).fillna(_global_demand_mean)

y_train = train_raw['demand'].values
n_train = len(train_raw)

# ── Proxy split masks ─────────────────────────────────────────────────────────
_d49_ts  = set(train_raw[train_raw['day']==49]['timestamp'].unique())
_test_ts = set(test_raw['timestamp'].unique())
proxy_train_mask = ((train_raw['day']==49) |
                    ((train_raw['day']==48) & train_raw['timestamp'].isin(_d49_ts))).values
proxy_val_mask   = ((train_raw['day']==48) & train_raw['timestamp'].isin(_test_ts)).values
print(f'Proxy train: {proxy_train_mask.sum():,}  Proxy val: {proxy_val_mask.sum():,}')

# ── Columns ───────────────────────────────────────────────────────────────────
CAT_COLS = ['geohash', 'gh4', 'gh5', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
NUM_COLS = ['day', 'hour', 'minute', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'NumberofLanes', 'Temperature', 'lat', 'lon', 'geo_day_trend', 'neighbor_demand_mean',
            'miss_RoadType', 'miss_Temperature', 'miss_Weather', 'd48_profile_smooth']
CAT_FEAT_COLS = CAT_COLS + NUM_COLS + ['time_slot', 'demand_lag']

# ── LGB imputation ────────────────────────────────────────────────────────────
train_lgb = train_raw.copy(); test_lgb = test_raw.copy()
for col in ['RoadType', 'Weather']:
    mode_val = train_lgb[col].mode()[0]
    train_lgb[col] = train_lgb[col].fillna(mode_val)
    test_lgb[col]  = test_lgb[col].fillna(mode_val)
temp_mean = train_lgb['Temperature'].mean()
train_lgb['Temperature'] = train_lgb['Temperature'].fillna(temp_mean)
test_lgb['Temperature']  = test_lgb['Temperature'].fillna(temp_mean)
profile_fill = _global_mean
train_lgb['d48_profile_smooth'] = train_lgb['d48_profile_smooth'].fillna(profile_fill)
test_lgb['d48_profile_smooth']  = test_lgb['d48_profile_smooth'].fillna(profile_fill)

encoders = {}
combined = pd.concat([train_lgb[CAT_COLS], test_lgb[CAT_COLS]], ignore_index=True)
for col in CAT_COLS:
    le = LabelEncoder()
    le.fit(combined[col].dropna().unique())
    encoders[col] = le
    train_lgb[col] = le.transform(train_lgb[col].astype(str))
    test_lgb[col]  = le.transform(test_lgb[col].astype(str))

# ── CatBoost imputation ───────────────────────────────────────────────────────
train_cat = train_raw.copy(); test_cat = test_raw.copy()
for col in CAT_COLS:
    train_cat[col] = train_cat[col].fillna('Missing').astype(str)
    test_cat[col]  = test_cat[col].fillna('Missing').astype(str)

# ── OOF target encodings (full KFold) ─────────────────────────────────────────
def oof_encode(tr_df, te_df, group_cols, y, kf, smoothing=10.0):
    gm = y.mean()
    tr_enc = np.full(len(tr_df), gm)
    tmp = tr_df[group_cols].copy(); tmp['__y'] = y
    for ti, vi in kf.split(tr_df):
        stats = (tmp.iloc[ti].groupby(group_cols)['__y']
                 .agg(['mean','count']).reset_index()
                 .rename(columns={'mean':'__m','count':'__n'}))
        stats['__b'] = (stats['__m']*stats['__n'] + gm*smoothing) / (stats['__n']+smoothing)
        merged = tmp.iloc[vi][group_cols].merge(stats[group_cols+['__b']], on=group_cols, how='left')
        tr_enc[vi] = merged['__b'].fillna(gm).values
    full = (tmp.groupby(group_cols)['__y']
            .agg(['mean','count']).reset_index()
            .rename(columns={'mean':'__m','count':'__n'}))
    full['__b'] = (full['__m']*full['__n'] + gm*smoothing) / (full['__n']+smoothing)
    te_enc = te_df[group_cols].merge(full[group_cols+['__b']], on=group_cols, how='left')['__b'].fillna(gm).values
    return tr_enc, te_enc

train_lgb['time_slot'] = train_raw['time_slot'].values
test_lgb['time_slot']  = test_raw['time_slot'].values
kf_full = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

print('Computing OOF encodings...')
tr_gh_hr,  te_gh_hr  = oof_encode(train_lgb, test_lgb, ['geohash','hour'],      y_train, kf_full)
tr_gh5_hr, te_gh5_hr = oof_encode(train_lgb, test_lgb, ['gh5','hour'],          y_train, kf_full)
tr_gh,     te_gh     = oof_encode(train_lgb, test_lgb, ['geohash'],             y_train, kf_full)
tr_rt_hr,  te_rt_hr  = oof_encode(train_lgb, test_lgb, ['RoadType','hour'],     y_train, kf_full)
tr_geo_ts, te_geo_ts = oof_encode(train_lgb, test_lgb, ['geohash','time_slot'], y_train, kf_full, smoothing=5.0)
tr_gh5_ts, te_gh5_ts = oof_encode(train_lgb, test_lgb, ['gh5','time_slot'],     y_train, kf_full, smoothing=5.0)

def build_X(df, e1, e2, e3, e4, e5, e6):
    out = df[CAT_COLS + NUM_COLS].copy().values.astype(float)
    lag = df['demand_lag'].fillna(df['demand_lag'].median()).values
    return np.column_stack([out, lag, e1, e2, e3, e4, e5, e6])

X_tr = build_X(train_lgb, tr_gh_hr, tr_gh5_hr, tr_gh, tr_rt_hr, tr_geo_ts, tr_gh5_ts)
X_te = build_X(test_lgb,  te_gh_hr, te_gh5_hr, te_gh, te_rt_hr, te_geo_ts, te_gh5_ts)

# Proxy subsets
X_proxy_tr = X_tr[proxy_train_mask];  y_proxy_tr = y_train[proxy_train_mask]
X_proxy_val= X_tr[proxy_val_mask];   y_proxy_val= y_train[proxy_val_mask]
cat_proxy_tr = train_cat.loc[proxy_train_mask, CAT_FEAT_COLS].reset_index(drop=True)
cat_proxy_val= train_cat.loc[proxy_val_mask,  CAT_FEAT_COLS].reset_index(drop=True)
cat_feat_idx = list(range(len(CAT_COLS)))

# CatBoost extra cols
for df in [train_cat, test_cat]:
    df['time_slot']  = train_raw['time_slot'].values if df is train_cat else test_raw['time_slot'].values
    df['demand_lag'] = train_raw['demand_lag'].values if df is train_cat else test_raw['demand_lag'].values

best_params = {}

# ── LGB Optuna ────────────────────────────────────────────────────────────────
if MODE in ('lgb', 'all'):
    print(f'\nRunning LGB Optuna ({N_TRIALS_LGB} trials)...')
    def lgb_objective(trial):
        p = dict(
            objective='regression', metric='rmse', verbose=-1, n_jobs=4,
            random_state=SEED,
            n_estimators=3000,
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int('num_leaves', 31, 255),
            min_child_samples=trial.suggest_int('min_child_samples', 5, 50),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
            reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True),
        )
        m = lgb.LGBMRegressor(**p)
        m.fit(X_proxy_tr, y_proxy_tr,
              eval_set=[(X_proxy_val, y_proxy_val)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(9999)])
        pred = np.clip(m.predict(X_proxy_val), 0, 1)
        return r2_score(y_proxy_val, pred)

    study_lgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study_lgb.optimize(lgb_objective, n_trials=N_TRIALS_LGB, show_progress_bar=True)
    best_params['lgb'] = study_lgb.best_params
    print(f'LGB best proxy R²={study_lgb.best_value:.4f}')
    print(f'LGB best params: {study_lgb.best_params}')

# ── XGB Optuna ────────────────────────────────────────────────────────────────
if MODE in ('xgb', 'all'):
    print(f'\nRunning XGB Optuna ({N_TRIALS_XGB} trials)...')
    def xgb_objective(trial):
        p = dict(
            objective='reg:squarederror', verbosity=0, n_jobs=4,
            random_state=SEED, n_estimators=3000,
            early_stopping_rounds=50,
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            max_depth=trial.suggest_int('max_depth', 4, 10),
            min_child_weight=trial.suggest_int('min_child_weight', 5, 50),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
            reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True),
        )
        m = XGBRegressor(**p)
        m.fit(X_proxy_tr, y_proxy_tr, eval_set=[(X_proxy_val, y_proxy_val)], verbose=False)
        pred = np.clip(m.predict(X_proxy_val), 0, 1)
        return r2_score(y_proxy_val, pred)

    study_xgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study_xgb.optimize(xgb_objective, n_trials=N_TRIALS_XGB, show_progress_bar=True)
    best_params['xgb'] = study_xgb.best_params
    print(f'XGB best proxy R²={study_xgb.best_value:.4f}')
    print(f'XGB best params: {study_xgb.best_params}')

# ── CatBoost Optuna ───────────────────────────────────────────────────────────
if MODE in ('cat', 'all'):
    print(f'\nRunning CatBoost Optuna ({N_TRIALS_CAT} trials)...')
    def cat_objective(trial):
        p = dict(
            loss_function='RMSE', random_seed=SEED, verbose=0,
            early_stopping_rounds=50, iterations=3000,
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            depth=trial.suggest_int('depth', 5, 10),
            l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            min_data_in_leaf=trial.suggest_int('min_data_in_leaf', 1, 20),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bylevel=trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        )
        m = CatBoostRegressor(**p)
        m.fit(cat_proxy_tr, y_proxy_tr, cat_features=cat_feat_idx,
              eval_set=(cat_proxy_val, y_proxy_val), use_best_model=True)
        pred = np.clip(m.predict(cat_proxy_val), 0, 1)
        return r2_score(y_proxy_val, pred)

    study_cat = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study_cat.optimize(cat_objective, n_trials=N_TRIALS_CAT, show_progress_bar=True)
    best_params['cat'] = study_cat.best_params
    print(f'CatBoost best proxy R²={study_cat.best_value:.4f}')
    print(f'CatBoost best params: {study_cat.best_params}')

# Save best params
with open('optuna_proxy_params.json', 'w') as f:
    json.dump(best_params, f, indent=2)
print(f'\nSaved optuna_proxy_params.json')

# ── Train final models on FULL data with best params ─────────────────────────
print('\nTraining final models on full data...')
oof_lgb_p = np.zeros(n_train); test_lgb_p = np.zeros(len(test_raw))
oof_xgb_p = np.zeros(n_train); test_xgb_p = np.zeros(len(test_raw))
oof_cat_p = np.zeros(n_train); test_cat_p = np.zeros(len(test_raw))

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

if 'lgb' in best_params:
    lgb_final_p = dict(objective='regression', metric='rmse', n_estimators=3000,
                        n_jobs=-1, verbose=-1, random_state=SEED, **best_params['lgb'])
    print('Training LGB full...')
    for fold, (ti, vi) in enumerate(kf.split(X_tr)):
        m = lgb.LGBMRegressor(**lgb_final_p)
        m.fit(X_tr[ti], y_train[ti], eval_set=[(X_tr[vi], y_train[vi])],
              callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(9999)])
        oof_lgb_p[vi] = np.clip(m.predict(X_tr[vi]), 0, 1)
        test_lgb_p   += np.clip(m.predict(X_te), 0, 1) / N_FOLDS
        print(f'  Fold {fold+1}  R²={r2_score(y_train[vi], oof_lgb_p[vi]):.4f}  iter={m.best_iteration_}')
    print(f'LGB OOF R²={r2_score(y_train[oof_lgb_p>0], oof_lgb_p[oof_lgb_p>0]):.4f}')

if 'xgb' in best_params:
    xgb_final_p = dict(objective='reg:squarederror', n_estimators=3000,
                        early_stopping_rounds=150, n_jobs=-1, random_state=SEED, verbosity=0,
                        **best_params['xgb'])
    print('Training XGB full...')
    for fold, (ti, vi) in enumerate(kf.split(X_tr)):
        m = XGBRegressor(**xgb_final_p)
        m.fit(X_tr[ti], y_train[ti], eval_set=[(X_tr[vi], y_train[vi])], verbose=False)
        oof_xgb_p[vi] = np.clip(m.predict(X_tr[vi]), 0, 1)
        test_xgb_p   += np.clip(m.predict(X_te), 0, 1) / N_FOLDS
        print(f'  Fold {fold+1}  R²={r2_score(y_train[vi], oof_xgb_p[vi]):.4f}  iter={m.best_iteration}')
    print(f'XGB OOF R²={r2_score(y_train[oof_xgb_p>0], oof_xgb_p[oof_xgb_p>0]):.4f}')

if 'cat' in best_params:
    cat_final_p = dict(loss_function='RMSE', random_seed=SEED, verbose=0,
                        early_stopping_rounds=200, iterations=8000, **best_params['cat'])
    print('Training CatBoost full...')
    for fold, (ti, vi) in enumerate(kf.split(train_cat)):
        m = CatBoostRegressor(**cat_final_p)
        m.fit(train_cat.iloc[ti][CAT_FEAT_COLS], y_train[ti], cat_features=cat_feat_idx,
              eval_set=(train_cat.iloc[vi][CAT_FEAT_COLS], y_train[vi]), use_best_model=True)
        oof_cat_p[vi] = np.clip(m.predict(train_cat.iloc[vi][CAT_FEAT_COLS]), 0, 1)
        test_cat_p   += np.clip(m.predict(test_cat[CAT_FEAT_COLS]), 0, 1) / N_FOLDS
        print(f'  Fold {fold+1}  R²={r2_score(y_train[vi], oof_cat_p[vi]):.4f}  iter={m.best_iteration_}')
    print(f'CatBoost OOF R²={r2_score(y_train[oof_cat_p>0], oof_cat_p[oof_cat_p>0]):.4f}')

# Blend search on available models
_preds = [p for p, arr in [('lgb', test_lgb_p), ('xgb', test_xgb_p), ('cat', test_cat_p)]
          if arr.sum() > 0]
_oof = {'lgb': oof_lgb_p, 'xgb': oof_xgb_p, 'cat': oof_cat_p}
_test = {'lgb': test_lgb_p, 'xgb': test_xgb_p, 'cat': test_cat_p}

if len(_preds) == 3:
    best_r2, best_wl, best_wx = 0, 0.2, 0.1
    for wl in np.arange(0.05, 0.65, 0.05):
        for wx in np.arange(0.05, 0.65, 0.05):
            if wl+wx >= 0.95: continue
            wc = 1.0-wl-wx
            blend = wl*oof_lgb_p + wx*oof_xgb_p + wc*oof_cat_p
            r2 = r2_score(y_train, blend)
            if r2 > best_r2: best_r2, best_wl, best_wx = r2, wl, wx
    best_wc = 1.0-best_wl-best_wx
    test_ensemble = np.clip(best_wl*test_lgb_p + best_wx*test_xgb_p + best_wc*test_cat_p, 0, 1)
    print(f'\nEnsemble OOF R²={best_r2:.4f}  weights: LGB={best_wl:.2f} XGB={best_wx:.2f} CAT={best_wc:.2f}')
elif len(_preds) == 1:
    test_ensemble = _test[_preds[0]]

np.savez(
    'proxy_oof_preds.npz',
    y=y_train,
    time_slot=train_raw['time_slot'].values,
    day=train_raw['day'].values,
    timestamp=train_raw['timestamp'].values,
    lgb=oof_lgb_p,
    xgb=oof_xgb_p,
    cat=oof_cat_p,
)
np.savez(
    'proxy_test_preds.npz',
    time_slot=test_raw['time_slot'].values,
    lgb=test_lgb_p,
    xgb=test_xgb_p,
    cat=test_cat_p,
)

np.save(f'preds_proxy_{MODE}.npy', test_ensemble)
sub = pd.DataFrame({'Index': test_raw['Index'].values, 'demand': test_ensemble})
sub.to_csv(f'submission_proxy_{MODE}.csv', index=False)
print(f'Saved submission_proxy_{MODE}.csv  range=[{test_ensemble.min():.4f},{test_ensemble.max():.4f}]')
print(f'Total time: {(time.time()-t0)/60:.1f} min')
