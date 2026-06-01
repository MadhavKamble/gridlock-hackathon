"""
Per-time-slot blending weights using proxy split (day48 daytime as validation).

Requires:
  - proxy_oof_preds.npz (from optuna_proxy.py)
  - proxy_test_preds.npz
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def fit_weights(X, y, l2=1e-3):
    xtx = X.T @ X + l2 * np.eye(X.shape[1])
    xty = X.T @ y
    w = np.linalg.solve(xtx, xty)
    w = np.clip(w, 0, None)
    if w.sum() == 0:
        return np.ones_like(w) / len(w)
    return w / w.sum()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oof', default='proxy_oof_preds.npz', help='OOF predictions npz')
    ap.add_argument('--test', default='proxy_test_preds.npz', help='Test predictions npz')
    ap.add_argument('--min-rows', type=int, default=150, help='Min rows per time_slot')
    ap.add_argument('--out', default=None, help='Output submission path')
    args = ap.parse_args()

    oof = np.load(args.oof)
    test = np.load(args.test)

    train_raw = pd.read_csv(next((p for p in [Path('data'), Path('dataset')] if p.exists()), Path('data')) / 'train.csv')
    test_raw  = pd.read_csv(next((p for p in [Path('data'), Path('dataset')] if p.exists()), Path('data')) / 'test.csv')

    test_ts = set(test_raw['timestamp'].unique())
    proxy_val_mask = (train_raw['day'].values == 48) & train_raw['timestamp'].isin(test_ts).values

    model_keys = [k for k in oof.files if k not in ('y','time_slot','day','timestamp')]
    if not model_keys:
        raise ValueError('No model predictions found in proxy_oof_preds.npz')

    X_oof = np.vstack([oof[k] for k in model_keys]).T
    y = oof['y']
    ts_train = oof['time_slot']

    X_proxy = X_oof[proxy_val_mask]
    y_proxy = y[proxy_val_mask]

    global_w = fit_weights(X_proxy, y_proxy)
    print(f'Global weights: {dict(zip(model_keys, global_w.round(3)))}')

    # Per-time-slot weights
    weights = {}
    for slot in range(96):
        slot_mask = proxy_val_mask & (ts_train == slot)
        if slot_mask.sum() < args.min_rows:
            weights[slot] = global_w
            continue
        X_slot = X_oof[slot_mask]
        y_slot = y[slot_mask]
        weights[slot] = fit_weights(X_slot, y_slot)

    # Apply to test
    X_test = np.vstack([test[k] for k in model_keys]).T
    ts_test = test['time_slot']
    blended = np.zeros(len(X_test))
    for slot in range(96):
        mask = ts_test == slot
        if mask.any():
            blended[mask] = X_test[mask] @ weights[slot]

    out_dir = Path('submissions/experiments')
    if not out_dir.exists():
        out_dir = Path('.')
    out_path = Path(args.out) if args.out else out_dir / 'submission_time_slot_blend.csv'

    sub = pd.DataFrame({'Index': test_raw['Index'], 'demand': np.clip(blended, 0, 1)})
    sub.to_csv(out_path, index=False)
    print(f'Saved {out_path}')

if __name__ == '__main__':
    main()
