"""Learned duration model trained from operator-logged outcomes.

This closes the post-event learning loop: each logged outcome in ``data/outcomes.jsonl`` carries the
real ``actual_duration_min`` that the source dataset never recorded. Once enough outcomes accumulate,
we train a genuine *congestion-duration* model on that real target and let ``04_predict_impact`` blend
it with the transparent operational risk estimator — trusting the model more as evidence grows.

See docs/dev/LEARNING_LOOP.md and docs/dev/DECISIONS.md (D1).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - xgboost is a hard dependency in practice
    XGBRegressor = None

from lib.data_utils import EventInput, MODEL_FEATURES, event_to_frame
from lib.paths import DATA_DIR, MODEL_DIR

# Below MIN_OUTCOMES we don't train at all — too little signal. At/above BLEND_TARGET_OUTCOMES we
# trust the learned model fully; in between we blend linearly with the estimator (see blend_weight).
MIN_OUTCOMES = 30
BLEND_TARGET_OUTCOMES = 200

OUTCOMES_PATH = DATA_DIR / "outcomes.jsonl"
OUTCOME_MODEL_PATH = MODEL_DIR / "duration_outcome_model.pkl"

_EVENT_FIELDS = (
    "event_type",
    "start_datetime",
    "priority",
    "corridor",
    "requires_road_closure",
    "event_cause",
    "latitude",
    "longitude",
)


def load_outcomes(path: Path = OUTCOMES_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def blend_weight(n_outcomes: int) -> float:
    """Confidence in the learned model: 0 below MIN_OUTCOMES, ramping to 1 at BLEND_TARGET_OUTCOMES."""
    if n_outcomes < MIN_OUTCOMES:
        return 0.0
    return float(min(1.0, n_outcomes / BLEND_TARGET_OUTCOMES))


def build_training_frame(outcomes: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.Series]:
    """Turn logged outcomes into a (features, label) training set keyed on real duration."""
    frames: list[pd.DataFrame] = []
    labels: list[float] = []
    for record in outcomes:
        duration = record.get("actual_duration_min")
        event = (record.get("prediction") or {}).get("event") or {}
        if duration is None or float(duration) <= 0:
            continue
        if not all(field in event for field in _EVENT_FIELDS):
            continue
        try:
            frame = event_to_frame(EventInput(**{field: event[field] for field in _EVENT_FIELDS}))
        except (TypeError, ValueError):
            continue
        if frame.empty:
            continue
        frames.append(frame[MODEL_FEATURES])
        labels.append(float(duration))
    if not frames:
        return pd.DataFrame(columns=MODEL_FEATURES), pd.Series(dtype=float)
    X = pd.concat(frames, ignore_index=True)
    return X, pd.Series(labels, dtype=float)


def train_outcome_model(
    outcomes_path: Path = OUTCOMES_PATH,
    model_path: Path = OUTCOME_MODEL_PATH,
) -> dict[str, Any]:
    """Train a duration model from logged outcomes when there are enough of them.

    Returns a status dict; only writes a model file when training actually happens.
    """
    outcomes = load_outcomes(outcomes_path)
    X, y = build_training_frame(outcomes)
    n = int(len(y))
    if n < MIN_OUTCOMES:
        return {
            "trained": False,
            "reason": f"insufficient outcomes ({n} < {MIN_OUTCOMES})",
            "n_outcomes": n,
        }
    if XGBRegressor is None:
        return {"trained": False, "reason": "xgboost unavailable", "n_outcomes": n}

    categorical_levels: dict[str, list[str]] = {}
    for col in ("corridor", "event_cause"):
        X[col] = X[col].astype("category")
        categorical_levels[col] = list(X[col].cat.categories)

    # Small datasets early in the loop's life: keep a held-out split only when it's meaningful.
    if n >= 4 * MIN_OUTCOMES:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        enable_categorical=True,
        tree_method="hist",
        random_state=42,
        n_jobs=4,
    )
    model.fit(X_train, y_train)
    predictions = np.maximum(model.predict(X_test), 1.0)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(root_mean_squared_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)) if len(set(y_test)) > 1 else 0.0,
        "target": "actual_duration_min",
        "n_outcomes": n,
        "blend_weight": blend_weight(n),
        "held_out": n >= 4 * MIN_OUTCOMES,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": MODEL_FEATURES,
            "categorical_levels": categorical_levels,
            "target": "actual_duration_min",
            "n_outcomes": n,
            "metrics": metrics,
        },
        model_path,
    )
    return {"trained": True, "n_outcomes": n, "metrics": metrics, "model_path": str(model_path)}


def load_outcome_model(model_path: Path = OUTCOME_MODEL_PATH) -> dict[str, Any] | None:
    if not model_path.exists():
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None


def predict_outcome_duration(bundle: dict[str, Any], event: EventInput) -> float:
    """Predict duration (minutes) for an event using a loaded outcome-model bundle."""
    frame = event_to_frame(event)
    categorical_levels = bundle.get("categorical_levels", {})
    for col in ("corridor", "event_cause"):
        levels = categorical_levels.get(col)
        frame[col] = pd.Categorical(frame[col], categories=levels) if levels else frame[col].astype("category")
    value = float(bundle["model"].predict(frame[bundle.get("features", MODEL_FEATURES)])[0])
    return max(value, 1.0)
