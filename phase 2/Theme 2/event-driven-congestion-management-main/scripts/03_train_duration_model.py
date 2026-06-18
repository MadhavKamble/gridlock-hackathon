from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

sys.path.append(str(Path(__file__).resolve().parents[1]))
from lib.data_utils import MODEL_FEATURES
from lib.logging_utils import get_logger
from lib.paths import DATA_DIR, MODEL_DIR, ensure_directories

LOGGER = get_logger("train_duration_model")


def train_model(train_path: Path, model_path: Path, metrics_path: Path) -> dict[str, float | str | int]:
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}. Run 01_prepare_data.py first.")
    data = pd.read_csv(train_path)
    categorical_levels: dict[str, list[str]] = {}
    for col in ("corridor", "event_cause"):
        data[col] = data[col].astype("category")
        categorical_levels[col] = list(data[col].cat.categories)
    data = data.dropna(subset=["report_creation_delay_min"])
    X = data[MODEL_FEATURES]
    y = data["report_creation_delay_min"].astype(float)
    if len(data) < 10:
        raise ValueError("At least 10 rows are required to train a reliable model.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if XGBRegressor is not None:
        model = XGBRegressor(
            n_estimators=350,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            enable_categorical=True,
            tree_method="hist",
            random_state=42,
            n_jobs=4,
        )
        model_type = "xgboost"
    else:
        LOGGER.warning("xgboost is not installed; using RandomForestRegressor fallback for this run.")
        numeric_features = [feature for feature in MODEL_FEATURES if feature not in {"corridor", "event_cause"}]
        model = Pipeline(
            steps=[
                (
                    "preprocessor",
                    ColumnTransformer(
                        transformers=[
                            ("categorical", OneHotEncoder(handle_unknown="ignore"), ["corridor", "event_cause"]),
                            ("numeric", "passthrough", numeric_features),
                        ]
                    ),
                ),
                ("regressor", RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1)),
            ]
        )
        model_type = "random_forest_fallback"
    model.fit(X_train, y_train)
    predictions = np.maximum(model.predict(X_test), 1)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(root_mean_squared_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "model_type": model_type,
        "target": "report_creation_delay_min",
        "training_rows": int(len(data)),
        "warning": "The source dataset has no event end/resolution timestamp; this model does not predict congestion duration.",
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": MODEL_FEATURES,
            "model_type": model_type,
            "target": "report_creation_delay_min",
            "categorical_levels": categorical_levels,
        },
        model_path,
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info("Saved model to %s with metrics %s", model_path, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the reporting-delay model available from the source timestamps.")
    parser.add_argument("--train", type=Path, default=DATA_DIR / "train_data.csv")
    parser.add_argument("--model", type=Path, default=MODEL_DIR / "duration_model.pkl")
    parser.add_argument("--metrics", type=Path, default=MODEL_DIR / "duration_metrics.json")
    args = parser.parse_args()
    ensure_directories()
    train_model(args.train, args.model, args.metrics)


if __name__ == "__main__":
    main()
