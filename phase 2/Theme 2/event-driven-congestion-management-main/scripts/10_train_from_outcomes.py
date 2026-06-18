from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import mlflow
except Exception:
    mlflow = None

sys.path.append(str(Path(__file__).resolve().parents[1]))
from lib.logging_utils import get_logger
from lib.outcome_model import OUTCOME_MODEL_PATH, OUTCOMES_PATH, train_outcome_model
from lib.paths import ROOT, ensure_directories

LOGGER = get_logger("train_from_outcomes")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the congestion-duration model from operator-logged outcomes (learning loop)."
    )
    parser.add_argument("--outcomes", type=Path, default=OUTCOMES_PATH)
    parser.add_argument("--model", type=Path, default=OUTCOME_MODEL_PATH)
    args = parser.parse_args()
    ensure_directories()

    result = train_outcome_model(args.outcomes, args.model)
    if not result["trained"]:
        LOGGER.info("Outcome model not trained: %s", result["reason"])
        return
    LOGGER.info(
        "Trained outcome duration model on %s outcomes (MAE %.2f, blend_weight %.2f) -> %s",
        result["n_outcomes"],
        result["metrics"]["mae"],
        result["metrics"]["blend_weight"],
        result["model_path"],
    )

    if mlflow is not None:
        database_path = (ROOT / "mlflow.db").resolve().as_posix()
        mlflow.set_tracking_uri(f"sqlite:///{database_path}")
        mlflow.set_experiment("bengaluru_event_congestion")
        with mlflow.start_run(run_name="outcome_duration_model"):
            for key, value in result["metrics"].items():
                if isinstance(value, bool):
                    mlflow.log_param(key, value)
                elif isinstance(value, (int, float)):
                    mlflow.log_metric(key, float(value))
                else:
                    mlflow.log_param(key, str(value))
            if args.model.exists():
                mlflow.log_artifact(args.model)


if __name__ == "__main__":
    main()
