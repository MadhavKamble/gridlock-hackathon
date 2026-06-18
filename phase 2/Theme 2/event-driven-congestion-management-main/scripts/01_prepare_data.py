from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from lib.data_utils import engineer_features
from lib.logging_utils import get_logger
from lib.paths import DATA_DIR, ensure_directories

LOGGER = get_logger("prepare_data")


def prepare_data(input_path: Path, output_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    raw = pd.read_csv(input_path)
    required = {
        "id",
        "event_type",
        "latitude",
        "longitude",
        "event_cause",
        "requires_road_closure",
        "start_datetime",
        "status",
        "corridor",
        "priority",
        "created_date",
        "police_station",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    prepared = engineer_features(raw)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)
    LOGGER.info("Prepared %s rows at %s", len(prepared), output_path)
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Bengaluru gridlock event data.")
    parser.add_argument("--input", type=Path, default=Path("../cleaned_gridlock.csv"))
    parser.add_argument("--output", type=Path, default=DATA_DIR / "train_data.csv")
    args = parser.parse_args()
    ensure_directories()
    prepare_data(args.input, args.output)


if __name__ == "__main__":
    main()
