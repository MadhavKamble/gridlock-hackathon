from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from lib.paths import DATA_DIR, MODEL_DIR, NETWORK_DIR, PREDICTIONS_DIR, ROOT, ensure_directories


REQUIRED_RUNTIME_FILES = (
    DATA_DIR / "train_data.csv",
    NETWORK_DIR / "bangalore_graph.graphml",
    MODEL_DIR / "duration_model.pkl",
)


def ensure_runtime_artifacts() -> dict[str, Any]:
    """Create missing runtime artifacts for first-run public deployments."""
    ensure_directories()
    missing = [path for path in REQUIRED_RUNTIME_FILES if not path.exists()]
    if not missing:
        return {"bootstrapped": False, "missing": []}

    raw_csv = _find_raw_csv()
    if raw_csv is None:
        return {
            "bootstrapped": False,
            "missing": [str(path.relative_to(ROOT)) for path in missing],
            "warning": "No cleaned_gridlock.csv found. Add it to data/cleaned_gridlock.csv or mount it beside the project.",
        }

    steps = [
        ["01_prepare_data.py", "--input", str(raw_csv)],
        ["02_build_network.py"],
        ["03_train_duration_model.py"],
    ]
    executed = []
    for command in steps:
        _run_script(command)
        executed.append(command[0])

    if not (PREDICTIONS_DIR / "latest_prediction.json").exists():
        for command in (
            ["04_predict_impact.py"],
            ["05_manpower_optimizer.py"],
            ["06_barricade_simulator.py"],
            ["07_diversion_routes.py"],
            ["08_generate_dashboard.py"],
        ):
            _run_script(command)
            executed.append(command[0])

    return {"bootstrapped": True, "executed": executed, "raw_csv": str(raw_csv)}


def _find_raw_csv() -> Path | None:
    candidates = (
        DATA_DIR / "cleaned_gridlock.csv",
        ROOT / "cleaned_gridlock.csv",
        ROOT.parent / "cleaned_gridlock.csv",
        Path("/cleaned_gridlock.csv"),
    )
    return next((path for path in candidates if path.exists()), None)


def _run_script(args: list[str]) -> None:
    command = [sys.executable, str(ROOT / "scripts" / args[0]), *args[1:]]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-40:])
        raise RuntimeError(f"Bootstrap step {args[0]} failed.\n{tail}")
