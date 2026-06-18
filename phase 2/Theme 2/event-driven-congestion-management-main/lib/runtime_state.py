from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lib.paths import ROOT

RUNTIME_DIR = ROOT / "output" / "runtime"
LOG_PATH = RUNTIME_DIR / "pipeline.log"
STATE_PATH = RUNTIME_DIR / "pipeline_state.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_state(
    steps: list[tuple[str, str]],
    run_id: str | None = None,
    run_type: str = "pipeline",
    message: str = "Ready",
    status: str = "idle",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "run_type": run_type,
        "status": status,
        "current_step": None,
        "started_at": utc_now() if status == "running" else None,
        "finished_at": None,
        "elapsed_seconds": 0,
        "message": message,
        "steps": [
            {
                "id": step_id,
                "name": name,
                "status": "pending",
                "started_at": None,
                "finished_at": None,
                "duration_seconds": None,
                "return_code": None,
            }
            for step_id, name in steps
        ],
    }


def public_state(state: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in state.items() if not key.startswith("_")}


def read_state(default_steps: list[tuple[str, str]]) -> dict[str, Any]:
    if not STATE_PATH.exists():
        return make_state(default_steps)
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return make_state(default_steps)


def write_state(state: dict[str, Any]) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(public_state(state), indent=2), encoding="utf-8")


def reset_log() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")


def append_log(message: str, level: str = "INFO") -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp} | {level:<7} | {message.rstrip()}\n")
