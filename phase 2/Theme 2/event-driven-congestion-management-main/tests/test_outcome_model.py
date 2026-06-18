from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lib.outcome_model import (
    BLEND_TARGET_OUTCOMES,
    MIN_OUTCOMES,
    blend_weight,
    build_training_frame,
    train_outcome_model,
)


def test_blend_weight_gates_and_ramps() -> None:
    assert blend_weight(0) == 0.0
    assert blend_weight(MIN_OUTCOMES - 1) == 0.0
    assert 0.0 < blend_weight(MIN_OUTCOMES) <= 1.0
    assert blend_weight(BLEND_TARGET_OUTCOMES) == 1.0
    assert blend_weight(BLEND_TARGET_OUTCOMES * 5) == 1.0  # capped


def _outcome(duration: float | None, **event_overrides) -> dict:
    event = {
        "event_type": "unplanned",
        "start_datetime": "2024-03-07T14:00:00+00:00",
        "priority": "Medium",
        "corridor": "Non-corridor",
        "requires_road_closure": False,
        "event_cause": "accident",
        "latitude": 12.9352,
        "longitude": 77.6245,
    }
    event.update(event_overrides)
    return {"actual_duration_min": duration, "prediction": {"event": event}}


def test_build_training_frame_skips_invalid_rows() -> None:
    outcomes = [
        _outcome(75.0),
        _outcome(0.0),       # non-positive duration -> skipped
        _outcome(None),      # missing label -> skipped
        {"prediction": {"event": {}}},  # missing fields -> skipped
    ]
    X, y = build_training_frame(outcomes)
    assert len(X) == 1
    assert len(y) == 1
    assert y.iloc[0] == 75.0


def test_train_outcome_model_gates_on_minimum(tmp_path) -> None:
    path = tmp_path / "outcomes.jsonl"
    path.write_text("", encoding="utf-8")
    result = train_outcome_model(outcomes_path=path, model_path=tmp_path / "m.pkl")
    assert result["trained"] is False
    assert result["n_outcomes"] == 0
    assert not (tmp_path / "m.pkl").exists()
