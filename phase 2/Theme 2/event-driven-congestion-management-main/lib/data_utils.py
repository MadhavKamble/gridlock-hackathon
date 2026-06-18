from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


EVENT_TYPE_MAP = {"planned": 1, "unplanned": 0}
PRIORITY_MAP = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4, "low": 1, "medium": 2, "high": 3, "critical": 4}


@dataclass(frozen=True)
class EventInput:
    event_type: str
    start_datetime: str
    priority: str
    corridor: str
    requires_road_closure: bool
    event_cause: str
    latitude: float
    longitude: float


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["start_datetime"] = pd.to_datetime(data["start_datetime"], utc=True, format="mixed", errors="coerce")
    data["created_date"] = pd.to_datetime(data["created_date"], utc=True, format="mixed", errors="coerce")
    data = data.dropna(subset=["start_datetime", "created_date", "latitude", "longitude"])
    data["report_creation_delay_min"] = (data["created_date"] - data["start_datetime"]).dt.total_seconds() / 60.0
    data["report_creation_delay_min"] = data["report_creation_delay_min"].where(
        data["report_creation_delay_min"].between(0, 60)
    )
    data["actual_duration_min"] = np.nan
    data["hour"] = data["start_datetime"].dt.hour.astype(int)
    data["day_of_week"] = data["start_datetime"].dt.dayofweek.astype(int)
    data["month"] = data["start_datetime"].dt.month.astype(int)
    data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)
    data["event_type_encoded"] = data["event_type"].astype(str).str.lower().map(EVENT_TYPE_MAP).fillna(0).astype(int)
    data["priority_encoded"] = data["priority"].astype(str).map(PRIORITY_MAP).fillna(2).astype(int)
    data["requires_road_closure_encoded"] = data["requires_road_closure"].apply(parse_bool).astype(int)
    data["corridor"] = data["corridor"].fillna("Unknown").astype("category")
    data["event_cause"] = data["event_cause"].fillna("others").astype("category")
    return data


def event_to_frame(event: EventInput) -> pd.DataFrame:
    row = {
        "event_type": event.event_type,
        "start_datetime": event.start_datetime,
        "priority": event.priority,
        "corridor": event.corridor,
        "requires_road_closure": event.requires_road_closure,
        "event_cause": event.event_cause,
        "latitude": event.latitude,
        "longitude": event.longitude,
        "created_date": event.start_datetime,
    }
    data = engineer_features(pd.DataFrame([row]))
    return data


MODEL_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "event_type_encoded",
    "priority_encoded",
    "requires_road_closure_encoded",
    "corridor",
    "event_cause",
    "latitude",
    "longitude",
]
