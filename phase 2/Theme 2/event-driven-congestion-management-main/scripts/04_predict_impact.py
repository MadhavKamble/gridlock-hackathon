from __future__ import annotations

import argparse
from datetime import datetime, timezone
import sys
from pathlib import Path
from typing import Any

import joblib
import networkx as nx
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from lib.data_utils import EventInput, MODEL_FEATURES, event_to_frame
from lib.logging_utils import get_logger
from lib.outcome_model import blend_weight, load_outcome_model, predict_outcome_duration
from lib.network_utils import edge_record, haversine_m, load_graph, nearest_node, write_json
from lib.paths import MODEL_DIR, NETWORK_DIR, PREDICTIONS_DIR, ensure_directories

LOGGER = get_logger("predict_impact")


CAUSE_BASE_MIN = {
    "vehicle_breakdown": 35,
    "accident": 75,
    "tree_fall": 95,
    "water_logging": 70,
    "pot_holes": 45,
    "public_event": 180,
    "procession": 150,
    "protest": 135,
    "construction": 120,
    "congestion": 50,
    "others": 55,
}
PRIORITY_FACTOR = {"low": 0.75, "medium": 1.0, "high": 1.35, "critical": 1.75}


def estimate_operational_impact(event: EventInput, road_context: dict[str, Any] | None = None) -> tuple[float, dict[str, Any]]:
    timestamp = datetime.fromisoformat(event.start_datetime.replace("Z", "+00:00"))
    hour = timestamp.hour
    cause = event.event_cause.lower()
    base = float(CAUSE_BASE_MIN.get(cause, CAUSE_BASE_MIN["others"]))
    priority_factor = PRIORITY_FACTOR.get(event.priority.lower(), 1.0)
    closure_factor = 1.4 if event.requires_road_closure else 1.0
    peak_factor = 1.25 if hour in {8, 9, 10, 17, 18, 19, 20} else 1.0
    planned_factor = 1.15 if event.event_type.lower() == "planned" else 1.0
    corridor_factor = 1.12 if event.corridor.lower() not in {"non-corridor", "unknown", ""} else 1.0
    road_context_factor = float((road_context or {}).get("duration_factor", 1.0))
    estimate = base * priority_factor * closure_factor * peak_factor * planned_factor * corridor_factor * road_context_factor
    estimate = round(min(max(estimate, 10.0), 480.0), 2)
    return estimate, {
        "method": "operational_risk_estimator_v1",
        "reason": "The dataset does not contain event end/resolution timestamps.",
        "base_cause_minutes": base,
        "priority_factor": priority_factor,
        "closure_factor": closure_factor,
        "peak_hour_factor": peak_factor,
        "planned_event_factor": planned_factor,
        "corridor_factor": corridor_factor,
        "road_context_factor": road_context_factor,
        "range_min": round(estimate * 0.7, 2),
        "range_max": round(estimate * 1.35, 2),
    }


def affected_edges(
    graph: Any,
    event: EventInput,
    duration_min: float,
    road_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    corridor = event.corridor.lower()
    radius_m = _impact_radius_m(event, duration_min, road_context or {})
    seed = nearest_node(graph, event.latitude, event.longitude)
    undirected = graph.to_undirected(as_view=True)
    distances = nx.single_source_dijkstra_path_length(undirected, seed, cutoff=radius_m, weight="length")
    local_nodes = set(distances)
    local_graph = graph.subgraph(local_nodes)
    records: list[dict[str, Any]] = []
    corridor_records: list[dict[str, Any]] = []
    for u, v, key, data in local_graph.edges(keys=True, data=True):
        name = str(data.get("name", "")).lower()
        if corridor != "non-corridor" and corridor in name:
            record = edge_record(graph, u, v, key, data)
            record["distance_from_event_m"] = round(min(distances.get(u, radius_m), distances.get(v, radius_m)), 1)
            record["impact_level"] = "high"
            corridor_records.append(record)
    if corridor_records:
        nearest_corridor_m = min(record["distance_from_event_m"] for record in corridor_records)
        if nearest_corridor_m <= 1_500:
            records = corridor_records
        else:
            LOGGER.warning(
                "Requested corridor '%s' is %.0fm from event; using local roads instead.",
                event.corridor,
                nearest_corridor_m,
            )
    if not records:
        for u, v, key, data in local_graph.edges(keys=True, data=True):
            distance = min(distances.get(u, radius_m), distances.get(v, radius_m))
            record = edge_record(graph, u, v, key, data)
            record["distance_from_event_m"] = round(distance, 1)
            record["impact_level"] = "high" if distance < radius_m * 0.35 else "medium"
            records.append(record)
    if not records:
        for u, v, key, data in graph.out_edges(seed, keys=True, data=True):
            records.append(edge_record(graph, u, v, key, data))
        for u, v, key, data in graph.in_edges(seed, keys=True, data=True):
            records.append(edge_record(graph, u, v, key, data))
    unique = {(r["u"], r["v"], r["key"]): r for r in records}
    enriched = []
    for record in unique.values():
        speed = _predicted_edge_speed(record, duration_min)
        record["predicted_speed_kph"] = round(speed, 2)
        record["free_flow_speed_kph"] = round(_free_flow_speed(record), 2)
        enriched.append(record)
    return sorted(enriched, key=lambda row: row.get("distance_from_event_m", 0))[:350]


def analyze_road_context(graph: Any, event: EventInput) -> dict[str, Any]:
    """Classify whether the event is on a through-road or a contained local access road."""
    seed = nearest_node(graph, event.latitude, event.longitude)
    incident_edges = list(graph.out_edges(seed, keys=True, data=True)) + list(graph.in_edges(seed, keys=True, data=True))
    highway_values = sorted(
        {
            str(value).lower()
            for _, _, _, data in incident_edges
            for value in _as_list(data.get("highway", "unknown"))
            if value
        }
    )
    service_values = sorted(
        {
            str(value).lower()
            for _, _, _, data in incident_edges
            for value in _as_list(data.get("service"))
            if value
        }
    )
    names = sorted({str(data.get("name")) for _, _, _, data in incident_edges if data.get("name")})
    street_count = int(float(graph.nodes[seed].get("street_count", graph.degree(seed)) or 0))
    unique_neighbors = len(set(graph.predecessors(seed)) | set(graph.successors(seed)))
    arterial_classes = {"motorway", "trunk", "primary", "secondary", "tertiary"}
    local_classes = {"residential", "service", "living_street", "unclassified", "road", "unknown"}
    has_arterial = bool(set(highway_values) & arterial_classes)
    is_local = not has_arterial and (not highway_values or set(highway_values).issubset(local_classes))
    is_terminal = street_count <= 1 or unique_neighbors <= 1
    requires_closure = bool(event.requires_road_closure)

    if is_terminal and is_local and not requires_closure:
        classification = "contained_terminal_local_road"
        duration_factor = 0.42
        radius_cap_m = 220.0
        response_factor = 0.45
        explanation = "The event is on a terminal local/residential access road, so spillover should stay near the pin."
    elif is_local and not requires_closure:
        classification = "local_access_road"
        duration_factor = 0.62
        radius_cap_m = 450.0
        response_factor = 0.65
        explanation = "The event is on a local road without closure, so area-wide congestion is limited."
    elif has_arterial:
        classification = "through_road"
        duration_factor = 1.0
        radius_cap_m = 4500.0
        response_factor = 1.0
        explanation = "The event touches a through-road, so broader spillover is possible."
    else:
        classification = "mixed_local_context"
        duration_factor = 0.8
        radius_cap_m = 900.0
        response_factor = 0.8
        explanation = "The nearby network is mixed, so the impact is partially contained."

    return {
        "nearest_node": str(seed),
        "street_count": street_count,
        "unique_neighbor_count": unique_neighbors,
        "highway_classes": highway_values,
        "service_classes": service_values,
        "road_names": names,
        "has_arterial_access": has_arterial,
        "is_terminal": is_terminal,
        "classification": classification,
        "duration_factor": duration_factor,
        "radius_cap_m": radius_cap_m,
        "response_factor": response_factor,
        "explanation": explanation,
    }


def _impact_radius_m(event: EventInput, duration_min: float, road_context: dict[str, Any]) -> float:
    base_radius = 700.0 + duration_min * 12.0
    if event.requires_road_closure:
        base_radius *= 1.25
    if road_context:
        base_radius = min(base_radius, float(road_context.get("radius_cap_m", base_radius)))
        if road_context.get("classification") == "contained_terminal_local_road":
            return max(120.0, base_radius)
        if road_context.get("classification") == "local_access_road":
            return max(220.0, base_radius)
    return min(4500.0, max(1000.0, base_radius))


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _free_flow_speed(edge: dict[str, Any]) -> float:
    length = float(edge.get("length_m", 0.0) or 0.0)
    travel_time = float(edge.get("travel_time_s", 0.0) or 0.0)
    if length > 0 and travel_time > 0:
        return max((length / travel_time) * 3.6, 1.0)
    return 35.0


def _predicted_edge_speed(edge: dict[str, Any], duration_min: float) -> float:
    free_flow = _free_flow_speed(edge)
    level = str(edge.get("impact_level", "medium")).lower()
    impact_factor = 0.28 if level == "high" else 0.55
    duration_factor = max(0.35, 1.0 - min(duration_min, 240.0) / 480.0)
    return max(free_flow * min(impact_factor, duration_factor), 1.0)


def predict(event: EventInput, model_path: Path, graph_path: Path, output_path: Path) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run 03_train_duration_model.py first.")
    bundle = joblib.load(model_path)
    model = bundle["model"]
    categorical_levels = bundle.get("categorical_levels", {})
    frame = event_to_frame(event)
    for col in ("corridor", "event_cause"):
        levels = categorical_levels.get(col)
        if levels:
            # Reuse the exact categories seen at training time so unseen values
            # become NaN (treated as missing) instead of crashing XGBoost.
            frame[col] = pd.Categorical(frame[col], categories=levels)
        else:
            frame[col] = frame[col].astype("category")
    reporting_delay = float(max(model.predict(frame[MODEL_FEATURES])[0], 0.0))
    graph = load_graph(graph_path)
    road_context = analyze_road_context(graph, event)
    estimator_duration, methodology = estimate_operational_impact(event, road_context)
    duration = estimator_duration
    # Learning loop: if a duration model trained from logged outcomes exists, blend it with the
    # estimator, trusting the model more as outcomes accumulate (blend_weight). Estimator is the floor.
    outcome_bundle = load_outcome_model()
    if outcome_bundle is not None:
        n_outcomes = int(outcome_bundle.get("n_outcomes", 0))
        weight = blend_weight(n_outcomes)
        if weight > 0:
            learned_duration = predict_outcome_duration(outcome_bundle, event)
            duration = round(min(max(weight * learned_duration + (1 - weight) * estimator_duration, 10.0), 480.0), 2)
            methodology = {
                **methodology,
                "method": "learned_estimator_blend_v1",
                "estimator_duration_min": round(estimator_duration, 2),
                "learned_duration_min": round(learned_duration, 2),
                "blend_weight": round(weight, 3),
                "learned_n_outcomes": n_outcomes,
                "range_min": round(duration * 0.7, 2),
                "range_max": round(duration * 1.35, 2),
            }
    edges = affected_edges(graph, event, duration, road_context)
    intersections = sorted({edge["u"] for edge in edges} | {edge["v"] for edge in edges})
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "event": event.__dict__,
        "predicted_duration_min": round(duration, 2),
        "predicted_duration_range_min": methodology["range_min"],
        "predicted_duration_range_max": methodology["range_max"],
        "duration_methodology": methodology,
        "predicted_reporting_delay_min": round(reporting_delay, 2),
        "road_context": road_context,
        "affected_edges": edges,
        "intersections": intersections,
        "nearest_node": str(nearest_node(graph, event.latitude, event.longitude)),
    }
    write_json(output_path, payload)
    LOGGER.info("Saved prediction with %s edges and %s intersections to %s", len(edges), len(intersections), output_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict event traffic impact.")
    parser.add_argument("--event-type", default="unplanned")
    parser.add_argument("--start-datetime", default=datetime.now(timezone.utc).isoformat())
    parser.add_argument("--priority", default="High")
    parser.add_argument("--corridor", default="Outer Ring Road")
    parser.add_argument("--requires-road-closure", action="store_true")
    parser.add_argument("--event-cause", default="accident")
    parser.add_argument("--latitude", type=float, default=12.9352)
    parser.add_argument("--longitude", type=float, default=77.6245)
    parser.add_argument("--model", type=Path, default=MODEL_DIR / "duration_model.pkl")
    parser.add_argument("--graph", type=Path, default=NETWORK_DIR / "bangalore_graph.graphml")
    parser.add_argument("--output", type=Path, default=PREDICTIONS_DIR / "latest_prediction.json")
    args = parser.parse_args()
    ensure_directories()
    event = EventInput(
        event_type=args.event_type,
        start_datetime=args.start_datetime,
        priority=args.priority,
        corridor=args.corridor,
        requires_road_closure=args.requires_road_closure,
        event_cause=args.event_cause,
        latitude=args.latitude,
        longitude=args.longitude,
    )
    predict(event, args.model, args.graph, args.output)


if __name__ == "__main__":
    main()
