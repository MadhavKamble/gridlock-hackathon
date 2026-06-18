from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx

try:
    from ortools.sat.python import cp_model
except Exception:
    cp_model = None

sys.path.append(str(Path(__file__).resolve().parents[1]))
from lib.logging_utils import get_logger
from lib.network_utils import haversine_m, load_graph, read_json, write_json
from lib.paths import NETWORK_DIR, PREDICTIONS_DIR, ensure_directories

LOGGER = get_logger("manpower_optimizer")


def optimize(prediction_path: Path, graph_path: Path, output_path: Path, officers: int) -> dict[str, Any]:
    prediction = read_json(prediction_path)
    graph = load_graph(graph_path)
    intersections = prediction.get("intersections", [])
    if not intersections:
        raise ValueError("Prediction contains no intersections to optimize.")
    candidates = [node for node in graph.nodes if str(node) in set(intersections)]
    event = prediction.get("event", {})
    affected = prediction.get("affected_edges", [])
    incident_count = {
        str(node): sum(1 for edge in affected if edge["u"] == str(node) or edge["v"] == str(node))
        for node in candidates
    }
    weights = {}
    for node in candidates:
        distance = haversine_m(
            float(event["latitude"]),
            float(event["longitude"]),
            float(graph.nodes[node]["y"]),
            float(graph.nodes[node]["x"]),
        )
        proximity = 1.0 / (1.0 + distance / 500.0)
        degree = min(float(graph.degree(node)), 8.0) / 8.0
        incident = min(incident_count[str(node)], 8) / 8.0
        weights[str(node)] = max(100, int(10_000 * (0.5 * proximity + 0.3 * incident + 0.2 * degree)))
    road_context = prediction.get("road_context", {})
    staffing_model = _staffing_model(event, float(prediction.get("predicted_duration_min", 0.0)), road_context)
    max_per_intersection = 4 if str(event.get("priority", "")).lower() in {"high", "critical"} else 3
    capacity_limit = len(candidates) * max_per_intersection
    demand_limit = math.ceil(max(officers, 0) * staffing_model["demand_factor"])
    deployable = min(max(officers, 0), capacity_limit, demand_limit)
    reserve = max(0, officers - deployable)
    if deployable == 0:
        payload = {
            "available_officers": officers,
            "assigned_officers": 0,
            "reserve_officers": max(0, officers),
            "coverage_score": 0,
            "optimizer": "demand_limited",
            "staffing_model": staffing_model,
            "deployment": [],
        }
        write_json(output_path, payload)
        LOGGER.info("Saved demand-limited manpower plan with no field deployment to %s", output_path)
        return payload
    if cp_model is None:
        LOGGER.warning("ortools is not installed; using greedy centrality fallback for this run.")
        allocation = {str(node): 0 for node in candidates}
        slots = sorted(
            (
                (weights[str(node)] * marginal, str(node))
                for node in candidates
                for marginal in (1.0, 0.72, 0.48, 0.3)[:max_per_intersection]
            ),
            reverse=True,
        )[:deployable]
        for _, node_id in slots:
            allocation[node_id] += 1
        deployment = _deployment(graph, candidates, weights, allocation)
        payload = {
            "available_officers": officers,
            "assigned_officers": sum(allocation.values()),
            "reserve_officers": reserve,
            "coverage_score": sum(item["deployment_score"] * item["recommended_officers"] for item in deployment),
            "optimizer": "greedy_fallback",
            "staffing_model": staffing_model,
            "deployment": deployment,
        }
        write_json(output_path, payload)
        LOGGER.info("Saved greedy manpower plan with %s officers to %s", len(deployment), output_path)
        return payload
    model = cp_model.CpModel()
    marginal = (100, 72, 48, 30)[:max_per_intersection]
    slots = {
        (str(node), slot): model.NewBoolVar(f"deploy_{node}_{slot}")
        for node in candidates
        for slot in range(max_per_intersection)
    }
    for node in candidates:
        for slot in range(1, max_per_intersection):
            model.Add(slots[(str(node), slot)] <= slots[(str(node), slot - 1)])
    model.Add(sum(slots.values()) == deployable)
    model.Maximize(
        sum(weights[str(node)] * marginal[slot] * slots[(str(node), slot)] for node in candidates for slot in range(max_per_intersection))
    )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("CP-SAT could not produce a manpower plan.")
    allocation = {
        str(node): sum(solver.Value(slots[(str(node), slot)]) for slot in range(max_per_intersection))
        for node in candidates
    }
    deployment = _deployment(graph, candidates, weights, allocation)
    payload = {
        "available_officers": officers,
        "assigned_officers": sum(allocation.values()),
        "reserve_officers": reserve,
        "coverage_score": int(solver.ObjectiveValue()),
        "optimizer": "ortools_cp_sat",
        "staffing_model": staffing_model,
        "deployment": deployment,
    }
    write_json(output_path, payload)
    LOGGER.info("Saved manpower plan with %s officers to %s", len(deployment), output_path)
    return payload


def _deployment(
    graph: nx.MultiDiGraph,
    candidates: list[Any],
    weights: dict[str, int],
    allocation: dict[str, int],
) -> list[dict[str, Any]]:
    deployment = []
    for node in candidates:
        node_id = str(node)
        count = allocation[node_id]
        if count <= 0:
            continue
        deployment.append(
            {
                "node": node_id,
                "latitude": float(graph.nodes[node]["y"]),
                "longitude": float(graph.nodes[node]["x"]),
                "deployment_score": weights[node_id],
                "recommended_officers": count,
                "role": "traffic_control" if count <= 2 else "traffic_control_and_diversion",
            }
        )
    return sorted(deployment, key=lambda item: item["deployment_score"], reverse=True)


def _staffing_model(
    event: dict[str, Any],
    predicted_duration_min: float,
    road_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate how much of the available force should leave reserve.

    The prediction stage already changes affected-road radius and duration for peak
    periods. This demand cap applies the same operational idea to staffing so a
    low-priority night event does not consume every available officer.
    """
    priority = str(event.get("priority", "medium")).lower()
    cause = str(event.get("event_cause", "others")).lower()
    requires_closure = bool(event.get("requires_road_closure", False))
    hour = _event_hour(event.get("start_datetime"))

    priority_score = {
        "low": 0.16,
        "medium": 0.38,
        "high": 0.68,
        "critical": 0.92,
    }.get(priority, 0.38)
    cause_score = {
        "accident": 0.18,
        "tree_fall": 0.14,
        "water_logging": 0.14,
        "public_event": 0.12,
        "vehicle_breakdown": 0.08,
        "pot_holes": 0.04,
        "others": 0.05,
    }.get(cause, 0.05)
    closure_score = 0.18 if requires_closure else 0.0
    duration_score = _clamp(predicted_duration_min / 180.0, 0.0, 1.0)
    road_context = road_context or {}
    road_response_factor = float(road_context.get("response_factor", 1.0))

    if hour in {8, 9, 10, 17, 18, 19, 20}:
        time_factor = 1.25
        time_band = "peak"
    elif hour <= 5 or hour >= 23:
        time_factor = 0.45
        time_band = "night_off_peak"
    elif hour in {6, 7, 11, 15, 16, 21, 22}:
        time_factor = 0.78
        time_band = "shoulder"
    else:
        time_factor = 1.0
        time_band = "daytime"

    severity_score = _clamp(priority_score + cause_score + closure_score, 0.0, 1.0)
    raw_demand = (0.7 * severity_score + 0.3 * duration_score) * time_factor * road_response_factor
    if time_band == "night_off_peak" and priority == "low" and not requires_closure:
        raw_demand = min(raw_demand, 0.12)
    if road_context.get("classification") == "contained_terminal_local_road" and not requires_closure:
        raw_demand = min(raw_demand, 0.35)
    if priority == "critical" or requires_closure:
        raw_demand = max(raw_demand, 0.35)

    demand_factor = round(_clamp(raw_demand, 0.0, 1.0), 3)
    return {
        "demand_factor": demand_factor,
        "time_band": time_band,
        "time_factor": time_factor,
        "hour": hour,
        "priority_score": round(priority_score, 3),
        "cause_score": round(cause_score, 3),
        "closure_score": round(closure_score, 3),
        "duration_score": round(duration_score, 3),
        "road_context_factor": round(road_response_factor, 3),
        "road_context": road_context.get("classification", "unknown"),
        "policy": "peak/off-peak demand cap with reserve retention",
    }


def _event_hour(value: Any) -> int:
    if not value:
        return datetime.now().hour
    try:
        text = str(value).replace("Z", "+00:00")
        return datetime.fromisoformat(text).hour
    except ValueError:
        return datetime.now().hour


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize traffic police deployment.")
    parser.add_argument("--prediction", type=Path, default=PREDICTIONS_DIR / "latest_prediction.json")
    parser.add_argument("--graph", type=Path, default=NETWORK_DIR / "bangalore_graph.graphml")
    parser.add_argument("--output", type=Path, default=PREDICTIONS_DIR / "manpower_plan.json")
    parser.add_argument("--officers", type=int, default=12)
    args = parser.parse_args()
    ensure_directories()
    optimize(args.prediction, args.graph, args.output, args.officers)


if __name__ == "__main__":
    main()
