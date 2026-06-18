from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import networkx as nx

sys.path.append(str(Path(__file__).resolve().parents[1]))
from lib.logging_utils import get_logger
from lib.network_utils import load_graph, read_json, write_json
from lib.paths import NETWORK_DIR, PREDICTIONS_DIR, ensure_directories

LOGGER = get_logger("barricade_simulator")


def _edge_lookup(graph: nx.MultiDiGraph) -> dict[tuple[str, str, str], tuple[Any, Any, Any, dict[str, Any]]]:
    return {(str(u), str(v), str(k)): (u, v, k, data) for u, v, k, data in graph.edges(keys=True, data=True)}


def simulate(prediction_path: Path, graph_path: Path, output_path: Path) -> dict[str, Any]:
    prediction = read_json(prediction_path)
    graph = load_graph(graph_path)
    affected = prediction.get("affected_edges", [])[:12]
    if not affected:
        raise ValueError("No affected edges available for barricade simulation.")
    cityflow_used = False
    try:
        import cityflow  # noqa: F401

        cityflow_used = True
    except Exception:
        cityflow_used = False
    lookup = _edge_lookup(graph)
    affected = sorted(affected, key=lambda edge: edge.get("distance_from_event_m", 0))
    edge_keys = [(e["u"], e["v"], e["key"]) for e in affected if (e["u"], e["v"], e["key"]) in lookup]
    event = prediction.get("event", {})
    road_context = prediction.get("road_context", {})
    if _should_use_incident_protection(event, road_context):
        protected_edges = [
            {
                "u": edge["u"],
                "v": edge["v"],
                "key": edge["key"],
                "name": edge.get("name", "Unnamed road"),
                "geometry": edge.get("geometry", []),
            }
            for edge in affected[: min(2, len(affected))]
        ]
        best = {
            "plan_name": "incident_protection",
            "cityflow_used": cityflow_used,
            "closed_edges": [],
            "protected_edges": protected_edges,
            "estimated_travel_time_s": 0.0,
            "congestion_score": 0.0,
            "throughput_score": 1.0,
            "safety_protection": 0.3,
            "residual_safety_risk": 0.35,
            "objective_score": 0.1575,
            "explanation": "No road closure is recommended because the event is contained on a local terminal/access road.",
        }
        payload = {
            "simulation_engine": "CityFlow" if cityflow_used else "python_graph_fallback",
            "plans": [best],
            "best_plan": best,
        }
        write_json(output_path, payload)
        LOGGER.info("Saved incident-protection plan to %s", output_path)
        return payload

    severity = {"low": 0.25, "medium": 0.45, "high": 0.7, "critical": 0.95}.get(
        str(event.get("priority", "medium")).lower(), 0.45
    )
    if event.get("requires_road_closure"):
        severity = min(1.0, severity + 0.25)
    if event.get("event_cause") in {"accident", "tree_fall", "water_logging"}:
        severity = min(1.0, severity + 0.1)
    plans = [
        ("minimal", edge_keys[: min(2, len(edge_keys))], 0.35),
        ("balanced", edge_keys[: min(5, len(edge_keys))], 0.7),
        ("full_closure", edge_keys[: min(9, len(edge_keys))], 1.0),
    ]
    results = []
    node_lookup = {str(node): node for node in graph.nodes}
    local_nodes = {
        node_lookup[node_id]
        for edge in affected
        for node_id in (edge["u"], edge["v"])
        if node_id in node_lookup
    }
    expanded_nodes = set(local_nodes)
    for node in list(local_nodes):
        expanded_nodes.update(graph.predecessors(node))
        expanded_nodes.update(graph.successors(node))
    simulation_graph = graph.subgraph(expanded_nodes).copy()
    base_travel_time = sum(
        float(data.get("travel_time", data.get("length", 0) / 8.33))
        for _, _, _, data in simulation_graph.edges(keys=True, data=True)
    )
    for name, closed, protection in plans:
        candidate = simulation_graph.copy()
        for edge in closed:
            u, v, k, _ = lookup[edge]
            if candidate.has_edge(u, v, k):
                candidate.remove_edge(u, v, k)
        reachable_ratio = _largest_component_ratio(candidate)
        closure_ratio = len(closed) / max(1, len(edge_keys))
        congestion = round(closure_ratio * (2 - reachable_ratio), 4)
        throughput = round(max(0.0, reachable_ratio * (1.0 - congestion / 2.5)), 4)
        travel_time = round(base_travel_time * (1 + congestion) / max(reachable_ratio, 0.1), 2)
        residual_safety_risk = round(severity * (1.0 - protection), 4)
        objective_score = round(congestion * 0.55 + residual_safety_risk * 0.45, 4)
        results.append(
            {
                "plan_name": name,
                "cityflow_used": cityflow_used,
                "closed_edges": [
                    {
                        "u": u,
                        "v": v,
                        "key": k,
                        "name": str(lookup[(u, v, k)][3].get("name", "Unnamed road")),
                        "geometry": affected[edge_keys.index((u, v, k))].get("geometry", []),
                    }
                    for u, v, k in closed
                ],
                "estimated_travel_time_s": travel_time,
                "congestion_score": congestion,
                "throughput_score": throughput,
                "safety_protection": protection,
                "residual_safety_risk": residual_safety_risk,
                "objective_score": objective_score,
            }
        )
    best = min(results, key=lambda row: row["objective_score"])
    payload = {"simulation_engine": "CityFlow" if cityflow_used else "python_graph_fallback", "plans": results, "best_plan": best}
    write_json(output_path, payload)
    LOGGER.info("Saved barricade simulation to %s", output_path)
    return payload


def _largest_component_ratio(graph: nx.MultiDiGraph) -> float:
    simple = nx.Graph(graph)
    if simple.number_of_nodes() == 0:
        return 0.0
    components = list(nx.connected_components(simple))
    if not components:
        return 0.0
    return len(max(components, key=len)) / simple.number_of_nodes()


def _should_use_incident_protection(event: dict[str, Any], road_context: dict[str, Any]) -> bool:
    if event.get("requires_road_closure"):
        return False
    return str(road_context.get("classification", "")).lower() in {
        "contained_terminal_local_road",
        "local_access_road",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate barricade options.")
    parser.add_argument("--prediction", type=Path, default=PREDICTIONS_DIR / "latest_prediction.json")
    parser.add_argument("--graph", type=Path, default=NETWORK_DIR / "bangalore_graph.graphml")
    parser.add_argument("--output", type=Path, default=PREDICTIONS_DIR / "barricade_plan.json")
    args = parser.parse_args()
    ensure_directories()
    simulate(args.prediction, args.graph, args.output)


if __name__ == "__main__":
    main()
