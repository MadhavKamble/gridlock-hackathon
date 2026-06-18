from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import networkx as nx

sys.path.append(str(Path(__file__).resolve().parents[1]))
from lib.bernoulli_pressure import compute_edge_pressure, detect_high_tension_nodes, pressure_features
from lib.logging_utils import get_logger
from lib.network_utils import haversine_m, load_graph, read_json, write_json
from lib.paths import NETWORK_DIR, PREDICTIONS_DIR, ensure_directories
from lib.sumo_simulation import run_sumo_if_available

LOGGER = get_logger("diversion_routes")


def generate_routes(prediction_path: Path, barricade_path: Path, graph_path: Path, output_path: Path) -> dict[str, Any]:
    prediction = read_json(prediction_path)
    barricades = read_json(barricade_path)
    pressure_graph = compute_edge_pressure(load_graph(graph_path), prediction)
    graph = pressure_graph.copy()
    closed = barricades.get("best_plan", {}).get("closed_edges", [])
    closed_set = {(edge["u"], edge["v"], edge["key"]) for edge in closed}
    for u, v, k in list(graph.edges(keys=True)):
        if (str(u), str(v), str(k)) in closed_set:
            graph.remove_edge(u, v, k)
    routes = []
    # graph and pressure_graph share the same nodes (only edges are removed), so a single
    # str->node lookup serves every stage below.
    node_lookup = {str(node): node for node in graph.nodes}
    for index, closed_edge in enumerate(closed[:6], start=1):
        source = node_lookup.get(str(closed_edge["u"]))
        target = node_lookup.get(str(closed_edge["v"]))
        if source is None or target is None:
            continue
        try:
            node_path = nx.shortest_path(graph, source, target, weight="travel_time")
            routes.append(
                {
                    "route_id": f"D{index}",
                    "route_type": "manual_closure_bypass",
                    "source": str(source),
                    "target": str(target),
                    "bypasses": closed_edge.get("name", "closed road"),
                    "nodes": [str(node) for node in node_path],
                    "geometry": [[float(graph.nodes[node]["y"]), float(graph.nodes[node]["x"])] for node in node_path],
                    "estimated_travel_time_s": round(_path_weight(graph, node_path), 2),
                    "distance_m": round(_path_weight(graph, node_path, "length"), 2),
                    "bernoulli_potential": round(_path_weight(graph, node_path, "E"), 4),
                }
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
    relevant_edges = _local_pressure_edges(pressure_graph, prediction, closed, node_lookup)
    field = pressure_features(pressure_graph, allowed_edges=relevant_edges, max_edges=350)
    pressure_threshold = _adaptive_pressure_threshold(field)
    tension_graph = pressure_graph.edge_subgraph(
        [(u, v, k) for u, v, k in pressure_graph.edges(keys=True) if (str(u), str(v), str(k)) in relevant_edges]
    ).copy()
    if _should_skip_auto_diversions(prediction, closed):
        tension_nodes = []
        auto_routes = []
    else:
        tension_nodes = detect_high_tension_nodes(tension_graph, pressure_threshold=pressure_threshold)
        if not tension_nodes:
            tension_nodes = _fallback_tension_nodes(pressure_graph, field, limit=20, node_lookup=node_lookup)
        auto_routes = _auto_diversions(graph, tension_nodes, prediction, limit=8)
    payload = {
        "closed_edges_removed": len(closed),
        "routes": routes[:12],
        "auto_diversion_routes": auto_routes,
        "all_routes": routes[:12] + auto_routes,
        "pressure_field": field,
        "high_tension_nodes": [str(node) for node in tension_nodes[:200]],
        "bernoulli_parameters": {"k": 0.02, "alpha": 0.7, "beta": 0.3, "pressure_threshold": pressure_threshold},
        "sumo_simulation": run_sumo_if_available(routes[:12] + auto_routes),
    }
    write_json(output_path, payload)
    LOGGER.info(
        "Saved %s manual and %s Bernoulli auto-diversion routes to %s",
        len(payload["routes"]),
        len(payload["auto_diversion_routes"]),
        output_path,
    )
    return payload


def _should_skip_auto_diversions(prediction: dict[str, Any], closed: list[dict[str, Any]]) -> bool:
    event = prediction.get("event", {})
    road_context = prediction.get("road_context", {})
    if event.get("requires_road_closure") or closed:
        return False
    return str(road_context.get("classification", "")).lower() in {
        "contained_terminal_local_road",
        "local_access_road",
    }


def _local_pressure_edges(
    graph: nx.MultiDiGraph,
    prediction: dict[str, Any],
    closed: list[dict[str, Any]],
    node_lookup: dict[str, Any],
) -> set[tuple[str, str, str]]:
    affected_keys = {(edge["u"], edge["v"], edge.get("key", "0")) for edge in prediction.get("affected_edges", [])}
    closed_keys = {(edge["u"], edge["v"], edge.get("key", "0")) for edge in closed}
    local_nodes = set()
    for u, v, _ in affected_keys | closed_keys:
        if u in node_lookup:
            local_nodes.add(node_lookup[u])
        if v in node_lookup:
            local_nodes.add(node_lookup[v])
    for node in list(local_nodes):
        local_nodes.update(graph.predecessors(node))
        local_nodes.update(graph.successors(node))
    local_edges = set()
    for u, v, key in graph.edges(keys=True):
        if u in local_nodes or v in local_nodes:
            local_edges.add((str(u), str(v), str(key)))
    return local_edges


def _adaptive_pressure_threshold(field: list[dict[str, Any]], default_threshold: float = 0.6) -> float:
    pressures = sorted(float(edge.get("pressure", 0.0)) for edge in field)
    if not pressures:
        return default_threshold
    if pressures[-1] >= default_threshold:
        return default_threshold
    baseline = pressures[0]
    elevated = [value for value in pressures if value > baseline + 0.01]
    if not elevated:
        return round(max(0.05, pressures[-1]), 4)
    return round(max(0.05, (min(elevated) + max(elevated)) / 2), 4)


def _fallback_tension_nodes(
    graph: nx.MultiDiGraph,
    field: list[dict[str, Any]],
    limit: int,
    node_lookup: dict[str, Any],
) -> list[Any]:
    nodes: list[Any] = []
    for edge in sorted(field, key=lambda item: float(item.get("pressure", 0.0)), reverse=True):
        for endpoint in (edge.get("u"), edge.get("v")):
            node = node_lookup.get(str(endpoint))
            if node is not None and node not in nodes:
                nodes.append(node)
        if len(nodes) >= limit:
            break
    return nodes


def _auto_diversions(
    graph: nx.MultiDiGraph,
    tension_nodes: list[Any],
    prediction: dict[str, Any],
    limit: int = 8,
) -> list[dict[str, Any]]:
    event = prediction.get("event", {})
    exits = _major_exit_nodes(graph, event, max_nodes=40)
    routes: list[dict[str, Any]] = []
    seen_paths: set[tuple[str, ...]] = set()
    for source in tension_nodes[:40]:
        best_path: list[Any] | None = None
        best_potential = float("inf")
        for target in exits:
            if source == target:
                continue
            try:
                candidate = nx.dijkstra_path(graph, source, target, weight="E")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if _path_weight(graph, candidate, "length") > 4_000:
                continue
            potential = _path_weight(graph, candidate, "E")
            if potential < best_potential:
                best_path = candidate
                best_potential = potential
        if not best_path or len(best_path) < 2:
            continue
        path_key = tuple(str(node) for node in best_path)
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        routes.append(
            {
                "route_id": f"B{len(routes) + 1}",
                "route_type": "bernoulli_auto_diversion",
                "source": str(best_path[0]),
                "target": str(best_path[-1]),
                "nodes": [str(node) for node in best_path],
                "geometry": [[float(graph.nodes[node]["y"]), float(graph.nodes[node]["x"])] for node in best_path],
                "estimated_travel_time_s": round(_path_weight(graph, best_path), 2),
                "distance_m": round(_path_weight(graph, best_path, "length"), 2),
                "bernoulli_potential": round(best_potential, 4),
                "mean_pressure": round(_path_mean(graph, best_path, "P"), 4),
            }
        )
        if len(routes) >= limit:
            break
    return routes


def _major_exit_nodes(graph: nx.MultiDiGraph, event: dict[str, Any], max_nodes: int = 40) -> list[Any]:
    event_lat = float(event.get("latitude", 12.9716))
    event_lon = float(event.get("longitude", 77.5946))
    candidates = []
    for node, data in graph.nodes(data=True):
        degree = graph.degree(node)
        if degree < 3:
            continue
        distance_m = haversine_m(event_lat, event_lon, float(data.get("y", event_lat)), float(data.get("x", event_lon)))
        if 600 <= distance_m <= 4_000:
            candidates.append((degree, distance_m, node))
    candidates.sort(key=lambda row: (-row[0], row[1]))
    return [node for _, _, node in candidates[:max_nodes]]


def _path_weight(graph: nx.MultiDiGraph, nodes: list[Any], field: str = "travel_time") -> float:
    total = 0.0
    for u, v in zip(nodes[:-1], nodes[1:]):
        edges = graph.get_edge_data(u, v, default={})
        if edges:
            total += min(
                float(data.get(field, data.get("length", 0) / 8.33 if field == "travel_time" else 0))
                for data in edges.values()
            )
    return total


def _path_mean(graph: nx.MultiDiGraph, nodes: list[Any], field: str) -> float:
    values = []
    for u, v in zip(nodes[:-1], nodes[1:]):
        edges = graph.get_edge_data(u, v, default={})
        if edges:
            values.append(min(float(data.get(field, 0.0)) for data in edges.values()))
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate alternative routes after closures.")
    parser.add_argument("--prediction", type=Path, default=PREDICTIONS_DIR / "latest_prediction.json")
    parser.add_argument("--barricades", type=Path, default=PREDICTIONS_DIR / "barricade_plan.json")
    parser.add_argument("--graph", type=Path, default=NETWORK_DIR / "bangalore_graph.graphml")
    parser.add_argument("--output", type=Path, default=PREDICTIONS_DIR / "diversion_routes.json")
    args = parser.parse_args()
    ensure_directories()
    generate_routes(args.prediction, args.barricades, args.graph, args.output)


if __name__ == "__main__":
    main()
