from __future__ import annotations

import math
from typing import Any

import networkx as nx


def compute_edge_pressure(
    G: nx.MultiDiGraph,
    predictions: dict[str, Any],
    k: float = 0.02,
    alpha: float = 0.7,
    beta: float = 0.3,
) -> nx.MultiDiGraph:
    """Annotate graph edges with Bernoulli-style traffic pressure and potential.

    The heuristic treats slow, high-delay links as high-pressure traffic segments.
    Edges receive:
    - `predicted_speed_kph`
    - `free_flow_speed_kph`
    - `traffic_pressure`
    - `bernoulli_potential`

    The graph is mutated and returned for convenient chaining.
    """
    affected = {
        (str(edge.get("u")), str(edge.get("v")), str(edge.get("key", "0"))): edge
        for edge in predictions.get("affected_edges", [])
    }
    capacity_norm = _capacity_norm(G)
    event_duration = float(predictions.get("predicted_duration_min", 0) or 0)

    for u, v, key, data in G.edges(keys=True, data=True):
        free_flow_speed = max(_speed_kph(data), 1.0)
        lanes = max(_lanes(data), 1.0)
        affected_edge = affected.get((str(u), str(v), str(key)))
        predicted_speed = _predicted_speed(affected_edge, free_flow_speed, event_duration)
        density = (free_flow_speed * lanes) / predicted_speed
        delay_penalty = max(0.0, min(1.0, 1.0 - predicted_speed / free_flow_speed))
        pressure = alpha * min(density / capacity_norm, 1.0) + beta * delay_penalty
        speed_loss = max(0.0, free_flow_speed - predicted_speed)
        potential = 0.5 * k * speed_loss**2 + pressure
        data["predicted_speed_kph"] = round(predicted_speed, 3)
        data["free_flow_speed_kph"] = round(free_flow_speed, 3)
        data["traffic_pressure"] = round(pressure, 6)
        data["bernoulli_potential"] = round(potential, 6)
        data["E"] = data["bernoulli_potential"]
        data["P"] = data["traffic_pressure"]
    return G


def detect_high_tension_nodes(G: nx.MultiDiGraph, pressure_threshold: float = 0.6) -> list[Any]:
    """Return nodes sitting on a pressure gradient.

    A node is high-tension when at least one incident edge is above the threshold
    and another neighbouring edge is below the threshold. Those are useful places
    to start automatic diversions because they mark a transition between stressed
    and available capacity.
    """
    nodes: list[Any] = []
    for node in G.nodes:
        pressures = [float(data.get("traffic_pressure", data.get("P", 0.0))) for _, _, data in G.edges(node, data=True)]
        if not pressures and G.is_directed():
            pressures = [float(data.get("traffic_pressure", data.get("P", 0.0))) for _, _, data in G.in_edges(node, data=True)]
        if any(value > pressure_threshold for value in pressures) and any(value < pressure_threshold for value in pressures):
            nodes.append(node)
    return nodes


def pressure_features(
    G: nx.MultiDiGraph,
    max_edges: int = 600,
    allowed_edges: set[tuple[str, str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Return map-ready pressure features for the most relevant edges."""
    edge_iter = [
        item
        for item in G.edges(keys=True, data=True)
        if allowed_edges is None or (str(item[0]), str(item[1]), str(item[2])) in allowed_edges
    ]
    edges = sorted(
        edge_iter,
        key=lambda item: float(item[3].get("traffic_pressure", item[3].get("P", 0.0))),
        reverse=True,
    )
    features = []
    for u, v, key, data in edges[:max_edges]:
        geometry = _edge_geometry(G, u, v, data)
        if not geometry:
            continue
        features.append(
            {
                "u": str(u),
                "v": str(v),
                "key": str(key),
                "name": str(data.get("name", "Unnamed road")),
                "pressure": float(data.get("traffic_pressure", data.get("P", 0.0))),
                "potential": float(data.get("bernoulli_potential", data.get("E", 0.0))),
                "predicted_speed_kph": float(data.get("predicted_speed_kph", 0.0)),
                "free_flow_speed_kph": float(data.get("free_flow_speed_kph", 0.0)),
                "geometry": geometry,
            }
        )
    return features


def _predicted_speed(affected_edge: dict[str, Any] | None, free_flow_speed: float, duration_min: float) -> float:
    if affected_edge:
        for field in ("predicted_speed", "predicted_speed_kph", "speed_kph"):
            if affected_edge.get(field) is not None:
                return max(float(affected_edge[field]), 1.0)
        impact_level = str(affected_edge.get("impact_level", "medium")).lower()
        base_factor = 0.28 if impact_level == "high" else 0.55
        duration_factor = max(0.35, 1.0 - min(duration_min, 240.0) / 480.0)
        return max(free_flow_speed * min(base_factor, duration_factor), 1.0)
    return max(free_flow_speed, 1.0)


def _capacity_norm(G: nx.MultiDiGraph) -> float:
    values = []
    for _, _, _, data in G.edges(keys=True, data=True):
        values.append(max(_speed_kph(data), 1.0) * max(_lanes(data), 1.0))
    if not values:
        return 1.0
    values.sort()
    index = min(len(values) - 1, max(0, int(len(values) * 0.85)))
    return max(values[index], 1.0)


def _speed_kph(data: dict[str, Any]) -> float:
    for field in ("maxspeed", "speed_kph"):
        value = data.get(field)
        if value is None:
            continue
        parsed = _first_number(value)
        if parsed:
            return parsed
    length = float(data.get("length", 0.0) or 0.0)
    travel_time = float(data.get("travel_time", 0.0) or 0.0)
    if length > 0 and travel_time > 0:
        return max((length / travel_time) * 3.6, 1.0)
    return 35.0


def _lanes(data: dict[str, Any]) -> float:
    parsed = _first_number(data.get("lanes"))
    return parsed if parsed and parsed > 0 else 1.0


def _first_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, list):
        for item in value:
            parsed = _first_number(item)
            if parsed:
                return parsed
        return None
    text = str(value)
    number = ""
    for char in text:
        if char.isdigit() or char == ".":
            number += char
        elif number:
            break
    if not number:
        return None
    try:
        parsed = float(number)
    except ValueError:
        return None
    if "mph" in text.lower():
        parsed *= 1.60934
    return parsed if math.isfinite(parsed) else None


def _edge_geometry(G: nx.MultiDiGraph, u: Any, v: Any, data: dict[str, Any]) -> list[list[float]]:
    geometry = data.get("geometry")
    if geometry is not None:
        return [[float(lat), float(lon)] for lon, lat in geometry.coords]
    if u not in G.nodes or v not in G.nodes:
        return []
    return [[float(G.nodes[u]["y"]), float(G.nodes[u]["x"])], [float(G.nodes[v]["y"]), float(G.nodes[v]["x"])]]
