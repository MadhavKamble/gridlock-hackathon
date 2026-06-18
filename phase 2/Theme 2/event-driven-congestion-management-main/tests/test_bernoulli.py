from __future__ import annotations

import networkx as nx
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lib.bernoulli_pressure import compute_edge_pressure


def test_pressure_increases_as_speed_decreases() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node("a", x=77.0, y=12.0)
    graph.add_node("b", x=77.1, y=12.1)
    graph.add_node("c", x=77.2, y=12.2)
    graph.add_edge("a", "b", key=0, maxspeed=60, lanes=1, length=100, travel_time=6)
    graph.add_edge("b", "c", key=0, maxspeed=60, lanes=1, length=100, travel_time=6)
    predictions = {
        "affected_edges": [
            {"u": "a", "v": "b", "key": "0", "predicted_speed_kph": 12},
            {"u": "b", "v": "c", "key": "0", "predicted_speed_kph": 45},
        ]
    }

    compute_edge_pressure(graph, predictions, k=0.001)

    slow_pressure = graph["a"]["b"][0]["P"]
    fast_pressure = graph["b"]["c"][0]["P"]
    assert slow_pressure > fast_pressure


def test_minimum_potential_path_prefers_low_pressure_even_if_slightly_longer() -> None:
    graph = nx.MultiDiGraph()
    for node in ("s", "a", "b", "t"):
        graph.add_node(node, x=77.0, y=12.0)
    graph.add_edge("s", "a", key=0, maxspeed=60, lanes=1, length=100, travel_time=6)
    graph.add_edge("a", "t", key=0, maxspeed=60, lanes=1, length=100, travel_time=6)
    graph.add_edge("s", "b", key=0, maxspeed=50, lanes=1, length=140, travel_time=10)
    graph.add_edge("b", "t", key=0, maxspeed=50, lanes=1, length=140, travel_time=10)
    predictions = {
        "affected_edges": [
            {"u": "s", "v": "a", "key": "0", "predicted_speed_kph": 10},
            {"u": "a", "v": "t", "key": "0", "predicted_speed_kph": 10},
            {"u": "s", "v": "b", "key": "0", "predicted_speed_kph": 50},
            {"u": "b", "v": "t", "key": "0", "predicted_speed_kph": 50},
        ]
    }

    compute_edge_pressure(graph, predictions, k=0.00001, alpha=0.9, beta=0.1)
    path = nx.dijkstra_path(graph, "s", "t", weight="E")

    assert path == ["s", "b", "t"]


if __name__ == "__main__":
    test_pressure_increases_as_speed_decreases()
    test_minimum_potential_path_prefers_low_pressure_even_if_slightly_longer()
    print("Bernoulli pressure tests passed.")
