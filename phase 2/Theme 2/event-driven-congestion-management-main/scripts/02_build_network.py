from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from lib.logging_utils import get_logger
from lib.network_utils import build_fallback_graph
from lib.paths import NETWORK_DIR, ensure_directories

LOGGER = get_logger("build_network")


def build_network(output_path: Path, place: str) -> None:
    try:
        import osmnx as ox

        LOGGER.info("Downloading drive network for %s", place)
        graph = ox.graph_from_place(place, network_type="drive", simplify=True)
        graph = ox.add_edge_speeds(graph)
        graph = ox.add_edge_travel_times(graph)
        ox.save_graphml(graph, output_path)
        LOGGER.info("Saved OSM graph with %s nodes and %s edges to %s", len(graph.nodes), len(graph.edges), output_path)
    except Exception as exc:
        if output_path.exists() and output_path.stat().st_size > 1_000_000:
            LOGGER.warning("OSMnx download failed (%s). Preserving existing road network at %s.", exc, output_path)
            return
        LOGGER.warning("OSMnx download failed (%s). Saving deterministic fallback graph.", exc)
        graph = build_fallback_graph()
        import networkx as nx

        output_path.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(graph, output_path)
        LOGGER.info("Saved fallback graph to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or create Bengaluru drive graph.")
    parser.add_argument("--place", default="Bengaluru, Karnataka, India")
    parser.add_argument("--output", type=Path, default=NETWORK_DIR / "bangalore_graph.graphml")
    args = parser.parse_args()
    ensure_directories()
    build_network(args.output, args.place)


if __name__ == "__main__":
    main()
