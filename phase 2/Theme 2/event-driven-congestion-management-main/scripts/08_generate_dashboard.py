from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import folium
except Exception:
    folium = None

sys.path.append(str(Path(__file__).resolve().parents[1]))
from lib.logging_utils import get_logger
from lib.map_utils import build_response_map
from lib.network_utils import read_json, write_json
from lib.paths import DASHBOARDS_DIR, PREDICTIONS_DIR, ensure_directories

LOGGER = get_logger("generate_dashboard")


def build_geojson(prediction: dict[str, Any], manpower: dict[str, Any], barricades: dict[str, Any], diversions: dict[str, Any]) -> dict[str, Any]:
    features = []
    for edge in prediction.get("affected_edges", []):
        coords = [[lon, lat] for lat, lon in edge.get("geometry", [])]
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {"layer": "affected_road", "name": edge.get("name"), "length_m": edge.get("length_m")},
            }
        )
    for item in manpower.get("deployment", []):
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [item["longitude"], item["latitude"]]},
                "properties": {"layer": "police_deployment", "officers": item["recommended_officers"], "node": item["node"]},
            }
        )
    for route in diversions.get("routes", []):
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[lon, lat] for lat, lon in route.get("geometry", [])]},
                "properties": {
                    "layer": "diversion_route",
                    "route_type": route.get("route_type", "manual_closure_bypass"),
                    "source": route["source"],
                    "target": route["target"],
                },
            }
        )
    for route in diversions.get("auto_diversion_routes", []):
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[lon, lat] for lat, lon in route.get("geometry", [])]},
                "properties": {
                    "layer": "bernoulli_auto_diversion",
                    "route_id": route.get("route_id"),
                    "mean_pressure": route.get("mean_pressure"),
                    "bernoulli_potential": route.get("bernoulli_potential"),
                },
            }
        )
    for edge in diversions.get("pressure_field", []):
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[lon, lat] for lat, lon in edge.get("geometry", [])]},
                "properties": {
                    "layer": "bernoulli_pressure",
                    "pressure": edge.get("pressure"),
                    "potential": edge.get("potential"),
                    "predicted_speed_kph": edge.get("predicted_speed_kph"),
                },
            }
        )
    for edge in barricades.get("best_plan", {}).get("closed_edges", []):
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[lon, lat] for lat, lon in edge.get("geometry", [])],
                },
                "properties": {
                    "layer": "barricade",
                    "plan_name": barricades.get("best_plan", {}).get("plan_name"),
                    "name": edge.get("name"),
                    "u": edge.get("u"),
                    "v": edge.get("v"),
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def generate(prediction_path: Path, manpower_path: Path, barricade_path: Path, diversion_path: Path, geojson_path: Path, html_path: Path) -> None:
    prediction = read_json(prediction_path)
    manpower = read_json(manpower_path)
    barricades = read_json(barricade_path)
    diversions = read_json(diversion_path)
    geojson = build_geojson(prediction, manpower, barricades, diversions)
    write_json(geojson_path, geojson)
    event = prediction["event"]
    if folium is None:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(_fallback_html(prediction, manpower, barricades, diversions, geojson), encoding="utf-8")
        LOGGER.warning("folium is not installed; saved fallback HTML dashboard to %s", html_path)
        return
    fmap = build_response_map(prediction, manpower, barricades, diversions)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(html_path)
    LOGGER.info("Saved GeoJSON to %s and dashboard to %s", geojson_path, html_path)


def _fallback_html(
    prediction: dict[str, Any],
    manpower: dict[str, Any],
    barricades: dict[str, Any],
    diversions: dict[str, Any],
    geojson: dict[str, Any],
) -> str:
    event = prediction.get("event", {})
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Bengaluru Congestion Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #17202a; }}
    main {{ max-width: 1100px; margin: 0 auto; }}
    section {{ border: 1px solid #d5d8dc; padding: 16px; margin: 16px 0; }}
    code, pre {{ background: #f4f6f7; padding: 8px; display: block; overflow: auto; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .metric {{ background: #f8f9f9; padding: 12px; border: 1px solid #e5e8e8; }}
  </style>
</head>
<body>
<main>
  <h1>Bengaluru Congestion Response</h1>
  <div class="grid">
    <div class="metric"><strong>Predicted duration</strong><br>{prediction.get("predicted_duration_min", 0)} min</div>
    <div class="metric"><strong>Affected roads</strong><br>{len(prediction.get("affected_edges", []))}</div>
    <div class="metric"><strong>Assigned officers</strong><br>{manpower.get("assigned_officers", 0)}</div>
    <div class="metric"><strong>Diversions</strong><br>{len(diversions.get("routes", []))}</div>
  </div>
  <section>
    <h2>Event</h2>
    <p>{event.get("event_cause", "event")} on {event.get("corridor", "unknown corridor")} at {event.get("latitude")}, {event.get("longitude")}.</p>
  </section>
  <section>
    <h2>Best Barricade Plan</h2>
    <pre>{json.dumps(barricades.get("best_plan", {}), indent=2)}</pre>
  </section>
  <section>
    <h2>GeoJSON</h2>
    <pre>{json.dumps(geojson, indent=2)}</pre>
  </section>
</main>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Folium/GeoJSON dashboard artifacts.")
    parser.add_argument("--prediction", type=Path, default=PREDICTIONS_DIR / "latest_prediction.json")
    parser.add_argument("--manpower", type=Path, default=PREDICTIONS_DIR / "manpower_plan.json")
    parser.add_argument("--barricades", type=Path, default=PREDICTIONS_DIR / "barricade_plan.json")
    parser.add_argument("--diversions", type=Path, default=PREDICTIONS_DIR / "diversion_routes.json")
    parser.add_argument("--geojson", type=Path, default=DASHBOARDS_DIR / "dashboard.geojson")
    parser.add_argument("--html", type=Path, default=DASHBOARDS_DIR / "dashboard.html")
    args = parser.parse_args()
    ensure_directories()
    generate(args.prediction, args.manpower, args.barricades, args.diversions, args.geojson, args.html)


if __name__ == "__main__":
    main()
