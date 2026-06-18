from __future__ import annotations

from typing import Any

import folium


ROUTE_COLORS = ["#14866d", "#2474b5", "#6b5fb5", "#00838f", "#5d8c2d", "#8c5a2d"]
PLAN_COLORS = {"minimal": "#6b7280", "balanced": "#7c3aed", "full_closure": "#111827"}


def build_response_map(
    prediction: dict[str, Any],
    manpower: dict[str, Any],
    barricades: dict[str, Any],
    diversions: dict[str, Any],
    zoom_start: int = 14,
    show_pressure_heatmap: bool = True,
    show_manual_diversions: bool = True,
    show_bernoulli_routes: bool = True,
) -> folium.Map:
    event = prediction.get("event", {"latitude": 12.9716, "longitude": 77.5946})
    center = [float(event["latitude"]), float(event["longitude"])]
    fmap = folium.Map(location=center, zoom_start=zoom_start, tiles="CartoDB positron", control_scale=True)

    event_popup = (
        f"<b>{str(event.get('event_cause', 'event')).replace('_', ' ').title()}</b><br>"
        f"Priority: {event.get('priority', '-')}<br>"
        f"Corridor: {event.get('corridor', '-')}<br>"
        f"Impact estimate: {prediction.get('predicted_duration_min', '-')} min<br>"
        f"Range: {prediction.get('predicted_duration_range_min', '-')}–"
        f"{prediction.get('predicted_duration_range_max', '-')} min"
    )
    folium.Marker(
        center,
        popup=folium.Popup(event_popup, max_width=320),
        tooltip="Event location",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(fmap)

    affected_group = folium.FeatureGroup(name="Affected roads", show=True)
    bounds = [center]
    for edge in prediction.get("affected_edges", []):
        geometry = edge.get("geometry", [])
        if not geometry:
            continue
        level = edge.get("impact_level", "medium")
        color = "#d73027" if level == "high" else "#f39c12"
        popup = (
            f"<b>{edge.get('name', 'Unnamed road')}</b><br>"
            f"Impact: {level}<br>"
            f"Distance from event: {edge.get('distance_from_event_m', '-')} m<br>"
            f"Length: {edge.get('length_m', 0):.0f} m"
        )
        folium.PolyLine(
            geometry,
            color=color,
            weight=5 if level == "high" else 3,
            opacity=0.78,
            tooltip=f"Affected: {edge.get('name', 'Unnamed road')}",
            popup=folium.Popup(popup, max_width=300),
        ).add_to(affected_group)
        bounds.extend(geometry)
    affected_group.add_to(fmap)

    pressure_group = folium.FeatureGroup(name="Bernoulli pressure field", show=show_pressure_heatmap)
    for edge in diversions.get("pressure_field", []):
        geometry = edge.get("geometry", [])
        if not geometry:
            continue
        pressure = float(edge.get("pressure", 0.0))
        popup = (
            f"<b>{edge.get('name', 'Unnamed road')}</b><br>"
            f"Pressure P: {pressure:.3f}<br>"
            f"Potential E: {float(edge.get('potential', 0.0)):.3f}<br>"
            f"Predicted speed: {float(edge.get('predicted_speed_kph', 0.0)):.1f} km/h<br>"
            f"Free-flow speed: {float(edge.get('free_flow_speed_kph', 0.0)):.1f} km/h"
        )
        folium.PolyLine(
            geometry,
            color=_pressure_color(pressure),
            weight=3 + min(5, pressure * 5),
            opacity=0.62,
            tooltip=f"Pressure {pressure:.2f}: {edge.get('name', 'road')}",
            popup=folium.Popup(popup, max_width=300),
        ).add_to(pressure_group)
    pressure_group.add_to(fmap)

    intersection_group = folium.FeatureGroup(name="Affected intersections", show=False)
    intersection_points: dict[str, list[float]] = {}
    for edge in prediction.get("affected_edges", []):
        geometry = edge.get("geometry", [])
        if len(geometry) >= 2:
            intersection_points.setdefault(edge["u"], geometry[0])
            intersection_points.setdefault(edge["v"], geometry[-1])
    for node, point in intersection_points.items():
        folium.CircleMarker(
            point,
            radius=3,
            color="#4b5563",
            fill=True,
            fill_color="#ffffff",
            fill_opacity=0.9,
            weight=1,
            tooltip=f"Affected intersection {node}",
        ).add_to(intersection_group)
    intersection_group.add_to(fmap)

    police_group = folium.FeatureGroup(name="Police deployment", show=True)
    for item in manpower.get("deployment", []):
        count = int(item.get("recommended_officers", 1))
        point = [item["latitude"], item["longitude"]]
        popup = (
            f"<b>Police deployment</b><br>"
            f"Officers: {count}<br>"
            f"Role: {str(item.get('role', 'traffic_control')).replace('_', ' ')}<br>"
            f"Deployment score: {item.get('deployment_score', '-')}"
        )
        folium.CircleMarker(
            point,
            radius=6 + min(count, 6),
            color="#075985",
            weight=2,
            fill=True,
            fill_color="#0ea5e9",
            fill_opacity=0.9,
            tooltip=f"Police: {count} officer{'s' if count != 1 else ''}",
            popup=folium.Popup(popup, max_width=280),
        ).add_to(police_group)
        folium.Marker(
            point,
            icon=folium.DivIcon(
                html=(
                    "<div style='color:white;font:bold 11px Segoe UI;text-align:center;"
                    "width:24px;margin-left:-12px;margin-top:-8px'>"
                    f"{count}</div>"
                )
            ),
        ).add_to(police_group)
        bounds.append(point)
    police_group.add_to(fmap)

    best_name = barricades.get("best_plan", {}).get("plan_name")
    for plan in barricades.get("plans", []):
        plan_name = plan.get("plan_name", "plan")
        is_best = plan_name == best_name
        group = folium.FeatureGroup(
            name=f"Barricade: {plan_name}{' (recommended)' if is_best else ''}",
            show=is_best,
        )
        color = PLAN_COLORS.get(plan_name, "#111827")
        for edge in plan.get("protected_edges", []):
            geometry = edge.get("geometry", [])
            if not geometry:
                continue
            popup = (
                f"<b>Incident protection</b><br>"
                f"Road: {edge.get('name', 'Unnamed road')}<br>"
                "Action: cones/soft protection near the obstruction; no closure recommended."
            )
            folium.PolyLine(
                geometry,
                color="#0f766e",
                weight=7,
                opacity=0.9,
                dash_array="3 8",
                tooltip=f"Incident protection: {edge.get('name', 'road')}",
                popup=folium.Popup(popup, max_width=300),
            ).add_to(group)
        for edge in plan.get("closed_edges", []):
            geometry = edge.get("geometry", [])
            if not geometry:
                continue
            popup = (
                f"<b>{plan_name.replace('_', ' ').title()} barricade</b><br>"
                f"Road: {edge.get('name', 'Unnamed road')}<br>"
                f"Safety protection: {plan.get('safety_protection', 0) * 100:.0f}%<br>"
                f"Congestion score: {plan.get('congestion_score', 0):.3f}<br>"
                f"Throughput score: {plan.get('throughput_score', 0):.3f}"
            )
            folium.PolyLine(
                geometry,
                color=color,
                weight=8 if is_best else 6,
                opacity=0.95 if is_best else 0.7,
                dash_array="8 6",
                tooltip=f"Barricade: {edge.get('name', 'road')}",
                popup=folium.Popup(popup, max_width=300),
            ).add_to(group)
        group.add_to(fmap)

    diversion_group = folium.FeatureGroup(name="Diversion routes", show=show_manual_diversions)
    for index, route in enumerate(diversions.get("routes", [])):
        geometry = route.get("geometry", [])
        if not geometry:
            continue
        color = ROUTE_COLORS[index % len(ROUTE_COLORS)]
        popup = (
            f"<b>Diversion {route.get('route_id', index + 1)}</b><br>"
            f"Bypasses: {route.get('bypasses', 'closure')}<br>"
            f"Distance: {route.get('distance_m', 0) / 1000:.2f} km<br>"
            f"Travel time: {route.get('estimated_travel_time_s', 0) / 60:.1f} min"
        )
        folium.PolyLine(
            geometry,
            color=color,
            weight=5,
            opacity=0.9,
            tooltip=f"Diversion {route.get('route_id', index + 1)}",
            popup=folium.Popup(popup, max_width=300),
        ).add_to(diversion_group)
        bounds.extend(geometry)
    diversion_group.add_to(fmap)

    bernoulli_group = folium.FeatureGroup(name="Bernoulli-optimal diversions", show=show_bernoulli_routes)
    for route in diversions.get("auto_diversion_routes", []):
        geometry = route.get("geometry", [])
        if not geometry:
            continue
        popup = (
            f"<b>Bernoulli diversion {route.get('route_id')}</b><br>"
            f"Mean pressure: {float(route.get('mean_pressure', 0.0)):.3f}<br>"
            f"Potential: {float(route.get('bernoulli_potential', 0.0)):.3f}<br>"
            f"Distance: {float(route.get('distance_m', 0.0)) / 1000:.2f} km<br>"
            f"Travel time: {float(route.get('estimated_travel_time_s', 0.0)) / 60:.1f} min"
        )
        folium.PolyLine(
            geometry,
            color="#2563eb",
            weight=5,
            opacity=0.95,
            dash_array="10 8",
            tooltip=f"Bernoulli diversion {route.get('route_id')}",
            popup=folium.Popup(popup, max_width=300),
        ).add_to(bernoulli_group)
        bounds.extend(geometry)
    bernoulli_group.add_to(fmap)

    reserve = manpower.get("reserve_officers", 0)
    legend = f"""
    <div style="position:fixed;bottom:22px;left:22px;z-index:9999;background:white;
      border:1px solid #9ca3af;padding:12px 14px;width:260px;font:12px/1.45 Segoe UI;color:#111827">
      <b style="font-size:13px">Map artifacts</b><br>
      <span style="color:#d73027">━━</span> High-impact road<br>
      <span style="color:#f39c12">━━</span> Medium-impact road<br>
      <span style="color:#111827">┄┄</span> Recommended barricade<br>
      <span style="color:#14866d">━━</span> Diversion around a closure<br>
      <span style="color:#2563eb">┄┄</span> Bernoulli-optimal diversion<br>
      <span style="color:#1a9850">━━</span>/<span style="color:#fee08b">━━</span>/<span style="color:#d73027">━━</span> Pressure low/medium/high<br>
      <span style="color:#0ea5e9">●</span> Police post; number is officers<br>
      <span style="color:#ef4444">●</span> Event location<br>
      <hr style="border:0;border-top:1px solid #ddd">
      Assigned: {manpower.get('assigned_officers', 0)} &nbsp; Reserve: {reserve}<br>
      Use the layer control to compare all barricade plans.
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(fmap)
    if len(bounds) > 1:
        fmap.fit_bounds(bounds, padding=(24, 24), max_zoom=16)
    return fmap


def _pressure_color(pressure: float) -> str:
    if pressure < 0.35:
        return "#1a9850"
    if pressure < 0.6:
        return "#fee08b"
    return "#d73027"
