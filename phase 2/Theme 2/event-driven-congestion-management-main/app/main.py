from __future__ import annotations

import json
import importlib
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import folium
import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from lib.bootstrap import ensure_runtime_artifacts
from lib.network_utils import read_json
import lib.map_utils as map_utils
from lib.paths import DASHBOARDS_DIR, PREDICTIONS_DIR, ensure_directories
from lib.runtime_state import append_log, make_state, public_state, reset_log, utc_now, write_state


def run_script(script_name: str, label: str, state: dict[str, Any], step_index: int, args: list[str] | None = None) -> None:
    command = [sys.executable, str(ROOT / "scripts" / script_name), *(args or [])]
    step = state["steps"][step_index]
    step.update({"status": "running", "started_at": utc_now()})
    state.update({"current_step": step["id"], "message": label})
    write_state(state)
    append_log(f"{label}: {script_name} {' '.join(args or [])}", "START")
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    output_lines = []
    for line in iter(process.stdout.readline, ""):
        clean = line.rstrip()
        if clean:
            output_lines.append(clean)
            append_log(f"{step['id']} | {clean}")
    return_code = process.wait()
    duration = round(time.monotonic() - started, 2)
    step.update(
        {
            "status": "completed" if return_code == 0 else "failed",
            "finished_at": utc_now(),
            "duration_seconds": duration,
            "return_code": return_code,
        }
    )
    state["elapsed_seconds"] = round(time.monotonic() - state["_started_monotonic"], 1)
    write_state(state)
    if return_code != 0:
        append_log(f"{label} failed with exit code {return_code}", "ERROR")
        raise RuntimeError("\n".join(output_lines[-30:]) or f"{label} failed")
    append_log(f"{label} completed in {duration:.2f}s", "DONE")


def run_event_response(predict_args: list[str], officers: int) -> None:
    steps = [
        ("predict_impact", "Predict traffic impact", "04_predict_impact.py", predict_args),
        ("optimize_manpower", "Optimize police deployment", "05_manpower_optimizer.py", ["--officers", str(officers)]),
        ("simulate_barricades", "Simulate barricade plans", "06_barricade_simulator.py", []),
        ("generate_diversions", "Generate diversion routes", "07_diversion_routes.py", []),
        ("build_dashboard", "Build traffic dashboard", "08_generate_dashboard.py", []),
        ("log_mlflow", "Log model run", "09_mlflow_logger.py", ["--mode", "log-latest"]),
    ]
    state = make_state(
        [(step_id, label) for step_id, label, _, _ in steps],
        run_id=f"event-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        run_type="event_response",
        status="running",
        message="Event response started",
    )
    state["_started_monotonic"] = time.monotonic()
    reset_log()
    write_state(state)
    append_log(f"Event response run {state['run_id']} started", "RUN")
    try:
        for index, (_, label, script, args) in enumerate(steps):
            run_script(script, label, state, index, args)
    except Exception:
        state.update(
            {
                "status": "failed",
                "finished_at": utc_now(),
                "elapsed_seconds": round(time.monotonic() - state["_started_monotonic"], 1),
                "message": f"{state['message']} failed",
            }
        )
        write_state(public_state(state))
        raise
    state.update(
        {
            "status": "completed",
            "current_step": None,
            "finished_at": utc_now(),
            "elapsed_seconds": round(time.monotonic() - state["_started_monotonic"], 1),
            "message": "Event response completed",
        }
    )
    write_state(public_state(state))
    append_log(f"Event response completed in {state['elapsed_seconds']:.1f}s", "DONE")


def load_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return read_json(path)


def append_outcome(payload: dict[str, Any]) -> None:
    path = ROOT / "data" / "outcomes.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _format_minutes(value: Any) -> str:
    try:
        return f"{float(value):.0f} minutes"
    except (TypeError, ValueError):
        return "unknown duration"


def _plain_language_summary(
    prediction: dict[str, Any],
    manpower: dict[str, Any],
    barricades: dict[str, Any],
    diversions: dict[str, Any],
) -> dict[str, str]:
    event = prediction.get("event", {})
    methodology = prediction.get("duration_methodology", {})
    staffing = manpower.get("staffing_model", {})
    road_context = prediction.get("road_context", {})
    best_plan = barricades.get("best_plan", {})
    affected_count = len(prediction.get("affected_edges", []))
    assigned = int(manpower.get("assigned_officers", 0))
    reserve = int(manpower.get("reserve_officers", 0))
    manual_routes = len(diversions.get("routes", []))
    auto_routes = len(diversions.get("auto_diversion_routes", []))
    start_time = str(event.get("start_datetime", "-"))
    cause = str(event.get("event_cause", "event")).replace("_", " ")
    priority = str(event.get("priority", "-"))
    closure = _yes_no(event.get("requires_road_closure", False))
    time_band = str(staffing.get("time_band", "unknown")).replace("_", " ")

    context_text = str(road_context.get("explanation", "")).strip()
    impact = (
        f"This is being treated as a {priority.lower()} priority {cause} near the red pin. "
        f"The system estimates about {_format_minutes(prediction.get('predicted_duration_min'))} of operational impact "
        f"and marks {affected_count} nearby road segments as affected."
    )
    if context_text:
        impact = f"{impact} {context_text}"
    response = (
        f"Because the event time is {start_time} ({time_band}), the system assigns {assigned} officers "
        f"and keeps {reserve} in reserve. Road closure requested: {closure}."
    )
    if best_plan.get("plan_name") == "incident_protection":
        barricade = (
            "No hard barricade or diversion is recommended. Use cones/soft protection around the vehicle "
            "and keep local access moving."
        )
    else:
        barricade = (
            f"The recommended barricade plan is '{best_plan.get('plan_name', 'none')}'. "
            f"It closes the smallest set of road pieces that protects the incident area while preserving throughput."
        )
    diversions_text = (
        f"The map contains {manual_routes} direct bypass route(s) around the selected barricade and "
        f"{auto_routes} pressure-release route(s). The pressure-release routes are planning candidates, "
        "not instructions to send every driver there."
    )
    limitation = (
        "Important: duration is an operational estimate because the CSV has no true event end time. "
        f"The estimate uses cause, priority, closure, corridor, and time factors; base cause minutes here are "
        f"{methodology.get('base_cause_minutes', 'unknown')}."
    )
    return {
        "impact": impact,
        "response": response,
        "barricade": barricade,
        "diversions": diversions_text,
        "limitation": limitation,
    }


def render_event_screen() -> None:
    st.subheader("Create Event")
    st.caption("Pick a location, choose event details, then generate a response plan.")

    if "event_latitude" not in st.session_state:
        st.session_state.event_latitude = 12.9352
    if "event_longitude" not in st.session_state:
        st.session_state.event_longitude = 77.6245
    if "event_start_date" not in st.session_state:
        st.session_state.event_start_date = datetime.now().date()
    if "event_start_time" not in st.session_state:
        st.session_state.event_start_time = datetime.now().time().replace(microsecond=0)

    top_left, top_right = st.columns([0.52, 0.48])
    with top_left:
        picker = folium.Map(
            location=[st.session_state.event_latitude, st.session_state.event_longitude],
            zoom_start=13,
            tiles="CartoDB positron",
        )
        folium.Marker(
            [st.session_state.event_latitude, st.session_state.event_longitude],
            tooltip="Selected event location",
            icon=folium.Icon(color="red", icon="warning-sign"),
        ).add_to(picker)
        picker_result = st_folium(
            picker,
            key="event_location_picker",
            use_container_width=True,
            height=520,
            returned_objects=["last_clicked"],
        )
        clicked = picker_result.get("last_clicked") if picker_result else None
        if clicked:
            clicked_lat = round(float(clicked["lat"]), 6)
            clicked_lon = round(float(clicked["lng"]), 6)
            if clicked_lat != st.session_state.event_latitude or clicked_lon != st.session_state.event_longitude:
                st.session_state.event_latitude = clicked_lat
                st.session_state.event_longitude = clicked_lon
                st.rerun()

    with top_right:
        event_type = st.segmented_control("Event type", ["unplanned", "planned"], default="unplanned")
        event_cause = st.selectbox(
            "Cause",
            ["vehicle_breakdown", "accident", "tree_fall", "water_logging", "pot_holes", "public_event", "others"],
        )
        priority = st.select_slider("Priority", ["Low", "Medium", "High", "Critical"], value="High")
        requires_closure = st.checkbox("Requires road closure")
        corridor = st.text_input("Corridor", "Non-corridor")

        col_a, col_b = st.columns(2)
        latitude = col_a.number_input("Latitude", key="event_latitude", format="%.6f")
        longitude = col_b.number_input("Longitude", key="event_longitude", format="%.6f")

        date_col, time_col = st.columns(2)
        event_date = date_col.date_input("Event date", key="event_start_date")
        event_time = time_col.time_input("Event time", key="event_start_time")
        start_datetime = datetime.combine(event_date, event_time).isoformat()

        officers = st.number_input("Available officers", min_value=0, max_value=500, value=12)
        hour = event_time.hour
        if hour in {8, 9, 10, 17, 18, 19, 20}:
            st.info("Peak-hour factor will increase impact and staffing pressure.")
        elif hour <= 5 or hour >= 23:
            st.info("Night/off-peak factor will reduce staffing pressure unless severity or closure is high.")
        else:
            st.info("Standard daytime factor will be used.")

        if st.button("Generate response plan", type="primary", use_container_width=True):
            try:
                predict_args = [
                    "--event-type",
                    str(event_type),
                    "--start-datetime",
                    start_datetime,
                    "--priority",
                    priority,
                    "--corridor",
                    corridor,
                    "--event-cause",
                    event_cause,
                    "--latitude",
                    str(latitude),
                    "--longitude",
                    str(longitude),
                ]
                if requires_closure:
                    predict_args.append("--requires-road-closure")
                with st.spinner("Generating response plan. Follow live progress at http://127.0.0.1:8765"):
                    run_event_response(predict_args, int(officers))
                st.session_state.screen = "results"
                st.rerun()
            except Exception as exc:
                st.error(str(exc))


def render_results_screen() -> None:
    prediction = load_payload(PREDICTIONS_DIR / "latest_prediction.json")
    manpower = load_payload(PREDICTIONS_DIR / "manpower_plan.json")
    barricades = load_payload(PREDICTIONS_DIR / "barricade_plan.json")
    diversions = load_payload(PREDICTIONS_DIR / "diversion_routes.json")
    event = prediction.get("event", {})

    title_col, action_col = st.columns([0.75, 0.25])
    title_col.subheader("Response Plan")
    if action_col.button("New event", use_container_width=True):
        st.session_state.screen = "event"
        st.rerun()

    metric_cols = st.columns(6)
    metric_cols[0].metric("Duration", f"{prediction.get('predicted_duration_min', 0)} min")
    metric_cols[1].metric("Affected roads", len(prediction.get("affected_edges", [])))
    metric_cols[2].metric("Officers", manpower.get("assigned_officers", 0))
    metric_cols[3].metric("Reserve", manpower.get("reserve_officers", 0))
    metric_cols[4].metric("Manual diversions", len(diversions.get("routes", [])))
    metric_cols[5].metric("Bernoulli routes", len(diversions.get("auto_diversion_routes", [])))

    st.caption(
        f"{event.get('event_cause', 'event').replace('_', ' ').title()} | "
        f"{event.get('priority', '-')} | {event.get('start_datetime', '-')}"
    )

    summary = _plain_language_summary(prediction, manpower, barricades, diversions)
    st.subheader("Plain-English Decision")
    st.markdown(f"**Impact:** {summary['impact']}")
    st.markdown(f"**Police response:** {summary['response']}")
    st.markdown(f"**Barricades:** {summary['barricade']}")
    st.markdown(f"**Diversions:** {summary['diversions']}")

    st.subheader("Map")
    layer_cols = st.columns(3)
    show_pressure = layer_cols[0].toggle("Show pressure field", value=False)
    show_manual = layer_cols[1].toggle("Show bypass diversions", value=True)
    show_bernoulli = layer_cols[2].toggle("Show pressure-release routes", value=False)
    reloaded_map_utils = importlib.reload(map_utils)
    fmap = reloaded_map_utils.build_response_map(
        prediction,
        manpower,
        barricades,
        diversions,
        show_pressure_heatmap=show_pressure,
        show_manual_diversions=show_manual,
        show_bernoulli_routes=show_bernoulli,
    )
    components.html(fmap.get_root().render(), height=720, scrolling=False)

    with st.expander("What the map shows", expanded=False):
        st.markdown(
            """
            - **Red/orange roads:** roads expected to be directly affected, ranked by distance from the event.
            - **Dashed dark roads:** barricaded edges in the recommended plan. Other plans are available in the layer control.
            - **Green/blue routes:** direct bypasses around the recommended barricade.
            - **Dashed blue routes:** pressure-release candidates. Use these for planning, not as final driver instructions.
            - **Green/yellow/red pressure layer:** local road tension, from low to high.
            - **Blue numbered circles:** police posts; the number is the officers allocated at that junction.
            - **Small intersection dots:** candidate junctions within the affected road neighborhood.
            """
        )
    methodology = prediction.get("duration_methodology", {})
    staffing = manpower.get("staffing_model", {})
    if methodology:
        st.warning(summary["limitation"])
    if staffing:
        st.info(
            "Staffing considers priority, closure, cause, peak/off-peak hour, and predicted duration. "
            f"Current demand factor: {staffing.get('demand_factor')}; time factor: {staffing.get('time_factor')}."
        )

    outcome_col, json_col = st.columns([0.35, 0.65])
    with outcome_col:
        st.subheader("Outcome")
        actual_duration = st.number_input("Actual duration minutes", min_value=0.0, value=0.0)
        outcome_status = st.selectbox("Outcome status", ["active", "resolved", "closed"])
        if st.button("Log outcome", use_container_width=True):
            append_outcome(
                {
                    "logged_at": utc_now(),
                    "actual_duration_min": actual_duration,
                    "status": outcome_status,
                    "prediction": prediction,
                }
            )
            st.success("Outcome logged for learning workflow.")
    with json_col:
        st.subheader("Recommendations")
        st.json(
            {
                "manpower": manpower,
                "barricades": barricades.get("best_plan", {}),
                "bernoulli": {
                    "routes": len(diversions.get("auto_diversion_routes", [])),
                    "parameters": diversions.get("bernoulli_parameters", {}),
                },
                "dashboard": str(DASHBOARDS_DIR / "dashboard.html"),
            }
        )


def main() -> None:
    ensure_directories()
    st.set_page_config(page_title="Bengaluru Congestion Manager", layout="wide")
    st.title("Bengaluru Event-Driven Congestion Management")
    with st.spinner("Checking runtime artifacts..."):
        bootstrap_status = ensure_runtime_artifacts()
    if bootstrap_status.get("warning"):
        st.warning(bootstrap_status["warning"])
    elif bootstrap_status.get("bootstrapped"):
        st.success("Runtime artifacts were generated for this deployment.")
    if "screen" not in st.session_state:
        st.session_state.screen = "event"
    if st.session_state.screen == "results":
        render_results_screen()
    else:
        render_event_screen()


if __name__ == "__main__":
    main()
