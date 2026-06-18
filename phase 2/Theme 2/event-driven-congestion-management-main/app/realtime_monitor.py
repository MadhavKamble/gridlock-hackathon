from __future__ import annotations

import json
import mimetypes
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from lib.runtime_state import LOG_PATH, RUNTIME_DIR, STATE_PATH, append_log, make_state, read_state as load_runtime_state, reset_log, utc_now, write_state

PROJECT_PYTHON = ROOT / "venv" / "Scripts" / "python.exe"
PIPELINE_PYTHON = str(PROJECT_PYTHON if PROJECT_PYTHON.exists() else Path(sys.executable))

PIPELINE = [
    ("prepare_data", "Prepare event data", ["scripts/01_prepare_data.py", "--input", "../cleaned_gridlock.csv"]),
    ("build_network", "Build Bengaluru road network", ["scripts/02_build_network.py"]),
    ("train_model", "Train duration model", ["scripts/03_train_duration_model.py"]),
    ("predict_impact", "Predict traffic impact", ["scripts/04_predict_impact.py"]),
    ("optimize_manpower", "Optimize police deployment", ["scripts/05_manpower_optimizer.py"]),
    ("simulate_barricades", "Simulate barricade plans", ["scripts/06_barricade_simulator.py"]),
    ("generate_diversions", "Generate diversion routes", ["scripts/07_diversion_routes.py"]),
    ("build_dashboard", "Build traffic dashboard", ["scripts/08_generate_dashboard.py"]),
    ("log_mlflow", "Log model run", ["scripts/09_mlflow_logger.py", "--mode", "log-latest"]),
]

ARTIFACTS = [
    ("Training data", "data/train_data.csv"),
    ("Road network", "road_network/bangalore_graph.graphml"),
    ("Duration model", "models/duration_model.pkl"),
    ("Model metrics", "models/duration_metrics.json"),
    ("Impact prediction", "output/predictions/latest_prediction.json"),
    ("Manpower plan", "output/predictions/manpower_plan.json"),
    ("Barricade plan", "output/predictions/barricade_plan.json"),
    ("Diversion routes", "output/predictions/diversion_routes.json"),
    ("Map data", "output/dashboards/dashboard.geojson"),
    ("Traffic map", "output/dashboards/dashboard.html"),
    ("Pipeline log", "output/runtime/pipeline.log"),
]

LOCK = threading.RLock()
PROCESS: subprocess.Popen[str] | None = None
RUNNER: threading.Thread | None = None
STOP_REQUESTED = False


def initial_state() -> dict[str, Any]:
    return make_state([(step_id, name) for step_id, name, _ in PIPELINE])


def read_state() -> dict[str, Any]:
    with LOCK:
        return load_runtime_state([(step_id, name) for step_id, name, _ in PIPELINE])


def update_step(state: dict[str, Any], step_id: str, **changes: Any) -> None:
    for step in state["steps"]:
        if step["id"] == step_id:
            step.update(changes)
            return


def run_pipeline() -> None:
    global PROCESS, STOP_REQUESTED
    state = initial_state()
    state.update(
        {
            "run_id": datetime.now().strftime("%Y%m%d-%H%M%S"),
            "status": "running",
            "started_at": utc_now(),
            "message": "Pipeline started",
        }
    )
    write_state(state)
    append_log(f"Pipeline run {state['run_id']} started", "RUN")
    started = time.monotonic()

    for step_id, name, command in PIPELINE:
        with LOCK:
            if STOP_REQUESTED:
                break
        step_started = time.monotonic()
        state["current_step"] = step_id
        state["message"] = name
        update_step(state, step_id, status="running", started_at=utc_now())
        state["elapsed_seconds"] = round(time.monotonic() - started, 1)
        write_state(state)
        append_log(f"{name}: {' '.join(command)}", "START")

        try:
            PROCESS = subprocess.Popen(
                [PIPELINE_PYTHON, *command],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert PROCESS.stdout is not None
            for output_line in iter(PROCESS.stdout.readline, ""):
                if output_line:
                    append_log(f"{step_id} | {output_line.rstrip()}")
                with LOCK:
                    if STOP_REQUESTED and PROCESS.poll() is None:
                        PROCESS.terminate()
            return_code = PROCESS.wait()
        except Exception as exc:
            return_code = -1
            append_log(f"{name} failed to launch: {exc}", "ERROR")
        finally:
            PROCESS = None

        duration = round(time.monotonic() - step_started, 2)
        state["elapsed_seconds"] = round(time.monotonic() - started, 1)
        if return_code != 0:
            update_step(
                state,
                step_id,
                status="failed",
                finished_at=utc_now(),
                duration_seconds=duration,
                return_code=return_code,
            )
            state.update(
                {
                    "status": "failed",
                    "finished_at": utc_now(),
                    "current_step": step_id,
                    "message": f"{name} failed with exit code {return_code}",
                }
            )
            write_state(state)
            append_log(state["message"], "ERROR")
            return

        update_step(
            state,
            step_id,
            status="completed",
            finished_at=utc_now(),
            duration_seconds=duration,
            return_code=return_code,
        )
        write_state(state)
        append_log(f"{name} completed in {duration:.2f}s", "DONE")

    state["elapsed_seconds"] = round(time.monotonic() - started, 1)
    state["finished_at"] = utc_now()
    state["current_step"] = None
    if STOP_REQUESTED:
        state["status"] = "stopped"
        state["message"] = "Pipeline stopped"
        append_log("Pipeline stopped by operator", "STOP")
    else:
        state["status"] = "completed"
        state["message"] = "All pipeline steps completed"
        append_log(f"Pipeline completed in {state['elapsed_seconds']:.1f}s", "DONE")
    write_state(state)
    with LOCK:
        STOP_REQUESTED = False


def start_pipeline() -> tuple[bool, str]:
    global RUNNER, STOP_REQUESTED
    with LOCK:
        if RUNNER and RUNNER.is_alive():
            return False, "Pipeline is already running"
        STOP_REQUESTED = False
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        reset_log()
        RUNNER = threading.Thread(target=run_pipeline, name="pipeline-runner", daemon=True)
        RUNNER.start()
    return True, "Pipeline started"


def stop_pipeline() -> tuple[bool, str]:
    global STOP_REQUESTED
    with LOCK:
        if not RUNNER or not RUNNER.is_alive():
            return False, "No pipeline is running"
        STOP_REQUESTED = True
        if PROCESS and PROCESS.poll() is None:
            PROCESS.terminate()
    return True, "Stop requested"


def live_payload() -> dict[str, Any]:
    state = read_state()
    if state["status"] == "running" and state.get("started_at"):
        start = datetime.fromisoformat(state["started_at"])
        state["elapsed_seconds"] = round((datetime.now(timezone.utc) - start).total_seconds(), 1)
    log_lines: list[str] = []
    if LOG_PATH.exists():
        try:
            log_lines = LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()[-400:]
        except OSError:
            pass
    return {
        "state": state,
        "logs": log_lines,
        "artifacts": artifact_payload(),
        "summary": output_summary(),
        "server_time": utc_now(),
    }


def artifact_payload() -> list[dict[str, Any]]:
    result = []
    for label, relative in ARTIFACTS:
        path = ROOT / relative
        exists = path.exists()
        stat = path.stat() if exists else None
        result.append(
            {
                "label": label,
                "path": relative.replace("\\", "/"),
                "exists": exists,
                "size_bytes": stat.st_size if stat else 0,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat() if stat else None,
            }
        )
    return result


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def output_summary() -> dict[str, Any]:
    metrics = read_json(ROOT / "models" / "duration_metrics.json")
    prediction = read_json(ROOT / "output" / "predictions" / "latest_prediction.json")
    manpower = read_json(ROOT / "output" / "predictions" / "manpower_plan.json")
    barricades = read_json(ROOT / "output" / "predictions" / "barricade_plan.json")
    diversions = read_json(ROOT / "output" / "predictions" / "diversion_routes.json")
    return {
        "model_type": metrics.get("model_type", "-"),
        "mae": metrics.get("mae"),
        "rmse": metrics.get("rmse"),
        "r2": metrics.get("r2"),
        "predicted_duration_min": prediction.get("predicted_duration_min"),
        "affected_roads": len(prediction.get("affected_edges", [])),
        "intersections": len(prediction.get("intersections", [])),
        "assigned_officers": manpower.get("assigned_officers"),
        "optimizer": manpower.get("optimizer", "-"),
        "barricade_plan": barricades.get("best_plan", {}).get("plan_name", "-"),
        "simulation_engine": barricades.get("simulation_engine", "-"),
        "diversion_routes": len(diversions.get("routes", [])),
    }


def safe_artifact_path(relative: str) -> Path | None:
    candidate = (ROOT / relative).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError:
        return None
    allowed = {(ROOT / path).resolve() for _, path in ARTIFACTS}
    return candidate if candidate in allowed and candidate.is_file() else None


class MonitorHandler(BaseHTTPRequestHandler):
    server_version = "CongestionMonitor/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_bytes(DASHBOARD_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/live":
            self.send_json(live_payload())
            return
        if parsed.path == "/artifact":
            relative = parse_qs(parsed.query).get("path", [""])[0]
            artifact = safe_artifact_path(relative)
            if artifact is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type = mimetypes.guess_type(artifact.name)[0] or "application/octet-stream"
            self.send_bytes(artifact.read_bytes(), content_type)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path == "/api/start":
            ok, message = start_pipeline()
            self.send_json({"ok": ok, "message": message}, HTTPStatus.ACCEPTED if ok else HTTPStatus.CONFLICT)
            return
        if self.path == "/api/stop":
            ok, message = stop_pipeline()
            self.send_json({"ok": ok, "message": message}, HTTPStatus.ACCEPTED if ok else HTTPStatus.CONFLICT)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_bytes(json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8", status)

    def send_bytes(self, payload: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        return


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Bengaluru Traffic Operations</title>
  <style>
    :root { color-scheme: dark; --bg:#111416; --panel:#181d20; --line:#30383d; --text:#edf2f4; --muted:#9eabb2; --cyan:#39c5bb; --green:#73d13d; --amber:#f5b942; --red:#ff6464; --blue:#58a6ff; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 "Segoe UI", Arial, sans-serif; letter-spacing:0; }
    header { height:64px; padding:0 24px; display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid var(--line); background:#15191c; position:sticky; top:0; z-index:4; }
    h1 { margin:0; font-size:19px; font-weight:650; }
    h2 { margin:0 0 12px; font-size:14px; text-transform:uppercase; color:#c5d0d5; }
    button { height:36px; padding:0 14px; border:1px solid var(--line); border-radius:5px; color:var(--text); background:#232a2e; cursor:pointer; font-weight:600; }
    button.primary { background:#16776f; border-color:#26988e; }
    button.danger { color:#ffb3b3; }
    button:disabled { opacity:.45; cursor:not-allowed; }
    main { width:min(1500px, 100%); margin:0 auto; padding:20px 24px 28px; }
    .toolbar { display:flex; gap:8px; align-items:center; }
    .pulse { width:9px; height:9px; border-radius:50%; background:var(--muted); display:inline-block; margin-right:8px; }
    .pulse.running { background:var(--cyan); box-shadow:0 0 0 5px rgba(57,197,187,.12); }
    .pulse.completed { background:var(--green); }
    .pulse.failed { background:var(--red); }
    .pulse.stopped { background:var(--amber); }
    .status-line { color:var(--muted); white-space:nowrap; }
    .metrics { display:grid; grid-template-columns:repeat(6, minmax(120px,1fr)); border:1px solid var(--line); background:var(--panel); margin-bottom:18px; }
    .metric { padding:13px 15px; min-height:72px; border-right:1px solid var(--line); }
    .metric:last-child { border-right:0; }
    .metric label { display:block; color:var(--muted); font-size:12px; margin-bottom:7px; }
    .metric strong { font-size:20px; font-weight:650; overflow-wrap:anywhere; }
    .layout { display:grid; grid-template-columns:minmax(500px, .9fr) minmax(520px, 1.1fr); gap:18px; }
    .section { border:1px solid var(--line); background:var(--panel); min-width:0; }
    .section-head { padding:13px 15px; border-bottom:1px solid var(--line); display:flex; align-items:center; justify-content:space-between; }
    .section-head h2 { margin:0; }
    table { border-collapse:collapse; width:100%; }
    th, td { padding:10px 12px; border-bottom:1px solid #293035; text-align:left; }
    th { color:var(--muted); font-size:11px; text-transform:uppercase; font-weight:600; }
    tr:last-child td { border-bottom:0; }
    .step-state { font-size:11px; font-weight:700; text-transform:uppercase; }
    .step-state.running { color:var(--cyan); }
    .step-state.completed { color:var(--green); }
    .step-state.failed { color:var(--red); }
    .step-state.pending { color:#738087; }
    .log { height:470px; overflow:auto; margin:0; padding:14px; background:#0c0f11; color:#c9d1d5; white-space:pre-wrap; overflow-wrap:anywhere; font:12px/1.55 Consolas, monospace; }
    .artifacts { margin-top:18px; }
    .artifact-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); }
    .artifact { display:flex; align-items:center; justify-content:space-between; gap:12px; padding:11px 14px; border-bottom:1px solid #293035; }
    .artifact:nth-child(odd) { border-right:1px solid #293035; }
    .artifact a { color:#a8d5ff; text-decoration:none; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .artifact small { color:var(--muted); white-space:nowrap; }
    .missing { color:#657178; }
    @media (max-width:1050px) { .metrics{grid-template-columns:repeat(3,1fr)} .metric:nth-child(3){border-right:0} .layout{grid-template-columns:1fr} }
    @media (max-width:650px) { header{height:auto;padding:14px 16px;align-items:flex-start;gap:12px} main{padding:14px 12px}.toolbar{flex-wrap:wrap;justify-content:flex-end}.metrics{grid-template-columns:repeat(2,1fr)}.metric:nth-child(3){border-right:1px solid var(--line)}.metric:nth-child(even){border-right:0}.artifact-grid{grid-template-columns:1fr}.artifact:nth-child(odd){border-right:0}.layout{display:block}.section{margin-bottom:14px} }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Bengaluru Traffic Operations</h1>
      <div class="status-line"><span id="pulse" class="pulse"></span><span id="runStatus">Connecting</span></div>
    </div>
    <div class="toolbar">
      <button id="startBtn" class="primary" onclick="command('start')">Run pipeline</button>
      <button id="stopBtn" class="danger" onclick="command('stop')" disabled>Stop</button>
    </div>
  </header>
  <main>
    <div class="metrics">
      <div class="metric"><label>Elapsed</label><strong id="elapsed">0s</strong></div>
      <div class="metric"><label>Predicted impact</label><strong id="duration">-</strong></div>
      <div class="metric"><label>Affected roads</label><strong id="roads">-</strong></div>
      <div class="metric"><label>Officers assigned</label><strong id="officers">-</strong></div>
      <div class="metric"><label>Diversions</label><strong id="diversions">-</strong></div>
      <div class="metric"><label>Model MAE</label><strong id="mae">-</strong></div>
    </div>
    <div class="layout">
      <section class="section">
        <div class="section-head"><h2>Pipeline Steps</h2><span id="runId" class="status-line">No run</span></div>
        <table>
          <thead><tr><th>Stage</th><th>Status</th><th>Duration</th></tr></thead>
          <tbody id="steps"></tbody>
        </table>
      </section>
      <section class="section">
        <div class="section-head"><h2>Live Log</h2><span id="logCount" class="status-line">0 lines</span></div>
        <pre id="log" class="log">Waiting for pipeline activity...</pre>
      </section>
    </div>
    <section class="section artifacts">
      <div class="section-head"><h2>Generated Files</h2><span id="artifactCount" class="status-line">0 files</span></div>
      <div id="artifacts" class="artifact-grid"></div>
    </section>
  </main>
  <script>
    let lastLogLength = 0;
    const fmtSeconds = value => value == null ? "-" : `${Number(value).toFixed(value < 10 ? 1 : 0)}s`;
    const fmtBytes = value => {
      if (!value) return "0 B";
      const units = ["B","KB","MB","GB"]; let index=0, size=value;
      while(size >= 1024 && index < units.length-1){ size/=1024; index++; }
      return `${size.toFixed(index ? 1 : 0)} ${units[index]}`;
    };
    const valueOrDash = value => value == null ? "-" : value;
    async function command(action) {
      const response = await fetch(`/api/${action}`, {method:"POST"});
      const result = await response.json();
      if (!result.ok && response.status !== 202) alert(result.message);
      await refresh();
    }
    async function refresh() {
      try {
        const response = await fetch("/api/live", {cache:"no-store"});
        const data = await response.json();
        const state = data.state, summary = data.summary;
        document.getElementById("pulse").className = `pulse ${state.status}`;
        document.getElementById("runStatus").textContent = `${state.status.toUpperCase()} · ${state.message}`;
        document.getElementById("runId").textContent = state.run_id ? `Run ${state.run_id}` : "No run";
        document.getElementById("elapsed").textContent = fmtSeconds(state.elapsed_seconds);
        document.getElementById("duration").textContent = summary.predicted_duration_min == null ? "-" : `${summary.predicted_duration_min} min`;
        document.getElementById("roads").textContent = valueOrDash(summary.affected_roads);
        document.getElementById("officers").textContent = valueOrDash(summary.assigned_officers);
        document.getElementById("diversions").textContent = valueOrDash(summary.diversion_routes);
        document.getElementById("mae").textContent = summary.mae == null ? "-" : Number(summary.mae).toFixed(2);
        document.getElementById("startBtn").disabled = state.status === "running";
        document.getElementById("stopBtn").disabled = state.status !== "running";
        document.getElementById("steps").innerHTML = state.steps.map(step =>
          `<tr><td>${step.name}</td><td><span class="step-state ${step.status}">${step.status}</span></td><td>${fmtSeconds(step.duration_seconds)}</td></tr>`
        ).join("");
        const log = document.getElementById("log");
        const shouldScroll = log.scrollTop + log.clientHeight >= log.scrollHeight - 30 || data.logs.length !== lastLogLength;
        log.textContent = data.logs.length ? data.logs.join("\n") : "Waiting for pipeline activity...";
        document.getElementById("logCount").textContent = `${data.logs.length} lines`;
        if (shouldScroll) log.scrollTop = log.scrollHeight;
        lastLogLength = data.logs.length;
        const existing = data.artifacts.filter(item => item.exists);
        document.getElementById("artifactCount").textContent = `${existing.length} files`;
        document.getElementById("artifacts").innerHTML = data.artifacts.map(item =>
          `<div class="artifact ${item.exists ? "" : "missing"}">
             ${item.exists ? `<a href="/artifact?path=${encodeURIComponent(item.path)}" target="_blank">${item.label}</a>` : `<span>${item.label}</span>`}
             <small>${item.exists ? fmtBytes(item.size_bytes) : "pending"}</small>
           </div>`
        ).join("");
      } catch (error) {
        document.getElementById("runStatus").textContent = "DISCONNECTED · Retrying";
      }
    }
    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>"""


def main() -> None:
    host = "127.0.0.1"
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    if not STATE_PATH.exists():
        write_state(initial_state())
    server = ThreadingHTTPServer((host, port), MonitorHandler)
    print(f"Realtime pipeline dashboard: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
