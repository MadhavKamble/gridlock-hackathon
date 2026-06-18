# Project Status — Event-Driven Congestion Management

**Theme 2 — Flipkart Gridlock 2.0 (Phase 2)**
Last updated: 2026-06-18 · Status: **Working end-to-end, submission-prep in progress**

> This is the single source of truth for "where the project stands today."
> For planned work see [ROADMAP.md](ROADMAP.md), for change history see [CHANGELOG.md](CHANGELOG.md),
> and for the reasoning behind key choices see [DECISIONS.md](DECISIONS.md).

---

## 1. Problem we are solving

From the official problem statement:

> Political rallies, festivals, sports events, construction activities, and sudden gatherings
> create localized traffic breakdowns. Event impact is not quantified in advance, resource
> deployment is experience-driven, and there is no post-event learning system.
> **How can historical and real-time data be used to forecast event-related traffic impact
> and recommend optimal manpower, barricading, and diversion plans?**

The submission must therefore deliver three recommendations — **manpower, barricading, diversion** —
plus a **post-event learning loop**. All four are implemented.

---

## 2. What the system does (current capabilities)

A 9-stage pipeline turns a single event (location, cause, priority, time, closure flag) into a
complete operational response, surfaced through a Streamlit operator UI.

| Stage | Script | Output |
|-------|--------|--------|
| 01 | `01_prepare_data.py` | Cleans 8,173 events, engineers temporal/categorical features → `data/train_data.csv` |
| 02 | `02_build_network.py` | Downloads real Bengaluru drive network via OSMnx (155,376 nodes / 393,737 edges); deterministic fallback if Overpass is unreachable |
| 03 | `03_train_duration_model.py` | Trains `XGBRegressor(enable_categorical=True)`; saves model bundle |
| 04 | `04_predict_impact.py` | Classifies road context, estimates impact duration, finds affected roads via corridor match + Dijkstra radius + edge expansion |
| 05 | `05_manpower_optimizer.py` | OR-Tools CP-SAT officer allocation maximizing weighted intersection coverage; greedy fallback |
| 06 | `06_barricade_simulator.py` | Scores 3 barricade plans (congestion / throughput / travel-time); CityFlow optional, graph engine default |
| 07 | `07_diversion_routes.py` | Direct bypass routes + experimental **Bernoulli pressure-field** diversion candidates; SUMO optional |
| 08 | `08_generate_dashboard.py` | Folium GeoJSON + `dashboard.html` |
| 09 | `09_mlflow_logger.py` | Logs metrics/params/artifacts to MLflow SQLite; threshold-based retrain hook |

**Operator-facing layer**
- `app/main.py` — Streamlit app: map-click event entry, one-click response generation, plain-English
  decision summary, layered interactive map, outcome logging for the learning loop.
- `app/realtime_monitor.py` — dependency-free live pipeline monitor (per-stage status, streaming logs, artifact links).

---

## 3. Verified working state (as of 2026-06-18)

- ✅ `.venv` created (Python 3.13.13); all `requirements.txt` deps installed cleanly.
- ✅ Full pipeline 01→09 runs end-to-end on real data.
  - Last run: 350 affected edges, 150 candidate intersections, 8 officers deployed,
    2 manual + 8 Bernoulli diversion routes, dashboard + MLflow run produced.
- ✅ OSM network downloaded live (155,376 nodes / 393,737 edges).
- ✅ Streamlit app boots (HTTP 200 on :8501).
- ✅ Test suite passes (`2 passed`).

---

## 4. Known limitations & risks

| # | Item | Severity | Notes |
|---|------|----------|-------|
| L1 | **"Duration model" is decorative** | High (framing risk) | Dataset has no event-end timestamp, so the model learns `report_creation_delay_min` (delay between event start and report creation), **not congestion duration**. Metrics: MAE 1.32, RMSE 2.91, **R² ≈ 0**. Real duration comes from the rule-based `estimate_operational_impact`. This is honest in code but must be framed proactively to judges — see [DECISIONS.md](DECISIONS.md) D1. |
| L2 | `run_all.sh` data path broken | Medium | Uses `../cleaned_gridlock.csv` (does not exist); real path is `data/cleaned_gridlock.csv`. README Quick Start fails out-of-box. Streamlit bootstrap works because it searches `data/` first. |
| L3 | `pytest` missing from requirements | Low | Tests depend on it; not declared. |
| L4 | `runtime.txt` pins 3.10.14 | Low | Verified working on 3.13; align before deploy. |
| L5 | Efficiency hot spots on 155k-node graph | Low | `set(intersections)` rebuilt per node (05:31); `node_lookup` 155k-dict rebuilt repeatedly (07); possibly-unused `_coerce_node` linear scan (04:228). |

---

## 5. Bugs already fixed

- **XGBoost unseen-category crash (blocking).** XGBoost 3.x raises on categories not seen in
  training (`Found a category not in the training set ... 'Outer Ring Road'`), which crashed
  stage 04 and cascaded to 05–08. Fixed by persisting `categorical_levels` in the model bundle
  (stage 03) and reapplying them at predict time so unseen values become NaN/missing (stage 04).
  See [CHANGELOG.md](CHANGELOG.md) and [DECISIONS.md](DECISIONS.md) D2.

---

## 6. Tech stack

Python 3.13 · pandas/numpy · scikit-learn · XGBoost · OSMnx + NetworkX · GeoPandas/Shapely ·
OR-Tools (CP-SAT) · MLflow · Streamlit + Folium + streamlit-folium · optional CityFlow / SUMO.

Deployment-ready: Dockerfile, render.yaml, Procfile, Streamlit Cloud config.
