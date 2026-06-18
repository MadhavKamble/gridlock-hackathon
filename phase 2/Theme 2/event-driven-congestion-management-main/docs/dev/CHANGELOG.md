# Changelog

All notable changes to the project during submission prep. Newest first.
Format: each entry is dated, tagged `[Fixed] / [Added] / [Changed] / [Docs]`, and links the
roadmap item ([ROADMAP.md](ROADMAP.md)) or decision ([DECISIONS.md](DECISIONS.md)) it relates to.

---

## 2026-06-18

### [Fixed] `run_all.sh` data path (R1)
- **Problem:** Stage 01 invoked `--input ../cleaned_gridlock.csv`, a path that does not exist; the
  CSV lives at `data/cleaned_gridlock.csv`. README Quick Start failed out-of-box.
- **Change:** `run_all.sh` now uses `--input data/cleaned_gridlock.csv`.
- **Verified:** Clean `bash run_all.sh` completes all 9 stages (350 edges, 150 intersections,
  12 officers, 2 manual + 8 Bernoulli routes, dashboard generated).

### [Added] `pytest` declared in requirements (R2)
- **Problem:** Test suite depends on pytest but it was not in `requirements.txt`.
- **Change:** Added `pytest>=8,<9`.
- **Verified:** `pip install --dry-run -r requirements.txt` resolves pytest 8.4.2; `pytest` → 2 passed.

### [Added] Close the learning loop — train + consume outcomes (R10)
- **What:** The post-event learning loop now genuinely learns from collected durations.
  - `lib/outcome_model.py` — builds a labeled set from `outcomes.jsonl` (`actual_duration_min` + event
    features via `event_to_frame`), trains an XGBoost duration model gated at `MIN_OUTCOMES=30`
    (held-out split at ≥4×), with `blend_weight(n) = min(1, n/200)`.
  - `scripts/10_train_from_outcomes.py` — CLI trainer; logs the model + metrics to MLflow.
  - `09_mlflow_logger.py` — `retrain-if-needed` now also triggers stage 10 (self-gating no-op until
    enough outcomes).
  - `04_predict_impact.py` — loads the learned model when present and blends:
    `duration = w·learned + (1−w)·estimator`; records `method = learned_estimator_blend_v1` with both
    components, the weight, and `n_outcomes`.
  - `tests/test_outcome_model.py` — covers blend-weight gating/ramp, training-frame filtering, and the
    insufficient-outcomes gate.
- **Why:** Makes the loop real (train + consume), not just collect/surface. Estimator stays as the
  transparent floor. See [DECISIONS.md](DECISIONS.md) D4 and [LEARNING_LOOP.md](LEARNING_LOOP.md).
- **Verified:** Seeded 150 synthetic outcomes → model trained (blend weight 0.75), stage 04 emitted
  `learned_estimator_blend_v1`; 0-outcome path is a clean no-op (pure estimator). Synthetic data removed
  after testing (gitignored). Full test suite: 5 passed.

### [Fixed/Added/Docs] Surface the post-event learning loop (R9)
- **Fixed:** `09_mlflow_logger.py` `retrain_if_needed` pointed at a non-existent input path
  (`ROOT.parent/cleaned_gridlock.csv` — same bug class as R1); corrected to `data/cleaned_gridlock.csv`
  so the retrain hook can actually run.
- **Added:** `_count_outcomes()` + a `logged_outcomes` MLflow metric (and fallback-log field) so the
  number of operator-logged ground-truth outcomes is visible over time. Verified: stage 09 logs
  "0 outcomes collected" on a fresh store.
- **Docs:** `docs/dev/LEARNING_LOOP.md` — honest map of the loop: what's wired (outcome capture incl.
  `actual_duration_min`, run tracking, retrain hook/cron) and the remaining gap (retrain doesn't yet
  consume `outcomes.jsonl`), with the build steps to close it.

### [Docs] Risk estimator reference & Bernoulli notes (R7, R8)
- **R7:** `docs/dev/RISK_ESTIMATOR.md` — formula, full factor reference, and a **verified** sensitivity
  table (computed by running `estimate_operational_impact`); framed as expert-set, uncalibrated defaults.
- **R8:** `docs/dev/BERNOULLI_NOTES.md` — current `k/alpha/beta` values flagged as uncalibrated defaults,
  with a concrete calibration path tied to logged outcomes.

### [Docs] Demo runbook (R6)
- Added `docs/dev/DEMO_RUNBOOK.md`: timed ~2.5-min script mapped to the 11 Flipkart-office screenshots,
  pre-flight checklist, headline talking points, anticipated Q&A, and a screenshot/HTML fallback path.

### [Changed] Efficiency cleanups on the 155k-node graph (R4)
- **Motivation:** Repeated O(N) work over the full Bengaluru graph (155,376 nodes) per event.
- **Change:**
  - `05_manpower_optimizer.py` — hoisted `set(intersections)` out of the per-node comprehension
    (was rebuilt 155k times) into a single `intersection_set`.
  - `07_diversion_routes.py` — build the `str->node` lookup once in `generate_routes` and pass it to
    `_local_pressure_edges` and `_fallback_tension_nodes` (was rebuilt 3×, once per function).
  - `04_predict_impact.py` — removed dead `_coerce_node` (defined, never called; full-node linear scan).
- **Verified:** Stages 04/05/07 produce identical outputs (350 edges, 150 intersections, 12 officers,
  2 manual + 8 Bernoulli routes); tests pass.

### [Changed] Align `runtime.txt` with tested interpreter (R5)
- **Problem:** `runtime.txt` pinned `python-3.10.14`, but the current dependency set was only verified on 3.13.
- **Change:** Pinned `python-3.13.13` (matches the validated `.venv`; supported by Streamlit Cloud and Render).
- **Note:** If a deploy target lacks 3.13, Python 3.11+ is a safe fallback.

### [Docs] Modeling approach & data limitation narrative (R3)
- **Motivation:** Biggest scoring risk — the "duration model" is decorative (R²≈0 on a proxy target).
  Owning this proactively reads as rigor; letting a judge discover it undermines trust.
- **Change:** Added a "Modeling Approach & Data Limitation" section to `README.md` (what's learned vs
  estimated, why, and the outcome→retrain loop). Also corrected the now-stale Quick Start (cross-platform
  venv activation, removed wrong `cd`, fixed data-path note to match R1). Recorded a rehearsed 30-second
  judge answer in [DECISIONS.md](DECISIONS.md) D1.

### [Fixed] XGBoost unseen-category crash (blocking)
- **Problem:** Stage 04 crashed with `Found a category not in the training set ... 'Outer Ring Road'`,
  cascading to stages 05–08. XGBoost 3.x raises on categories unseen during training (older versions
  treated them as missing); the prediction frame also rebuilt its own category dtype, misaligning codes.
- **Change:**
  - `scripts/03_train_duration_model.py` — capture `categorical_levels` for `corridor` and
    `event_cause` and store them in the saved model bundle.
  - `scripts/04_predict_impact.py` — reapply the trained categories via
    `pd.Categorical(values, categories=levels)` (unseen → NaN/missing); added `import pandas as pd`.
- **Why this approach:** More robust than pinning `xgboost<3`; keeps categorical codes aligned between
  train and predict. See [DECISIONS.md](DECISIONS.md) D2.
- **Verified:** Retrained, full pipeline 01→09 now completes; Streamlit boots (HTTP 200); tests pass.

### [Docs] Added developer tracking docs
- Created `docs/dev/PROJECT_STATUS.md`, `ROADMAP.md`, `CHANGELOG.md`, `DECISIONS.md` to track
  progress, plans, changes, and reasoning through submission.

### [Added] Local environment
- Created `.venv` (Python 3.13.13), installed all `requirements.txt` deps + `pytest`.
- Note: `pytest` not yet added to `requirements.txt` — tracked as [ROADMAP.md](ROADMAP.md) R2.

---

## Template for new entries

```
## YYYY-MM-DD

### [Fixed|Added|Changed|Docs] Short title  (R# / D#)
- Problem / motivation:
- Change:
- Verified:
```
