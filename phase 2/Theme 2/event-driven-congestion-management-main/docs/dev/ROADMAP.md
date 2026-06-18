# Roadmap — Submission Prep

Last updated: 2026-06-18 · Target: submission in ~2 days

> Prioritized plan with reasoning. Each item states **why** it matters and the **acceptance check**
> that tells us it's done. Status legend: ☐ todo · ◐ in progress · ☑ done.

---

## P0 — Must fix before submission (correctness & credibility)

### ☑ R1. Fix `run_all.sh` data path  — done 2026-06-18
- **What:** Change `--input ../cleaned_gridlock.csv` → `--input data/cleaned_gridlock.csv`.
- **Why:** The README Quick Start is the first thing a judge runs after cloning. Today it fails at
  stage 01 because the parent path doesn't exist. Highest-leverage, lowest-effort credibility fix.
- **Done when:** A clean clone + `bash run_all.sh` completes all 9 stages without manual edits.

### ☑ R2. Add `pytest` to `requirements.txt`  — done 2026-06-18
- **What:** Declare the test dependency.
- **Why:** Reproducibility — judges who run tests shouldn't hit `No module named pytest`.
- **Done when:** Fresh `pip install -r requirements.txt && pytest` passes.

### ☑ R3. Prepare the "why no ML duration model" narrative  — done 2026-06-18
- **What:** A short, rehearsed answer + one slide/README paragraph owning limitation L1.
- **Why:** This is our single biggest scoring risk. If a judge discovers the model is decorative,
  it undermines trust in everything. If *we* frame it first — "the dataset has no event-end
  timestamp, so a supervised duration model isn't possible; we built a transparent operational
  risk estimator and a learning loop to collect ground truth over time" — it becomes evidence of
  rigor. See [DECISIONS.md](DECISIONS.md) D1.
- **Done when:** README has an explicit "Modeling approach & data limitation" section and we can
  answer the question in <30s.

---

## P1 — Strengthens the submission (do if P0 done)

### ☑ R4. Efficiency cleanups on the 155k-node graph  — done 2026-06-18
- **What:** Hoist `set(intersections)` out of the per-node loop (`05:31`); build `node_lookup` once
  and pass it through `07`; delete or justify unused `_coerce_node` (`04:228`).
- **Why:** Faster, cleaner code; reduces per-event latency in the live demo. Low risk.
- **Done when:** Pipeline runtime unchanged or better, all stages still produce identical artifacts.

### ☑ R5. Align `runtime.txt` with the tested interpreter  — done 2026-06-18
- **What:** Decide on 3.10 vs 3.13 and make `runtime.txt` + local venv consistent.
- **Why:** Avoids a deploy-time surprise (Streamlit Cloud / Render read `runtime.txt`).
- **Done when:** Documented target version; deploy config matches what we tested.

### ☑ R6. Demo script & assets  — done 2026-06-18 (see DEMO_RUNBOOK.md)
- **What:** Rehearse the Flipkart-office walkthrough using the 11 existing screenshots; scripted
  2–3 min flow: enter event → live monitor → plain-English summary → map layers.
- **Why:** Judges remember a concrete story over a feature list. Assets already exist; just sequence them.
- **Done when:** A written demo runbook exists and a dry run fits the time limit.

---

## P2 — Nice to have (only if time remains)

### ☑ R7. Quantify the operational risk estimator  — done 2026-06-18 (RISK_ESTIMATOR.md)
- **What:** Sensitivity table showing how duration/radius/staffing respond to cause, priority,
  closure, peak-hour, and road context.
- **Why:** Turns "hand-tuned rules" into "transparent, defensible policy." Strengthens R3.

### ☑ R8. Calibrate / document the Bernoulli heuristic  — done 2026-06-18 (BERNOULLI_NOTES.md)
- **What:** Note that `k, alpha, beta` are uncalibrated; show the pressure-field on the demo map as
  a differentiator and state the calibration path.
- **Why:** It's our most memorable/novel idea — present it as deliberate, experimental, with a path forward.

### ☑ R9. Surface the learning loop  — done 2026-06-18 (LEARNING_LOOP.md)
- **What:** Show that logged outcomes (`data/outcomes.jsonl`) feed the MLflow retrain hook.
- **Why:** Directly answers the problem statement's "no post-event learning system" gap.

---

## Sequencing

1. **Day 1:** R1, R2, R3 (P0 — correctness + narrative). Commit.
2. **Day 1–2:** R4, R6 (cleanups + demo prep).
3. **Day 2:** R5, then R7/R8/R9 as time allows. Final dry run.

Update item status here and log every code change in [CHANGELOG.md](CHANGELOG.md).
