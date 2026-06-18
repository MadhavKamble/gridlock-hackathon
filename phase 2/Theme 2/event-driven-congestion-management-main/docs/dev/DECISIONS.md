# Decision Log

Key technical and strategic decisions, with reasoning and trade-offs. ADR-lite: each decision
records the context, the choice, why, and consequences. Newest decisions appended at the bottom.

---

## D1. Use a transparent rule-based estimator for impact duration, not a supervised ML model

- **Status:** Accepted (inherited from original design; reaffirmed)
- **Context:** The source dataset (`cleaned_gridlock.csv`, 8,173 events) contains `start_datetime`
  and `created_date` but **no event-end / resolution timestamp**. There is no ground-truth label for
  "how long did the congestion last," so a supervised duration model cannot be trained honestly.
- **Decision:** Predict operational impact duration with a transparent, documented rule-based
  estimator (`estimate_operational_impact`) combining cause base-minutes, priority, closure, peak-hour,
  planned/unplanned, corridor, and road-context factors. Keep an XGBoost model only for what the data
  *can* support (`report_creation_delay_min`), clearly labeled as such.
- **Why:**
  - Honesty — fabricating a duration label would produce a meaningless model (current R² ≈ 0 confirms
    there's no signal for the proxy target either).
  - Transparency — operators and judges can see and challenge every factor.
  - Extensibility — the outcome-logging loop is designed to collect real durations so a genuine model
    can replace the estimator later.
- **Consequences / risk:** Must be framed proactively (see [ROADMAP.md](ROADMAP.md) R3). Unmanaged,
  a judge may read "XGBoost duration prediction" and assume more than the data supports. Managed, it
  reads as rigor + a clear data-collection roadmap.
- **Rehearsed 30-second answer (if asked "what's your model's accuracy?"):**
  > "The dataset has no event-end timestamp, so there's no ground-truth duration to train on — a
  > supervised duration model would mean inventing a label. So duration comes from a transparent
  > operational risk estimator where every factor is inspectable. We do train an XGBoost model on the
  > one honest target the data supports — reporting delay — and we log real outcomes through the UI so
  > a true duration model can be trained once that data accumulates. That outcome loop is exactly the
  > 'post-event learning system' the problem statement says is missing."
- **README:** documented in the "Modeling Approach & Data Limitation" section.

## D2. Fix XGBoost categorical handling in code, not by pinning the library

- **Status:** Accepted (2026-06-18)
- **Context:** XGBoost 3.x raises on categories unseen during training; `requirements.txt` allows
  `xgboost>=2,<4`, so 3.x installs. Stage 04 crashed on `'Outer Ring Road'`.
- **Decision:** Persist training `categorical_levels` in the model bundle and reapply them at predict
  time (`pd.Categorical(values, categories=levels)`), mapping unseen values to NaN/missing.
- **Why:** Pinning `xgboost<3` would mask the real issue (train/predict category-code misalignment)
  and freeze us on an older library. The code fix is version-robust and correct in intent.
- **Consequences:** Model bundle format changed (added `categorical_levels`); models must be retrained
  (already done). No API change for callers.

## D3. Documentation lives in `docs/dev/`, separate from user/demo assets

- **Status:** Accepted (2026-06-18)
- **Context:** `docs/assets/` already holds the Flipkart-office demo screenshots used in the README.
- **Decision:** Keep internal progress-tracking docs (status, roadmap, changelog, decisions) under
  `docs/dev/` so they're discoverable but don't clutter the user-facing README/asset story.
- **Why:** Separates "how we built it / what's next" (for the team and judges who dig in) from "what it
  does" (the polished narrative). Easy to reference from PROJECT_STATUS.
- **Consequences:** One place to look for project state; the public README stays focused on the demo.

## D4. Close the learning loop with a confidence-weighted blend, estimator as floor

- **Status:** Accepted (2026-06-18)
- **Context:** `outcomes.jsonl` captures `actual_duration_min` — the real duration label the source data
  lacks (D1). We chose to fully wire it in (train + consume), not just collect.
- **Decision:** Train an XGBoost duration model from outcomes, gated at `MIN_OUTCOMES=30`; consume it in
  `04_predict_impact` as `duration = w·learned + (1−w)·estimator` with `w = min(1, n/200)`.
- **Why:**
  - A hard switch from estimator → model at an arbitrary count would be brittle and dishonest with few
    samples. Linear confidence weighting degrades gracefully: rules dominate early, the model takes over
    as evidence grows.
  - Keeps the transparent estimator as a floor/fallback, so the system never depends on a thinly-trained
    model. The prediction records both components + the weight for full auditability.
- **Consequences:** New `lib/outcome_model.py` + `scripts/10_train_from_outcomes.py`; stage 09 triggers
  stage 10; stage 04 gained an optional dependency on the outcome model (no-op when absent). Verified with
  150 synthetic outcomes (blend weight 0.75) and the 0-outcome no-op path; unit tests added.

## D5. Scope discipline against the problem statement (keep reporting model, bury Bernoulli)

- **Status:** Accepted (2026-06-18)
- **Context:** Audited every component against the problem statement's one cause — *forecast event impact;
  recommend manpower, barricading, diversion; close the post-event learning gap*. Two components don't map
  to the cause: (a) the reporting-delay model (stage 03 / `predicted_reporting_delay_min`, R²≈0, used in no
  decision), and (b) the Bernoulli fluid-dynamics diversion (uncalibrated, additive on top of the direct
  closure-bypass routes that already satisfy the diversion ask).
- **Decision:**
  - **Reporting-delay model — keep as-is.** Owner chose not to refactor it out this close to submission.
    It stays documented honestly (D1, README "Modeling Approach"); it is **not** foregrounded in the pitch.
  - **Bernoulli diversion — keep but bury.** Code stays (already off by default); removed from the demo
    script's core beat and headline talking points; surfaced only if a judge asks, clearly labeled
    experimental/optional. The recommended diversion is the direct closure-bypass route.
- **Why:** Stay on-cause. The actionable, defensible deliverables lead; experimental/irrelevant pieces
  neither get cut at risk near submission nor get promoted into questions we can't cleanly defend.
- **Consequences:** No code change. `DEMO_RUNBOOK.md` updated to de-feature Bernoulli; this is the
  authoritative pitch framing. README already labels Bernoulli experimental.

---

## Template

```
## D#. Title

- **Status:** Proposed | Accepted | Superseded by D#
- **Context:**
- **Decision:**
- **Why:**
- **Consequences / risk:**
```
