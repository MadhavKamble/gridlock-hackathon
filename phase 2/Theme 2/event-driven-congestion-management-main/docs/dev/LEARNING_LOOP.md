# Post-Event Learning Loop

The problem statement explicitly calls out the absence of a *post-event learning system*. This is how
ours works, what is wired today, and the one connection that still needs building — stated honestly.

---

## The loop

```
 Operator logs outcome (UI)                      data/outcomes.jsonl
   actual_duration_min, status,        ──────►    (append-only ground truth:
   full prediction snapshot                        the event-END info the raw
                                                    dataset never had)
        │
        ▼
 09_mlflow_logger.py --mode log-latest    ──────► MLflow run + `logged_outcomes` metric
        │                                          (loop progress is now visible)
        ▼
 09 --mode retrain-if-needed --threshold-mae N ──► re-prepares data + retrains model
```

## What is wired today ✅

- **Outcome capture.** The results screen in `app/main.py` (`append_outcome`) writes each operator-logged
  outcome — including **`actual_duration_min`** — to `data/outcomes.jsonl`. This is significant: it
  captures the *event-end / true-duration* signal the source dataset (`cleaned_gridlock.csv`) lacks
  (see [DECISIONS.md](DECISIONS.md) D1). Every logged outcome is a future training label.
- **Run tracking.** `09_mlflow_logger.py --mode log-latest` records each pipeline run to an MLflow SQLite
  store and now logs a **`logged_outcomes`** metric, so loop progress (how much ground truth we've
  collected) is visible over time.
- **Automated retrain hook + scheduling.** `--mode retrain-if-needed` retrains when MAE exceeds a
  threshold; the README documents cron entries for periodic logging/retraining.

## The honest gap ⚠️ (next build step)

`retrain_if_needed` currently re-trains on the **static** `cleaned_gridlock.csv` (the reporting-delay
target), driven by an MAE threshold — it does **not yet consume `outcomes.jsonl`**. So today the loop
*collects* ground truth and *surfaces* progress, but the retrain step doesn't yet *learn from* the
collected durations. Closing that wire is the path from "reporting-delay model" to a real
"congestion-duration model":

1. Join `outcomes.jsonl` (`actual_duration_min` + event features from the stored prediction) into a
   labeled training set.
2. Once enough labeled events accumulate, train a duration model on that real target and register it in MLflow.
3. Either replace the operational risk estimator or use it as a prior, and recalibrate the
   Bernoulli parameters against observed relief (see [BERNOULLI_NOTES.md](BERNOULLI_NOTES.md)).

## Recent fix

`retrain_if_needed` previously pointed at a non-existent input path (`../cleaned_gridlock.csv`, the same
class of bug fixed for `run_all.sh` in R1); corrected to `data/cleaned_gridlock.csv` so the retrain hook
actually runs. See [CHANGELOG.md](CHANGELOG.md) R9.

## Demo framing

> "Operators log what actually happened — including the real duration the raw data never recorded. That
> feeds MLflow, where you can watch the labeled-outcome count grow. The retrain hook and schedule are in
> place; wiring those collected durations into a trained duration model is the deliberate next step, and
> it's exactly the post-event learning system the brief says is missing."
