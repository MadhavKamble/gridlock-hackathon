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

## The loop is now closed ✅

`outcomes.jsonl` is consumed end-to-end (`lib/outcome_model.py`, `scripts/10_train_from_outcomes.py`):

1. **Build labeled set** — `build_training_frame` joins each outcome's `actual_duration_min` (the real
   target) with the stored event features (reusing `event_to_frame`, so features match the live path).
2. **Gated training** — `train_outcome_model` trains an `XGBRegressor` on that real duration target only
   when `n_outcomes >= MIN_OUTCOMES` (30); below that it's a no-op (avoids training on noise). A held-out
   split kicks in once `n >= 4×MIN_OUTCOMES`. The model + metrics are saved and logged to MLflow.
3. **Consumption with confidence weighting** — `04_predict_impact` loads the learned model when present
   and blends it with the estimator: `duration = w·learned + (1−w)·estimator`, where
   `w = blend_weight(n) = min(1, n / 200)`. So early on the transparent estimator dominates; as outcomes
   accumulate, the learned model takes over. The prediction records `method = learned_estimator_blend_v1`
   with both components, the weight, and `n_outcomes` for full transparency.
4. **Trigger** — `09_mlflow_logger.py --mode retrain-if-needed` runs stage 10 every time, so the model
   refreshes as outcomes grow (cron-friendly).

Verified by seeding 150 synthetic outcomes: model trained (blend weight 0.75), and `04` emitted
`learned_estimator_blend_v1` blending estimator and learned durations; with 0 outcomes the path is a
clean no-op (pure estimator). Synthetic data was removed after testing (`outcomes.jsonl` and the model
file are gitignored).

### Remaining refinements (genuine future work)
- Recalibrate the operational risk estimator factors against the same outcomes (see [RISK_ESTIMATOR.md](RISK_ESTIMATOR.md)).
- Recalibrate the Bernoulli `k/alpha/beta` against observed congestion relief (see [BERNOULLI_NOTES.md](BERNOULLI_NOTES.md)).
- Per-corridor models once data volume per corridor is sufficient.

## Recent fix

`retrain_if_needed` previously pointed at a non-existent input path (`../cleaned_gridlock.csv`, the same
class of bug fixed for `run_all.sh` in R1); corrected to `data/cleaned_gridlock.csv` so the retrain hook
actually runs. See [CHANGELOG.md](CHANGELOG.md) R9.

## Demo framing

> "Operators log what actually happened — including the real duration the raw data never recorded. Once
> enough outcomes accumulate, the system trains a duration model on that real target and blends it into
> predictions, trusting it more as evidence grows — with the transparent estimator as the floor. That's a
> live post-event learning system: it starts honest with rules and gets sharper with every logged event."
