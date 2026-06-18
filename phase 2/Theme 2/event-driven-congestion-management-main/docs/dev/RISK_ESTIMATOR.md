# Operational Risk Estimator — Reference & Sensitivity

How `predicted_duration_min` is produced, and how each input moves it. This is the transparent
alternative to a supervised duration model (see [DECISIONS.md](DECISIONS.md) D1). Source:
`estimate_operational_impact` in `scripts/04_predict_impact.py`.

---

## Formula

```
estimate = base_cause_minutes
         × priority_factor
         × closure_factor
         × peak_hour_factor
         × planned_event_factor
         × corridor_factor
         × road_context_factor
estimate = clamp(estimate, 10, 480)        # minutes
range    = [estimate × 0.70, estimate × 1.35]
```

Every multiplier is an explicit, inspectable constant — an operator can see exactly why a number
came out the way it did and challenge any single factor.

## Factor reference

| Factor | Values |
|--------|--------|
| **Base cause (min)** | vehicle_breakdown 35 · pot_holes 45 · congestion 50 · others 55 · water_logging 70 · accident 75 · tree_fall 95 · construction 120 · protest 135 · procession 150 · public_event 180 |
| **Priority** | low 0.75 · medium 1.00 · high 1.35 · critical 1.75 |
| **Closure** | none 1.00 · requires closure 1.40 |
| **Peak hour** (08–10, 17–20) | off-peak 1.00 · peak 1.25 |
| **Planned event** | unplanned 1.00 · planned 1.15 |
| **Corridor** | non-corridor/unknown 1.00 · named corridor 1.12 |
| **Road context** | from the road-context classifier: terminal-local 0.42 · local-access 0.62 · mixed 0.80 · through-road 1.00 |

## Sensitivity (verified by running the estimator)

Baseline: *unplanned accident, Medium priority, no closure, 14:00 (off-peak), non-corridor* → **75.0 min**.
One factor varied at a time from that baseline:

| Dimension | Variation → minutes |
|-----------|---------------------|
| Cause | vehicle_breakdown **35** · accident **75** · construction **120** · protest **135** · procession **150** · public_event **180** |
| Priority | Low **56.2** · Medium **75** · High **101.2** · Critical **131.2** |
| Closure | no **75** · yes **105** |
| Hour | 03:00 **75** · 14:00 **75** · 18:00 (peak) **93.8** |
| **Worst case** (public_event + Critical + closure + 18:00) | **480** (hits the cap) |

Takeaways: cause and priority dominate; closure (+40%) and peak hour (+25%) are meaningful modifiers;
the 10–480 min clamp keeps extreme combinations operationally sane.

## Honest framing

These multipliers are **expert-set defaults, not fitted parameters** — chosen to be reasonable and
transparent, not calibrated against ground-truth durations (which the dataset lacks). The logged-outcome
loop ([CHANGELOG.md](CHANGELOG.md), R9) is designed to collect the real durations needed to either
recalibrate these factors or replace the estimator with a trained model. See [ROADMAP.md](ROADMAP.md) R7.
