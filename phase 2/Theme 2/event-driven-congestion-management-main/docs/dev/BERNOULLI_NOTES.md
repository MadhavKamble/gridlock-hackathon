# Bernoulli Pressure-Field Diversion — Notes & Calibration Path

Our differentiating diversion heuristic. The user-facing explanation (formula, map layers) is in the
README "Bernoulli-Tension Diversion" section; this note is the engineering honesty + roadmap companion.
Source: `lib/bernoulli_pressure.py`, used by `scripts/07_diversion_routes.py`.

---

## The idea (one line)

Treat each road edge as a fluid-flow channel: slow, high-delay links carry high "pressure," and we route
diversions from high-tension nodes (where a high-pressure edge meets a low-pressure neighbour) toward
major exit nodes, minimizing a Bernoulli-style potential `E` via Dijkstra.

```
density        = (free_flow_speed × lanes) / predicted_speed
pressure  P    = alpha · min(density / capacity_norm, 1) + beta · delay_penalty
potential E    = 0.5 · k · speed_loss² + P        # edge weight for routing
```

## Parameters (current values)

| Param | Value | Controls | Status |
|-------|-------|----------|--------|
| `k` | 0.02 | kinetic term — penalty on lost speed² | **uncalibrated default** |
| `alpha` | 0.70 | weight on density pressure | **uncalibrated default** |
| `beta` | 0.30 | weight on delay penalty | **uncalibrated default** |
| `pressure_threshold` | 0.60 (adaptive) | high-tension node cutoff; auto-lowered when no edge clears 0.60 | heuristic |
| `capacity_norm` | 85th-percentile of `speed×lanes` over the graph | normalizer | data-derived |

## Honest status

These are **deliberately-chosen, uncalibrated defaults** — a qualitatively sensible analogy, **not** a
validated microscopic traffic-flow model (SUMO/CityFlow do that; we fall back to this NetworkX engine when
they're unavailable). We present it as an experimental planning layer:
- Pressure-release routes are shown **dashed** and labeled as *candidates*, explicitly **not** blanket
  driver instructions — this is reinforced in the UI and the plain-English summary.
- Direct closure-bypass routes (solid) are the operationally actionable output; Bernoulli routes augment them.

## Calibration path (how it stops being a guess)

1. Log real outcomes via the UI → `data/outcomes.jsonl` (already wired; see R9).
2. Once enough events accumulate, compare Bernoulli-suggested routes against observed congestion relief.
3. Fit `k, alpha, beta` (e.g. grid/Bayesian search) to maximize agreement with observed relief;
   recompute `capacity_norm` per-corridor rather than graph-wide.
4. Promote from "experimental layer" to "recommended" only after that validation.

See [ROADMAP.md](ROADMAP.md) R8 and [DECISIONS.md](DECISIONS.md) D1 for the broader honesty stance.
