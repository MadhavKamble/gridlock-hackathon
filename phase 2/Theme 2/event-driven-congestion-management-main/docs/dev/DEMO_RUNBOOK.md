# Demo Runbook

A scripted ~2.5-minute walkthrough for judges, plus pre-flight, fallback, and Q&A.
Scenario: **unplanned critical accident near the Flipkart office (Gear School Rd / Bhoganahalli), 18:30 (evening peak).**

> Screenshots backing every beat live in `docs/assets/flipkart-office/01..11`. If the live app fails,
> narrate from those images — the story is identical.

---

## 0. Pre-flight (do this before judges arrive)

```bash
cd event-driven-congestion-management-main
. .venv/bin/activate          # Windows: . .venv/Scripts/activate
# Artifacts already built? skip run_all.sh. Otherwise:
# bash run_all.sh
streamlit run app/main.py
```

Checklist:
- [ ] App loads at `http://localhost:8501` (open it in advance, zoom the map to Bengaluru).
- [ ] `output/dashboards/dashboard.html` exists (backup artifact to show offline).
- [ ] Optional: live monitor running — `python app/realtime_monitor.py 8765` → `http://127.0.0.1:8765`.
- [ ] Screenshots folder open in a tab as the fallback.
- [ ] One sentence ready: *"This turns a single reported event into a complete manpower + barricade + diversion plan, and learns from outcomes."*

---

## 1. The script (timed)

**[0:00–0:20] Frame the problem.**
> "Today, when a rally, accident, or festival hits Bengaluru, impact isn't quantified in advance,
> deployment is by experience, and nothing is learned afterward. We built a system that takes one
> event and produces a full operational response — and closes the learning loop."

**[0:20–0:45] Enter the event.** → `01-event-entry-form.png`
- Click the map at the Flipkart office area; set cause = accident, priority = Critical, time = 18:30, unplanned.
- Point out the **peak-hour warning** the UI raises.
> "I drop a pin, set a critical accident at evening peak. Notice it already warns that peak hour will
> raise both impact and staffing pressure."

**[0:45–1:05] Generate & show the pipeline.** → `02-live-pipeline-monitor.png`
- Hit **Generate response plan**; show the live monitor streaming each stage.
> "Behind one click: impact prediction on the real OSM road network, OR-Tools officer optimization,
> barricade simulation, and diversion routing — each stage live with its artifacts."

**[1:05–1:35] Plain-English decision.** → `03-response-summary.png`
> "It doesn't dump numbers — it gives an operator a plain-English plan: estimated impact, affected
> segments, officers assigned vs held in reserve, the recommended barricade, and which diversions are
> actionable vs experimental."

**[1:35–2:10] Walk the map layers.** → `08`, `07`, `05`, `11`
- Affected roads (`08`) → police deployment (numbered, `07`) → recommended barricade (`05`) →
  direct diversion route (`11`).
> "Red = directly affected roads. Blue numbers = officers per junction from the optimizer. Dashed =
> barricaded segments. The solid blue route is the actionable bypass around the closure."

> **Do not lead with the Bernoulli pressure layer.** It's an optional, experimental planning view (off by
> default) and is not part of the core plan. Only show it if a judge specifically asks about additional
> diversion analysis — see the Q&A.

**[2:10–2:30] The learning loop + honesty close.** → mention `outcomes.jsonl` / MLflow
> "Operators log the actual outcome, which feeds an MLflow retrain hook — the post-event learning system
> the brief says is missing today. And to be upfront: the dataset has no event-end timestamp, so duration
> is a transparent risk estimator, not a black box — and the outcome loop is exactly how we'd train a real
> duration model over time."

---

## 2. Headline talking points (if time is cut short)

1. **Real network, real data** — live Bengaluru OSM graph (155k nodes), 8,173 events.
2. **All three asks delivered** — manpower (OR-Tools CP-SAT), barricades (simulated plans), diversions (closure bypass).
3. **Closes the learning gap** — outcomes (real durations) train a duration model that sharpens the
   forecast over time (stage 10); estimator is the transparent floor.
4. **Operator-first** — plain-English summary, not a metrics dump.
5. **Honest modeling** — see the Q&A below; we own the data limitation.

---

## 3. Anticipated Q&A

**Q: What's your model's accuracy / how well does it predict duration?**
> See [DECISIONS.md](DECISIONS.md) D1 — the rehearsed 30-second answer. Short version: no event-end
> timestamp exists, so duration is a transparent estimator; we train only the honest target (reporting
> delay) and log outcomes to enable a real model later.

**Q: Is there other diversion analysis / what's that optional pressure layer?** *(only if asked)*
> There's an optional, experimental pressure-field view (off by default), not part of the core plan. It's a
> deliberately-labeled heuristic with uncalibrated parameters and a clear calibration path against logged
> outcomes. The recommended, actionable diversion is the direct closure-bypass route.

**Q: Does it use real-time data?**
> The pipeline is built to accept a live event and recompute in seconds; today's inputs are the event
> report + static OSM network. Real-time feeds (speed/incident APIs) are a drop-in extension.

**Q: Will it scale beyond Bengaluru?**
> The network is fetched by place name via OSMnx — point it at any city. Everything downstream is graph-generic.

---

## 4. Fallback if the live app breaks

1. Open `output/dashboards/dashboard.html` directly (no server needed) — shows the full layered map.
2. If that's missing, narrate the 11 screenshots in `docs/assets/flipkart-office/` in order — they mirror
   this exact scenario beat-for-beat.
3. Worst case, run `bash run_all.sh` once beforehand so artifacts are guaranteed present.
