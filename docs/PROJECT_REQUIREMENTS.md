# PROJECT REQUIREMENTS — GeoStrom AI

**Phase:** 0 (Architecture) · **Status:** Complete · **Audience:** whole team

---

## 1. Problem Statement

> "To develop an Artificial Intelligence (AI) / Machine Learning (ML) based system for
> identification, classification, and prediction of different tropical cyclone patterns using
> multi-source satellite data."

### 1.1 What this actually asks for

The statement bundles **four distinct ML problems** with different input modalities, different
label sources, and different evaluation regimes. Treating them as one model would be a mistake.
Decomposed:

| # | Problem | ML type | Primary input | Primary label source |
|---|---|---|---|---|
| A | Identification | Binary image classification | Satellite IR scene | Constructed (see §2.A) |
| B | Pattern/stage classification | Multi-class image classification (ordinal) | Storm-centric IR image | ADT scene type *or* IBTrACS-derived category |
| C | Intensity prediction | Multivariate multi-step time-series regression | Best-track sequence (+ image features) | IBTrACS wind / pressure |
| D | Track prediction | Multi-step geospatial regression | Best-track sequence (+ image features) | IBTrACS lat / lon |
| E | Geospatial visualisation | Not ML — presentation layer | Outputs of A–D | — |

**The critical reading:** "multi-source" is the hard part, not the models. The scientific
contribution and the majority of engineering effort is **fusing storm-centric satellite imagery
with best-track records into a leak-free, temporally-aligned training set.** Problems A–D are
well-trodden in the literature; the fusion is where this project succeeds or fails.

### 1.2 System-level objective

Given a historical tropical cyclone and a point in its lifetime, GeoStrom AI must:

1. Confirm from imagery that an organised TC is present, with a calibrated confidence.
2. Describe its current structural pattern / intensity stage.
3. Forecast its intensity for the next 24 hours.
4. Forecast its centre position for the next 24 hours, with an honest uncertainty envelope.
5. Present all of the above on an interactive map and timeline.
6. Explain the result in natural language, grounded strictly in the values produced above.

### 1.3 Operating mode — a locked decision

**DECISION: GeoStrom AI is a *retrospective replay* system, not a live operational system.**

The user selects a historical storm and a time index; the system shows what the models predicted at
that moment and what actually happened. Rationale:

- Live TC feeds (real-time ATCF / agency advisories) introduce operational data plumbing, uptime
  requirements, and a duty of care that a hackathon prototype cannot honour.
- Retrospective mode gives **ground truth for every prediction**, which makes the dashboard far more
  compelling: predicted track vs actual track, side by side, with error in km.
- It permits **offline batch inference**, which collapses the entire latency, GPU-hosting, and
  scaling risk surface at once (see SYSTEM_ARCHITECTURE.md §5).

A live-ingest adapter is *designed for* but not built — see Advanced scope.

---

## 2. Capability Requirements

### A. Cyclone Identification

**Goal:** determine whether an organised tropical cyclone is present in a satellite scene, with a
confidence score.

**The central problem — there is no negative class.**
HURSAT-B1 is *storm-centric by construction*: every frame is re-gridded around a known TC centre.
A detector trained only on HURSAT sees 100% positives and learns nothing. **This must be resolved
before detection is implementable.** It is the single most important finding of this analysis.

Three candidate strategies, in preference order:

| Path | Negative source | Cost | Scientific validity |
|---|---|---|---|
| **A (preferred)** | GridSat-B1 tiles sampled at times/locations with no IBTrACS storm within a radius (e.g. 1000 km) | One extra dataset; needs careful sampling | Strong — true "no TC here" scenes, same instrument lineage as HURSAT |
| **B (fallback)** | HURSAT frames whose IBTrACS `NATURE` is non-tropical (extratropical / disturbance / not-reported) plus pre-genesis and post-dissipation frames | Zero extra download | Weaker — negatives still contain *a* vortex, so the task becomes "organised tropical system vs non-tropical system". A legitimate problem, but a different one. |
| C (rejected) | Off-centre crops of HURSAT frames | Zero | Invalid — a 301×301 storm-centric grid at roughly 8 km/px spans on the order of 2400 km; off-centre crops still contain the storm's cirrus canopy |

**DECISION:** implement **Path B first** (unblocked, zero download cost), and upgrade to Path A if
the GridSat sampling pipeline fits the schedule. Path B's honest framing —
*"organised tropical system vs non-tropical / disorganised system"* — is stated explicitly in the UI
and on the methodology page. We do not claim general "cyclone detection in arbitrary imagery" unless
Path A is delivered.

- **Input:** single-channel IR brightness-temperature image resampled to a fixed grid.
- **Output:** `P(tropical cyclone present)` in `[0,1]`, plus a calibrated confidence.
- **MVP:** Yes, scoped as above and clearly labelled.

### B. Cyclone Pattern / Stage Classification

**Goal:** assign each storm-centric frame to a structural / intensity class.

**No classes are hard-coded in Phase 0.** The label set is chosen only after a label-analysis
notebook inspects the real data. Three candidate label tiers:

| Tier | Label set | Availability | Notes |
|---|---|---|---|
| **A — Intensity category** | Saffir–Simpson-style stage derived from IBTrACS wind (TD, TS, Cat 1–5) | **Guaranteed** — derivable from wind speed | Ordinal, objective, well-precedented. A CNN mapping IR imagery to intensity category is effectively learning the Dvorak relationship, which is a published, defensible approach. |
| **B — Dvorak / ADT scene type** | Eye, Embedded Centre, Central Dense Overcast, Irregular CDO, Curved Band, Shear, Uniform | **TO VERIFY** — depends on whether the ADT-HURSAT release exposes an ADT scene-type field | This is the *true* "pattern" classification and matches the problem statement's wording most directly. Highest scientific value **if present**. |
| **C — Lifecycle stage** | Developing / Intensifying / Mature / Weakening | Derivable, but **engineered, not ground truth** | Computed from the sign of dV/dt and V relative to maximum-so-far. **Risk of circularity:** the model learns our heuristic, not nature. |

**DECISION:**
- **Primary target = Tier A** — guaranteed to exist, objective, ordinal.
- **Promote Tier B to the headline "pattern" model if ADT scene types are confirmed present.**
- **Tier C is a derived UI annotation only**, computed deterministically from the intensity curve and
  explicitly marked as heuristic. It is never presented as a model output or a scientific finding.

> The example labels in the brief (*Developing / Organized / Mature / Weakening*) map to Tier C.
> They are **not dataset labels.** Treating them as ground truth without saying so would misrepresent
> the science. This is flagged because the brief explicitly asked us not to invent unsupported labels.

- **Input:** storm-centric IR image, optionally concatenated with the scalar state vector.
- **Output:** class probability distribution; for ordinal targets, also an expected-category value.
- **MVP:** Yes (Tier A).

### C. Cyclone Intensity Prediction

**Goal:** forecast intensity at future lead times from observed history.

**Targets** (TO VERIFY against actual column population rates):

- **`wind`** — maximum sustained wind. **Primary target.** Best-populated intensity field.
- **`pressure`** — minimum central pressure. **Secondary.** Sparser and more agency-dependent.
- **`category`** — *derived from the predicted wind*, not modelled separately. Modelling it
  independently would allow the system to emit a predicted wind and a predicted category that
  contradict each other.

**Horizons:** +6, +12, +18, +24 h, produced as a direct multi-output head. Justification in
ML_ARCHITECTURE.md §6.3.

**Honest expectation:** the dominant physical predictors of intensity change — sea-surface
temperature, ocean heat content, and vertical wind shear — are **absent** from IBTrACS and HURSAT.
Without them, intensity skill will be modest. **The success criterion is beating persistence, not
beating an operational agency forecast.** Any other claim would misrepresent what the available data
supports. Adding reanalysis-derived environmental fields is the top Advanced-scope upgrade.

- **MVP:** Yes — wind only, 24 h horizon.

### D. Cyclone Track Prediction

**Goal:** forecast centre position at future lead times.

- **Output representation:** predict **displacements (Δlat, Δlon) per horizon**, not absolute
  coordinates. Displacements are near-stationary and roughly zero-mean, the model does not have to
  memorise basin geography, and the ±180° discontinuity in absolute longitude is avoided.
- **Uncertainty:** an empirical error radius per lead time, taken from validation-set error
  quantiles. This is conceptually how a forecast cone is constructed, and it is honest, cheap, and
  directly renderable without any change to the model.
- **Error metric:** great-circle (Haversine) distance in km, additionally decomposed into
  along-track and cross-track components — the decomposition distinguishes *"right speed, wrong
  direction"* from *"right direction, wrong speed"*, which a single distance number hides.
- **Baselines are mandatory before any deep model** — see §4.2.
- **MVP:** Yes — 24 h horizon.

### E. Geospatial Visualisation

Must render, on one interactive map:

1. **Historical track** — observed positions up to the selected time (intensity-coloured polyline).
2. **Current location** — highlighted marker with the current state readout.
3. **Predicted track** — model output from the selected time forward, visually distinct.
4. **Actual future track** — ground truth, for direct comparison. This is the payoff of retrospective mode.
5. **Prediction points** — discrete markers at +6 / +12 / +18 / +24 h.
6. **Uncertainty** — a widening cone/ellipse envelope built from the empirical error radii.

Supporting views: intensity timeline chart, the IR image for the selected timestep, classification
output, and the grounded Gemini narrative.

---

## 3. Non-Functional Requirements

| Requirement | Target | Notes |
|---|---|---|
| API p95 latency | < 300 ms on cached reads | Achievable because inference is precomputed |
| Gemini endpoint latency | < 6 s, streamed | Streaming keeps perceived latency low |
| Frontend LCP | < 2.5 s | Directly constrains 3D usage — see UI_UX_ARCHITECTURE.md §3 |
| Map interaction | 60 fps pan/zoom | Requires server-side track simplification |
| Training run | ≤ 6 h per model within 6 GB VRAM | Binding constraint on architecture choice |
| Reproducibility | Fixed seeds, pinned deps, versioned split manifests | Split manifests committed, never regenerated ad hoc |
| Model swappability | Replace any model without touching API or frontend | Enforced by the `ModelRegistry` interface |
| Accessibility | WCAG AA contrast; `prefers-reduced-motion` respected | Non-negotiable despite the heavy-motion design direction |

---

## 4. Methodological Requirements (non-negotiable)

### 4.1 Leakage control

Consecutive 3-hourly frames of the same storm are near-duplicates. A random row-level train/test
split would produce spectacular and completely meaningless scores. **Mandatory rules:**

1. **Split by storm ID, never by row or frame.**
2. **Prefer a temporal split by season** (train ≤ year *X*; validate *X+1..X+2*; test ≥ *X+3*).
   This tests generalisation to future storms, which is the actual deployment condition.
3. **Fit all scalers and encoders on the training split only.**
4. **All engineered features must be causal.** "Lifetime maximum wind" computed over the whole storm
   leaks the future into the past. Use *maximum-so-far* only. This is subtle and easy to get wrong.
5. **Split manifests are written to disk and version-controlled**, so every model is compared on an
   identical partition.

### 4.2 Baselines before deep learning

No deep model may be reported without its baseline on the same split:

| Task | Mandatory baseline |
|---|---|
| Detection | Majority class, plus logistic regression on simple IR statistics |
| Classification | Class-prior, plus gradient boosting on scalar IR features |
| Intensity | **Persistence** (wind held constant), plus a GBM on the state vector |
| Track | **Persistence** (constant-velocity extrapolation), plus a **CLIPER-style** regression (climatology + persistence) |

Persistence and CLIPER-style regressions are the standard meteorological reference forecasts.
**Skill is reported relative to them.** A deep model that fails to beat persistence is reported as
having failed to beat persistence.

### 4.3 Honesty requirements

- Every metric is reported together with the split it was measured on.
- Uncertainty is shown wherever a prediction is shown.
- Derived/heuristic labels are marked as heuristic in the UI.
- A persistent disclaimer: **not for operational, navigational, or emergency use.**

---

## 5. MVP vs Advanced Scope

### MVP — hackathon-deliverable

| Area | In scope |
|---|---|
| Data | One basin (**North Atlantic**), roughly 20–25 seasons, IR window channel only |
| Fusion | HURSAT ↔ IBTrACS join at 6-hourly synoptic times |
| Detection | Binary classifier, Path-B negatives, explicitly scoped |
| Classification | Tier-A intensity category, single CNN |
| Intensity | Wind only, +24 h; GRU/LSTM plus GBM and persistence baselines |
| Track | Δlat/Δlon, +24 h; GRU/LSTM plus CLIPER-style and persistence baselines |
| Uncertainty | Empirical error radii from validation quantiles |
| Backend | FastAPI, read-mostly, precomputed predictions in PostgreSQL/PostGIS |
| Frontend | 5 pages, 2D Leaflet map, Recharts timeline, one 3D globe hero |
| Gemini | Grounded explanation of a selected forecast, plus a storm summary |
| Deploy | Vercel + one Docker API host + managed Postgres |

### Advanced — post-hackathon / research track

Multi-basin · multi-channel imagery (water vapour + visible) · ADT scene-type pattern model ·
reanalysis environmental predictors (SST, shear, ocean heat content) · rapid-intensification
specialist model · sequence-to-sequence and Transformer forecasters · deep ensembles and quantile
uncertainty · vision-embedding fusion into the temporal models · live data ingest · physics-informed
constraints · model cards and a full benchmark harness · PostGIS spatial analytics (landfall
statistics, basin climatology).

### Explicitly out of scope — permanently, for this prototype

Operational or real-time forecasting · storm surge or rainfall modelling · numerical weather
prediction · damage or casualty estimation · any public-safety advisory function.

---

## 6. Success Criteria

| Criterion | Threshold |
|---|---|
| Fusion pipeline produces a validated, leak-free dataset | Join QC gate passes (DATA_STRATEGY.md §4.4) |
| Detection | ROC-AUC > 0.90 on held-out storms |
| Classification | Macro-F1 meaningfully above class-prior; quadratic-weighted κ reported |
| Intensity | 24 h wind MAE **below persistence** on held-out storms |
| Track | 24 h mean great-circle error **below persistence and the CLIPER-style baseline** |
| System | End-to-end demo: pick storm → see prediction vs truth → read grounded explanation |
| Integrity | Zero fabricated numbers in Gemini output, enforced by the guardrail validator (API_ARCHITECTURE.md §8) |

---

## 7. Requirements Traceability

| Brief section | Addressed in |
|---|---|
| 1 Problem understanding | This document §1–2 |
| 2 Dataset strategy | DATA_STRATEGY.md §1–4 |
| 3 Multi-source fusion | DATA_STRATEGY.md §5–6 |
| 4 ML architecture | ML_ARCHITECTURE.md §1–3 |
| 5 Detection architecture | ML_ARCHITECTURE.md §4 |
| 6 Classification architecture | ML_ARCHITECTURE.md §5 |
| 7 Intensity architecture | ML_ARCHITECTURE.md §6 |
| 8 Track architecture | ML_ARCHITECTURE.md §7 |
| 9 Geospatial architecture | SYSTEM_ARCHITECTURE.md §6 |
| 10 Backend architecture | API_ARCHITECTURE.md §1–5 |
| 11 Frontend tech evaluation | UI_UX_ARCHITECTURE.md §2–3 |
| 12 Page structure | UI_UX_ARCHITECTURE.md §5 |
| 13 Gemini architecture | API_ARCHITECTURE.md §6–8 |
| 14 Database architecture | SYSTEM_ARCHITECTURE.md §7 |
| 15 Complete system architecture | SYSTEM_ARCHITECTURE.md §2–5 |
| 16 Directory structure | SYSTEM_ARCHITECTURE.md §8 |
| 17 Development phases | DEVELOPMENT_ROADMAP.md §1–3 |
| 18 Risks | DEVELOPMENT_ROADMAP.md §4 |
| 19 MVP vs advanced | This document §5 |
| 20 Final recommendations | DEVELOPMENT_ROADMAP.md §5–6 |
