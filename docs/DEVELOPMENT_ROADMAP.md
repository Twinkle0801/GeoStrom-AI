# DEVELOPMENT ROADMAP — GeoStrom AI

**Phase:** 10 (Premium Frontend Dashboard & Scientific Visualization) · **Status:** Complete,
two panels blocked by existing data/API contract (satellite imagery, classification).
Full results in [PHASE_10_FRONTEND_DASHBOARD.md](PHASE_10_FRONTEND_DASHBOARD.md).
Phases 0-10 are complete: Phase 3's vertical slice (Phase 2 predictions → PostgreSQL/PostGIS →
FastAPI → generated contract → Next.js/Leaflet map); Phase 4's real satellite fusion pipeline
(12 storms / 627 fused samples, see [PHASE_4_SATELLITE_PIPELINE.md](PHASE_4_SATELLITE_PIPELINE.md));
Phase 5's evidence-driven `scene_taxonomy_v1` classification taxonomy plus non-deep-learning
baseline (test macro-F1 0.559); Phase 6's CNN training (small CNN + ResNet-18, both evaluated,
neither beating the Phase 5 baseline at the current 353-image scale — the baseline ships, per the
roadmap's own "ship the honest winner" principle); Phase 7's GRU intensity model (absolute +
Δwind variants, per §5.2/§6's pre-existing spec), which likewise does not beat the Phase 2
LightGBM baseline at any horizon; and Phase 8's GRU track model (cos-latitude-weighted
displacement loss, per §5.2/§7's pre-existing spec), which does not beat the Phase 2 CLIPER-style
Ridge baseline at any horizon either (though it does beat plain Persistence at 18h/24h) — the
Phase 2 baselines ship for both intensity and track, same "ship the honest winner" principle; and
Phase 9's Gemini explanation layer (`backend/app/gemini/`), integrated as a strictly backend-only,
evidence-grounded narration service over the existing Phase 2/3 stored predictions — never a
forecasting model itself, per §6-8's pre-existing architecture. This "Phase 6"/"Phase 7"/"Phase 8"
numbering is informal (CNN-training deferred from Phase 5's numbering note, and the intensity/track
sequence-model items the brief calls Phase 7/8 but this roadmap places at **P2**); "Phase 9" and
"Phase 10" match the P0–P12 sequence's own **P9 (Gemini Explanation Layer)** and **P7 (UI/UX
Build)** respectively in substance, though this "Phase 10" ships before the roadmap's own P8
(Geospatial Dashboard Completion) — a deliberate, requested reordering, not a silent renumbering.
Phase 10 rebuilt the frontend into a five-page product (landing, Storm Explorer, the flagship
`/predict/[sid]` analysis workspace, Model Performance, Methodology) with a real TimeScrubber,
observed-vs-predicted charts, and a GeminiPanel/EvidenceDrawer wired to Phase 9's backend
endpoint — still distinct from **P6 (Detection)**, which has not started.

---

## 1. Re-sequencing the Phase Plan

The brief proposes: Foundation → UI/UX → Dataset → Preprocessing → Detection → Classification →
Intensity → Track → Gemini → Dashboard → Integration → Testing.

**Analysis suggests three changes.** Each is a scheduling decision with a concrete justification.

| # | Change | Why |
|---|---|---|
| **1** | **Move dataset acquisition and verification to Phase 1**, merged with foundation | Every unknown in DATA_STRATEGY.md §8 blocks design decisions downstream. Building UI for two weeks and *then* discovering that HURSAT cannot be cleanly joined to IBTrACS would be unrecoverable. **Verify the riskiest assumption first.** |
| **2** | **Build track and intensity before detection and classification** | Track and intensity need **only IBTrACS** — a small tabular download. Detection and classification need **HURSAT** — a large, slow, format-complex download whose join is unverified. Doing the IBTrACS-only work first produces a **complete, demonstrable, honestly-evaluated forecasting system while the satellite risk is still being retired.** |
| **3** | **Insert a thin vertical slice before the full UI build** | The brief builds UI second and integrates second-to-last, which concentrates all integration risk at the end — the classic failure mode. A minimal end-to-end path (one storm → one baseline forecast → one map) proves every seam early, when fixing them is cheap. The polished UI is then built onto a spine that already works. |

**The organising principle: retire the highest-uncertainty, highest-blast-radius risks first, and
have something demonstrable at the end of every phase.**

### 1.1 Mapping to the brief's phases

| Brief phase | Where it lives now |
|---|---|
| 0 Architecture | **P0** — this document set |
| 1 Foundation | **P1** (merged with dataset verification) |
| 2 UI/UX | **P7** (with a thin slice pulled forward into P3) |
| 3 Dataset Pipeline | **P1** (IBTrACS) + **P4** (satellite) |
| 4 Preprocessing & Fusion | **P4** |
| 5 Detection | **P6** |
| 6 Classification | **P5** |
| 7 Intensity Prediction | **P2** |
| 8 Track Prediction | **P2** |
| 9 Gemini | **P9** |
| 10 Geospatial Dashboard | **P8** |
| 11 Integration | **P10** |
| 12 Testing & Deployment | **P11** |

Nothing is dropped. The order is changed to front-load risk.

---

## 2. Phase Definitions

Each phase states its dependency, deliverables, and an **exit criterion** — an objective test for
whether it is finished. A phase is not complete until its exit criterion passes.

### P0 — Architecture ✅ COMPLETE
**Depends on:** nothing
**Deliverables:** the seven documents in `docs/` and the root README
**Exit:** the validation checklist in §3 passes and the plan is approved

---

### P1 — Foundation & Dataset Verification ✅ COMPLETE
**Depends on:** P0

1. ✅ Repository scaffold: `ml/geostrom_ml/` package, `ml/scripts/`, `ml/manifests/`, `ml/reports/`
2. ✅ **`$DATA_ROOT` relocated and guarded.** `C:\GeoStromData`, verified outside OneDrive (whose
   Known Folder Move on this machine also redirects Desktop/Documents/Pictures — all avoided) and
   outside the Git repo. `ml/geostrom_ml/config.py` enforces this programmatically at every call,
   not just by convention — an unsafe path raises `DataRootError` rather than silently proceeding.
3. ⏳ **Deferred to Phase 2.** Local PostgreSQL + PostGIS and Alembic were not needed to answer any
   Phase 1 verification question and were correctly left out per the strict scope rules.
4. ✅ **IBTrACS downloaded** (NA 57 MB + NI 28 MB, v04r01) and **all 7 IBTrACS TO-VERIFY items
   resolved** — see DATA_STRATEGY.md §8. Found and fixed a critical loader bug along the way
   (pandas silently parsing basin code `"NA"` as `NaN`, corrupting 99.5% of rows).
5. ✅ **HURSAT sample downloaded** (3 storms / 195 frames spanning 1995/2005/2015) and **all 3
   blocking satellite items resolved**: crosswalk (#8) — 100% exact SID match, embedded 3 ways;
   satellite-dedup field (#10) — `VZA` present on 100% of frames; per-file size (#9) — measured
   and extrapolated to ~13.8 GB for the full NA 1980–2015 subset.
6. ✅ **ADT-HURSAT inspected — scene-type field CONFIRMED PRESENT** (#16). This is the single
   highest-value finding of Phase 1: genuine Dvorak pattern labels (`Scene`/`EyeScene`/
   `CloudScene`) exist and are now the recommended primary classification target.
7. ✅ Published [`docs/PHASE_1_DATASET_VERIFICATION.md`](PHASE_1_DATASET_VERIFICATION.md) recording
   every answer; DATA_STRATEGY.md §8 updated in place with per-item verification status.

**Exit — all criteria met:** IBTrACS loads with asserted dtypes and physical ranges (after the
sentinel-handling fix) ✅ · a sample HURSAT storm joins to IBTrACS with position agreement inside
tolerance — measured median separation **0.0 km**, max 10.65 km, well inside the 50 km gate ✅ ·
every ⛔ item answered ✅ · a preliminary QC gate (19 automated checks, see the report) passes 19/19 ✅.
`make test`/`make lint`/CI were not built in Phase 1 (no application code exists yet to test) —
carried forward as a Phase 2 foundation task, not a Phase 1 gap.

> **The crosswalk did not fail — it exceeded expectations** (exact match, not merely "reliable").
> The project does not need to pivot to the IBTrACS-only fallback; the full multi-source plan is
> confirmed viable. P2's IBTrACS-only forecasting path remains valuable as a parallel deliverable
> and schedule hedge, not as a fallback from a failure that did not occur.

---

### P2 — Forecasting Baselines (IBTrACS only) ✅ COMPLETE
**Depends on:** P1 (IBTrACS portion only — **not** blocked on satellite data)

1. ✅ Causal feature engineering (20 per-timestep features, explicit temporal windows); sliding
   windows `L=8` (48h), horizons `H∈{6,12,18,24}h`, flattened to 160 columns for the tabular models
2. ✅ **Split by storm/season; froze `ml/manifests/splits_v1.json` and committed it** — season-block
   temporal split (train 1980–2004 / val 2005–2009 / test 2010–2015), 9,398 total windows, exact
   match to the Phase 1 pre-registered estimate
3. ✅ Track: persistence (constant-velocity) · CLIPER-style Ridge · LightGBM
4. ✅ Intensity: persistence · Ridge · LightGBM
5. ✅ Evaluation: Haversine error, along/cross-track decomposition, MAE/RMSE/bias, skill vs.
   persistence, all per-horizon
6. ✅ Benchmark harness + comparison tables (`ml/reports/phase2_benchmark_results.json`) + 6 plots

**Exit — met:** LightGBM beats persistence on intensity by **+19.8%** at 24h; CLIPER-style Ridge
beats persistence on track by **+11.4%** at 24h — both on the frozen, held-out, 1,732-window test
set (2010–2015, 88 storms). Leakage audit: 80/80 automated tests pass, including storm-level split
disjointness (verified on the materialised Parquet files, not just the manifest), per-timestep
causality (mutation-based regression tests), and the two antimeridian longitude-wrapping cases the
Phase 2 task specified verbatim. Full results: `docs/PHASE_2_FORECASTING_BASELINES.md`.

> **Honest finding, reported without adjustment:** CLIPER-style Ridge beats LightGBM on track at
> every horizon tested (though by a small, 1–3 km margin) — the opposite of what won on intensity.
> Both are reported; neither result was suppressed or reframed to favour a particular model.
>
> **After this phase the project has a real, defensible scientific result.** Everything that follows
> improves it. This is the insurance policy against a satellite-pipeline failure.

> **Update ("Phase 7"): intensity extended with the §5.2/§6-specified GRU (1 layer, hidden 64 —
> within §6.5's "1–2 layers, hidden 64–128" range; 4-horizon head), absolute + Δwind variants,
> Huber loss.** Trained and evaluated on
> the identical frozen split/features above — no new split, no re-derivation of Phase 2's numbers.
> **Honest finding: neither GRU variant beats LightGBM's MAE at any of the four horizons** (24h:
> 8.826 kt GRU-absolute vs. 8.535 kt LightGBM, a 3.4% gap that widens sharply at shorter horizons).
> RI recall is 0.0 at every horizon with true RI cases — reported as a diagnostic, not concealed.
> LightGBM remains the model to ship. Reproducibility verified bit-identical across two independent
> runs; 22 new tests pass. Full detail: [PHASE_7_INTENSITY_PREDICTION.md](PHASE_7_INTENSITY_PREDICTION.md).

> **Update ("Phase 8"): track extended with the §5.2/§7-specified GRU (1 layer, hidden 64 — within
> §7.4's "1–2 layers, hidden 64–128" range; 8-output displacement head), cos(latitude)-weighted
> Huber loss on displacement exactly per §7.2.** Trained and evaluated on the identical frozen
> split/features above — no new split, no re-derivation of Phase 2's numbers. **Honest finding:
> the GRU does not beat CLIPER-style Ridge (or LightGBM) at any of the four horizons** (24h:
> 209.505 km GRU vs. 200.445 km CLIPER, a 4.5% gap that widens sharply at shorter horizons), though
> it does beat plain Persistence at 18h/24h. Antimeridian handling and the cos(latitude) weighting
> were both directly tested. Reproducibility verified bit-identical across two independent runs; 19
> new tests pass. Full detail: [PHASE_8_TRACK_PREDICTION.md](PHASE_8_TRACK_PREDICTION.md).

---

### P3 — Vertical Slice (thin end-to-end) ✅ COMPLETE
**Depends on:** P2

1. ✅ Ingestion script writes P2 predictions to Postgres/PostGIS with `model_version` — long-form
   `predictions` table, idempotent upsert (verified: two consecutive runs produce identical row
   counts and model IDs). 41,568 predictions / 2,084 observations / 88 storms.
2. ✅ Minimal FastAPI, real routes (named per API_ARCHITECTURE.md's `/cyclones` resource, not the
   sketch's literal names): `/api/v1/cyclones`, `/api/v1/cyclones/{sid}`,
   `/api/v1/cyclones/{sid}/observations`, `/api/v1/tracks/{sid}`, `/api/v1/prediction/{sid}`,
   `/health`, `/api/v1/meta` — all read-only, zero ML imports in the request path (verified by
   import inspection).
3. ✅ Exported `contracts/openapi.json` from the live app; generated `frontend/lib/api-types.ts` via
   `openapi-typescript`; the whole frontend type-checks cleanly against it (`tsc --noEmit`, 0 errors).
4. ✅ Minimal Next.js page (`/predict/[sid]`): pick a storm → Leaflet map → observed track (solid
   teal) + predicted track (dashed, per-model colour), verified end-to-end against the real backend
   and real database (server-rendered HTML captured and inspected, not assumed).
5. ⏳ **Not done — scope-cut, documented, not silently dropped.** "Deploy to a real environment"
   (Vercel/Render/managed Postgres) was not performed in this session; the vertical slice runs
   against a local Docker Postgres+PostGIS and local `uvicorn`/`next dev` processes only. Promoting
   this to a real deployed URL is the natural first task of a later integration phase (P10) and
   requires no architecture change — the same Docker image and `DATABASE_URL`-driven config already
   supports it.

**Exit:** 154/154 tests pass (61 backend + 80 Phase 2 regression + 13 frontend), zero Phase 2
regression. The local vertical slice is real and runnable end-to-end; the deployed-URL portion of
the original exit criterion is deferred, not met. Full detail:
[PHASE_3_VERTICAL_SLICE.md](PHASE_3_VERTICAL_SLICE.md).

> Building the ingestion/API/frontend seam in P3 rather than P11 still paid off: the schema,
> contract-generation workflow, and observed/predicted map distinction are now proven against real
> data, long before any deployment-specific problem would need solving on top of them.

---

### P4 — Satellite Pipeline: Preprocessing & Fusion — ✅ **PIPELINE COMPLETE** (MVP-scale processing partial)
**Depends on:** P1 (verification) · parallelisable with P3

1. ✅ Configurable, idempotent HURSAT downloader (`ml/scripts/download_hursat_sample.py`) +
   real archive discovery (`ml/scripts/discover_hursat_archive.py`, measured 531/547 = 97.07%
   frozen-split-storm coverage, ~11.6 GB for full NA 1980–2015 coverage — refines, not just
   matches, Phase 1's 96%/13.8 GB extrapolation). **Full-archive download NOT executed** — see
   docs/PHASE_4_SATELLITE_PIPELINE.md §16/§19 for the measured reason and resume command.
2. ✅ NetCDF → **native 301×301 grid preserved** (not resampled to 224²; a Phase 0 placeholder
   this phase explicitly revised with documented reason — see DATA_STRATEGY.md §4.5) →
   physically-ranged (150–350 K) uint8 quantisation alongside a canonical float32 Kelvin array →
   **Zarr**; deterministic VZA-based satellite deduplication
3. ✅ Fusion join (identity + ±90 min, configurable); observed/interpolated/missing preserved;
   ADT-HURSAT Scene join (never intensity ground truth)
4. ✅ **18-point QC gate implemented** (`ml/geostrom_ml/satellite/qc.py`) — supersedes the
   preliminary 8/19-check Phase 1 gate with the Phase 4 task's full required report; publishes
   `ml/reports/satellite_qc_gate.json`
5. ⏳ Pre-render IR thumbnails for serving — deferred (frontend/API changes are explicitly out
   of Phase 4's scope)
6. ✅ Dataset manifest with build version (`ml/manifests/satellite_dataset_v1_manifest.json`)

**Exit:** the QC gate passes on real data at two scales (docs/PHASE_4_SATELLITE_PIPELINE.md §16):
the original 3-storm/109-sample Phase 1 set, and a real downloaded 12-storm/**627-fused-sample**
build spanning all three frozen splits (train 404 / val 128 / test 95) with 100% IBTrACS join,
0 spatial/temporal QC failures (max separation observed: 48.9 km, under the 50 km gate), and
100% ADT-HURSAT Scene coverage — QC gate **PASS (5/5)** both times. A random sample of fused
rows was inspected (rendered thumbnails + pixel-value histogram in
`ml/reports/figures/`). **Full MVP-scale (531-storm) processing was NOT completed** in this
phase — pipeline correctness is proven on real, measured, multi-storm, multi-split, multi-decade
data (1980–2015); see docs/PHASE_4_SATELLITE_PIPELINE.md for the exact resume command.

> **Decision point — pre-resolved by Phase 1 (TO-VERIFY #20), still pending full re-verification.**
> The estimated surviving count for NA 1980–2015 remains **~14,500 fused frames / ~9,000 sequence
> windows with imagery** (Phase 1's extrapolation; Phase 4's real discovery step refined the
> storm-coverage and byte-size inputs to that estimate but did not re-run the full join at scale).
> **Proceed with the pretrained-CNN plan (ResNet-18) as scoped in ML_ARCHITECTURE.md** — the
> fallback path stays documented as a contingency but is not the expected outcome. Re-verify the
> exact count once the full P4 join actually runs (resume command in
> docs/PHASE_4_SATELLITE_PIPELINE.md §19).

---

### P5 — Classification — ✅ **LABEL ANALYSIS + BASELINE COMPLETE** (CNN training NOT started)
**Depends on:** P4

> **Numbering note.** This roadmap originally bundled the label-analysis gate, the
> LightGBM/statistical baseline, AND CNN training into one "P5". The actual Phase 5 work
> session scoped itself to items 1-2 only (label analysis + non-deep-learning baseline),
> explicitly deferring items 3-5 (CNN, batch inference) to a separate, not-yet-started
> "Phase 6" — this section's item numbers are kept as originally written (not silently
> renumbered); ✅/⏳ markers below show what is actually done.

1. ✅ **Label-analysis gate — done, evidence-driven, real data.** Selected `scene_taxonomy_v1`
   (Tier B, ADT Scene labels, grounded in the ADT algorithm's own EyeScene/CloudScene code
   tables — not the invented Tier C wording): CDO (+IrrCDO), CurvedBand, Eye (+LargeEye),
   Shear. `Land` and `EmbCenter` excluded with explicit, documented reasons (not merged, not
   silently dropped). The pre-declared ~200-sample merge rule was inapplicable at the
   current 627-sample scale and was replaced with an explicit storm-level support test —
   see the annotation in `docs/ML_ARCHITECTURE.md` §5.1 and
   `docs/PHASE_5_CLASSIFICATION_LABEL_ANALYSIS.md`.
2. ✅ **Baseline: statistical image features + Logistic Regression / LightGBM — done, real
   data.** Test macro-F1 0.559 (logistic regression) vs. 0.079 majority-class floor. NOT yet
   the "class prior + LightGBM" wording literally — logistic regression edged out LightGBM
   on this small (353-train-sample) dataset; both are reported honestly in
   `ml/reports/phase5_baseline_results.json`.
3. ✅ **CNN (small from-scratch net, ResNet-18) — done in "Phase 6."** Both trained and evaluated
   on the frozen split; test macro-F1 0.370 (ResNet-18) and 0.356 (small CNN), **neither beats**
   the 0.559 logistic-regression baseline at this sample size (353 train images). Root cause
   (overfitting / small val-test per-class support) analysed, not just reported; baseline ships.
   Two real bugs (NaN-bleed from augmenting before invalid-pixel fill; a reproducibility bug from
   seeding after model construction) were found and fixed during development — both are documented
   in the Phase 6 doc, not silently corrected.
   See [PHASE_6_DEEP_LEARNING_CLASSIFICATION.md](PHASE_6_DEEP_LEARNING_CLASSIFICATION.md).
4. ⏳ Full metric suite incl. quadratic-weighted κ / category MAE — macro-F1, per-class
   precision/recall, confusion matrix ARE done for the baseline (§Exit below); κ/MAE assume
   an ordinal target and were not computed (`scene_taxonomy_v1` is nominal, not ordinal).
5. ⏳ Batch inference → `classifications` table — not started (no backend/DB work permitted
   in the Phase 5 work session).

**Exit (see numbering note above):** the class list is data-derived and documented
(✅); macro-F1 clears the class-prior/majority baseline (✅, 0.559 vs 0.079); no class has
zero recall without documented justification (✅ — `Eye`'s zero test-split recall is
extensively documented as a dataset-size limitation, not silently accepted). CNN-specific
work is now done (✅, "Phase 6") — the pre-declared bar (test macro-F1 > 0.559) was **not**
cleared by either CNN, an honestly-reported negative result, not a gap; the logistic-regression
baseline remains the shipped classification model until more labelled imagery is available.

---

### P6 — Detection
**Depends on:** P4, P5 (shares the vision training harness)

1. **Construct the negative set** — Path B first; Path A (GridSat) if schedule permits
2. Baseline: majority class + logistic regression on IR scalars
3. CNN detector + temperature-scaling calibration
4. Metrics: PR-AUC, ROC-AUC, precision/recall/F1, calibration curve, Brier score
5. Batch inference → `detections` table

**Exit:** ROC-AUC > 0.90 on held-out storms; the calibration curve is reported; **the scope of the
detection claim is written verbatim into the UI and methodology page.**

> If the baseline logistic regression achieves near-perfect separation, the negatives are trivially
> separable and the negative-construction design must be revisited. That check is part of the exit.

---

### P7 — UI/UX Build — ✅ **DASHBOARD COMPLETE** ("Phase 10"; Lighthouse not measured)
**Depends on:** P3 (working spine) · P5/P6 for full content

Design tokens · component library · landing page with the globe · Monitor · Analysis · Prediction
Dashboard · `TimeScrubber` · charts · motion system · accessibility pass.

**Exit:** all MVP pages render real data; Lighthouse targets met; `prefers-reduced-motion` honoured;
contrast verified over actual glass backdrops; colour-blind check on the intensity ramp passed.

> **Update ("Phase 10"): built, with two explicit exceptions.** Design tokens extended (`Inter`
> font actually loaded, glass-panel/badge/metric-card primitives), a lightweight `cobe` globe on
> the landing page, the flagship `/predict/[sid]` Analysis workspace with a real `TimeScrubber`,
> observed-vs-predicted charts, model selector, and Gemini panel/evidence drawer, plus a new
> Model Performance page and Methodology page. `prefers-reduced-motion` is honoured
> (`lib/motion.ts`); observed/predicted remain colour **and** line-style distinguishable.
> **Not measured**: Lighthouse scores (no Lighthouse run was performed this phase — performance
> was addressed via lazy-loading and memoization, but not benchmarked numerically). **Explicitly
> blocked by the existing data/API contract, not built**: real satellite-frame serving and
> per-storm classification results — no backend endpoint or database table exists for either;
> both panels show honest, correct empty states instead of fabricated data. Full detail:
> [PHASE_10_FRONTEND_DASHBOARD.md](PHASE_10_FRONTEND_DASHBOARD.md).

---

### P8 — Geospatial Dashboard Completion
**Depends on:** P7

Uncertainty cones (PostGIS-generated) · actual-vs-predicted layers · antimeridian splitting ·
zoom-dependent simplification · legends · layer toggles · error tooltips · model comparison view.

**Exit:** a storm crossing the antimeridian renders correctly; the map holds 60 fps while scrubbing;
the cone visibly widens with lead time.

---

### P9 — Gemini Explanation Layer — ✅ **BACKEND INTEGRATION COMPLETE** (`GeminiPanel`/`EvidenceDrawer` UI NOT built)
**Depends on:** P5–P8 (there must be real outputs to explain)

Evidence Packet Builder · system prompt · structured output schema · streaming ·
**Guardrail Validator** · deterministic template fallback · response caching · rate limiting ·
`GeminiPanel` + `EvidenceDrawer`.

**Exit:** on a 50-case adversarial test set, **zero unvalidated numeric claims reach the UI**; the
fallback path is exercised and verified; the API key is provably absent from the frontend bundle.

> **Late by design.** Gemini explains model outputs. Building it before the models exist means
> building against fabricated inputs — which trains the wrong instincts and hides grounding bugs.

> **Update: backend done, per the task's explicit minimal-scope instruction.** Evidence Packet
> Builder, system prompt, structured output schema, and the Guardrail Validator are implemented and
> tested (132 backend tests, 71 new) — `docs/PHASE_9_GEMINI_INTEGRATION.md`. **Not built this
> phase, deliberately deferred**: streaming, response caching, rate limiting, and the frontend
> `GeminiPanel`/`EvidenceDrawer` components (frontend work is explicitly out of Phase 9's scope;
> UI/UX build is a later phase). The exit bar's spirit — zero unvalidated claims reaching the
> caller — is met on a smaller, representative adversarial set (the task's own ~24 named
> scenarios, not a full 50-case sweep) plus 5/5 real Gemini calls passing grounding cleanly after
> fixing 3 real bugs found via the manual smoke test; expanding to the roadmap's full 50-case bar is
> noted as a legitimate next step, not claimed as already met.

---

### P10 — Integration & Hardening
**Depends on:** P1–P9

End-to-end wiring · caching and ETags · error and empty states · loading skeletons ·
`/analytics/model-performance` wired into the methodology page · full seed script · dataset
attribution and licensing · disclaimers placed.

**Exit:** every MVP user journey completes without a console error; every page has defined loading,
empty, and error states.

---

### P11 — Testing & Deployment
**Depends on:** P10

Unit tests (features, Haversine, fusion joins) · **leakage regression test** (assert no storm spans
two splits) · API contract tests · one end-to-end journey test · production deploy · seeded database ·
performance validation · README/demo script.

**Exit:** CI green; production URL live and seeded; a cold visitor can complete the core journey;
the leakage test is in CI permanently.

---

### P12 — Advanced Track (post-hackathon)
Multi-basin · multi-channel imagery · ADT scene-type pattern model · reanalysis predictors (SST,
shear, ocean heat content) · L2 embedding fusion · seq2seq and Transformer forecasters · deep
ensembles and quantile uncertainty · rapid-intensification specialist · live ingest adapter ·
model cards · PostGIS climatology analytics.

### 2.1 Dependency graph

```
 P0 ─▶ P1 ─┬─▶ P2 ─▶ P3 ──────────────┐
           │                          │
           └─▶ P4 ─┬─▶ P5 ─▶ P6 ──────┤
                   │                  │
                   └──────────────────┴─▶ P7 ─▶ P8 ─▶ P9 ─▶ P10 ─▶ P11 ─▶ P12

 Critical path: P0 → P1 → P4 → P5 → P7 → P8 → P9 → P10 → P11
 P2/P3 run in parallel with P4 and form the fallback deliverable if P4 fails.
```

**The parallel branch is deliberate.** P2→P3 needs only IBTrACS; P4→P6 needs satellite data. If the
satellite branch fails or overruns, P3 still yields a complete, honest, deployed product.

---

## 3. Phase 0 Validation Checklist

| # | Check | Result |
|---|---|---|
| 1 | All 23 brief sections addressed | ✅ Traceability table, PROJECT_REQUIREMENTS §7 |
| 2 | Architecture internally consistent | ✅ One data contract (§DATA §5.2) flows into ML specs, DB schema, API responses, and UI components without contradiction |
| 3 | Frontend / backend / ML responsibilities separated | ✅ SYSTEM_ARCHITECTURE §3, enforced by separate packages and a generated contract |
| 4 | Gemini is not the prediction engine | ✅ API_ARCHITECTURE §6.1 — structural isolation, not just prompt instruction; Gemini receives only a JSON packet and has no tools or data access |
| 5 | Dataset assumptions marked | ✅ 23 numbered TO-VERIFY items, ⛔-flagged where blocking |
| 6 | ML models replaceable | ✅ `ModelRegistry` contract; `model_version` on every stored prediction; API depends on output spec, not model |
| 7 | MVP realistic | ✅ Scope cut to one basin, one channel, 24 h horizon, 5 pages; fallback deliverable defined at P3 |
| 8 | No gratuitous technology | ✅ Rejected with reasons: GSAP, Mapbox, deck.gl, raw R3F, MongoDB, TimescaleDB, Redis, Celery, GraphQL, Spark, Airflow, model-serving frameworks |

---

## 4. Risk Register

Ordered by expected impact. **Risks 1–4 were discovered during this analysis and are not in the
brief's list** — they are the ones most likely to derail the project.

| # | Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|---|
| **1** | **No negative class for detection.** HURSAT is storm-centric; a detector trained on it sees only positives | **Critical** — the capability is unbuildable as naively specified | **Certain** (structural) | Constructed negatives, Path B (non-tropical/pre-genesis frames) as the default, Path A (GridSat sampling) as the upgrade. **Scope the claim honestly in the UI.** Decided in P0, built in P6 |
| **2** | **HURSAT ↔ IBTrACS crosswalk fails or is ambiguous** | **Critical** — the entire fusion premise collapses | Medium | **Verified in P1 on a small sample, before the full download.** Fallback: name+season+basin matching with manual validation. Ultimate fallback: the P2/P3 IBTrACS-only product |
| **3** | **Project directory is inside OneDrive** | High — sync storms, file locks, dehydrated files failing mid-epoch, corrupted checkpoints, quota exhaustion | **Certain if unaddressed** | **`$DATA_ROOT` outside OneDrive, set in P1 as a blocking task.** Checkpoints and Zarr stores also outside. Only source and docs stay synced |
| **4** | **Mixed wind-averaging conventions across agencies** | High — a systematic label inconsistency the model learns as noise or a basin artefact | High if unhandled | **One agency's wind column for the entire training set; no fallback to another agency.** Recorded in the dataset manifest |
| **5** | **Data leakage** via random splits, near-duplicate adjacent frames, duplicate satellite views, non-causal features, or scalers fitted on all data | **Critical** — produces excellent, meaningless results, and is often discovered only after publication | High without controls | Split by storm and season · dedup before splitting · causal-features-only rule · train-only scalers · **frozen committed split manifests** · **an automated leakage regression test in CI** |
| **6** | **Dataset volume exceeds disk or download time** | High | Medium | One basin, one channel, ~20–25 seasons, 6-hourly · uint8 quantisation (4× reduction) · size measured on a sample in P1 before committing |
| **7** | **NetCDF/HDF format complexity** — unfamiliar structures, scale/offset attributes, fill values, projection conventions | Medium | High | `xarray` with labelled dimensions · a verification notebook before bulk conversion · **assert the brightness-temperature range physically** (a wrong scale factor silently destroys imagery while training proceeds normally) |
| **8** | **Missing data** — nulls in pressure, daylight-only visible channel, scan gaps | Medium | High | Wind as the primary target; IR-only for MVP; require contiguous windows rather than interpolating targets; report population rates in the manifest |
| **9** | **Temporal mismatch** — satellite scan times off synoptic hours | Medium | Certain | ±90 min nearest join, Δt retained per row for audit; tolerance re-tuned to the measured offset distribution in P1 |
| **10** | **Spatial mismatch** — HURSAT centre disagrees with best-track position | Medium | Low | QC assertion 1 (< 50 km separation for > 99% of rows); non-conforming rows dropped and counted |
| **11** | **Class imbalance** — intense categories are rare | Medium | Certain | Class weights → balanced sampling → targeted augmentation → pre-declared merge rule; **macro-F1 and per-class recall as headline metrics**, never accuracy |
| **12** | **Limited labels / no true pattern labels** (ADT scene types absent) | Medium | Medium | Tier-A intensity categories as the guaranteed fallback; **the model is then described as intensity-stage classification, not Dvorak pattern classification.** Both wordings pre-drafted so no rewrite is needed |
| **13** | **Overfitting** — small dataset, large pretrained models | Medium | High | Small backbones (ResNet-18) · pretrained weights · early stopping on validation · augmentation · dropout · **baselines that expose an unearned gap** |
| **14** | **Track/intensity uncertainty is irreducibly large** | Medium | Certain | Empirical error radii shown with every forecast · skill reported against persistence and CLIPER · **honest expectation-setting in the methodology page rather than claims of operational competitiveness** |
| **15** | **Intensity skill is weak** because SST, shear, and ocean heat content are unavailable | Medium | High | Stated as a known limitation from the outset · Δwind variant to expose true skill · RI recall reported as a diagnostic · reanalysis predictors listed as the top Advanced upgrade |
| **16** | **GPU limits (6 GB VRAM)** | Medium | Certain | Model choices sized to the budget (ML_ARCHITECTURE §9) · mixed precision · gradient accumulation · **ViT explicitly rejected for MVP** |
| **17** | **Gemini hallucination** | **High** — a fabricated cyclone statement is a credibility failure and, in this domain, a safety-adjacent one | Medium without controls | **Five independent layers** (API_ARCHITECTURE §8): structural isolation · prompt constraints · structured output · **deterministic numeric guardrail validator** · UI transparency + evidence drawer. Deterministic template fallback on validation failure |
| **18** | **API latency** | Low | Low | Precomputed inference · indexed reads · ETag caching · Gemini streamed and cached. **Largely designed out by the offline/online split** |
| **19** | **Frontend performance** — heavy map + charts + 3D | Medium | Medium | Server-side track simplification · dynamic imports · one 3D instance, lazy and below the fold, with a static fallback · memoised charts · measured budgets |
| **20** | **3D rendering performance on low-end devices** | Low | Medium | Capability detection with an image fallback; 3D is decorative and never load-bearing for information |
| **21** | **Scope creep** | **High** — the most common cause of hackathon failure | High | MVP table is a contract · every phase has an objective exit criterion · Advanced list exists as a parking lot · P3 guarantees a shippable product early |
| **22** | **Integration risk concentrated at the end** | High | Medium | **Vertical slice in P3 and a real deployment in P3**, not P11 |
| **23** | **Windows-specific friction** — Store Python, compiled geospatial wheels, `spawn` multiprocessing, path length | Medium | Medium | Docker for Postgres/PostGIS · prefer wheels, use conda-forge if a build fails · `num_workers` tuned conservatively · consider a standard python.org or `uv`-managed interpreter over the Store build |

---

## 5. Final Recommendations

### 5.1 Recommended tech stack

| Layer | Recommendation |
|---|---|
| **Frontend** | Next.js 15 (App Router) · TypeScript (strict) · Tailwind CSS · Framer Motion · shadcn/ui · TanStack Query · Recharts · react-leaflet · react-globe.gl (one instance) · *optional* Lenis (landing only) |
| **Backend** | Python 3.11 · FastAPI · Pydantic v2 · SQLAlchemy 2.0 + GeoAlchemy2 · Alembic · Uvicorn · ONNX Runtime (reserved) |
| **ML** | PyTorch · timm · scikit-learn · LightGBM · NumPy/pandas · xarray + netCDF4 · zarr · Albumentations |
| **Database** | **PostgreSQL + PostGIS** (serving) · **DuckDB + Parquet** (offline analytics) · Zarr/HDF5 (imagery) |
| **Geospatial** | PostGIS (server) · GeoPandas/Shapely (offline) · Leaflet (client) · GeoJSON as the wire format |
| **Visualisation** | Recharts (charts) · Leaflet (maps) · react-globe.gl (single 3D hero) |
| **AI Assistant** | Gemini — **backend-only**, evidence-grounded, schema-constrained, guardrail-validated, with a deterministic fallback |
| **Deployment** | Vercel (frontend) · Render or Fly.io (backend Docker, CPU) · Neon or Supabase (managed Postgres+PostGIS) · ML pipeline run locally, writing results to the managed database |

### 5.2 Recommended initial models

| Task | Baseline (mandatory) | **Ship this** | Later |
|---|---|---|---|
| **Detection** | Majority class + logistic regression on IR scalars | **ResNet-18 (ImageNet-pretrained, 1-channel) + temperature scaling** | EfficientNet-B0 · ViT/Swin · self-supervised pretraining |
| **Classification** | Class prior + LightGBM on IR scalars | **ResNet-18**, class-weighted CE + label smoothing, physics-aware augmentation | Ordinal (CORAL/CORN) head · ADT scene-type target · image+scalar multimodal head |
| **Intensity** | **Persistence** + Ridge + **LightGBM** | **GRU (2×128) → 4-horizon head**, Huber loss; absolute + Δwind variants | Temporal CNN · Transformer · quantile head · RI specialist · reanalysis predictors |
| **Track** | **Persistence** + **CLIPER-style regression** + LightGBM | **GRU (2×128) → 8-output displacement head**, cos-latitude-weighted Huber | Seq2seq + attention · probabilistic output · deep ensembles · physics constraints |

**On LightGBM:** it is listed as a baseline but may well win. On tabular sequences of this size,
gradient boosting is frequently competitive with or better than small recurrent networks. **If it
wins, ship it and say so.** Shipping the honest winner is the correct outcome, not a disappointment.

### 5.3 Recommended dataset strategy

| Tier | Dataset | Role |
|---|---|---|
| **Primary** | **IBTrACS** | The spine: identity, position, intensity, all labels, all joins, the database. Small, fast, and sufficient on its own for a complete forecasting product |
| **Secondary** | **HURSAT-B1** | The eyes: storm-centric IR imagery for detection, classification, and L1 image features. One basin, one channel, ~20–25 seasons |
| **Optional** | **ADT-HURSAT** | The interpreter: potential true pattern labels (decisive for classification framing), a satellite-only intensity benchmark, and a QC signal |
| **Optional (stretch)** | **GridSat-B1** | Negative-class source that would let detection make a general, rather than scoped, claim |

**Sequencing:** IBTrACS in P1 → forecasting product in P2/P3 → HURSAT in P4 → vision models in P5/P6.
This ordering means the project always has a working deliverable.

### 5.4 Biggest unknowns

Full list in DATA_STRATEGY.md §8. **The five that change the architecture rather than a parameter:**

1. **Does a clean HURSAT ↔ IBTrACS storm-identifier crosswalk exist?** *If not, there is no image
   fusion, and the project becomes an IBTrACS-only forecasting system.*
2. **Does ADT-HURSAT expose an ADT scene-type field?** *Decides whether we deliver true Dvorak
   pattern classification or intensity-stage classification. Changes the headline claim.*
3. **How many samples survive fusion, deduplication, synoptic filtering, and window construction?**
   *If the count is small, the CNNs are replaced by scalar-feature GBMs.*
4. **What is the real on-disk size and download time of the chosen HURSAT subset?** *Determines
   whether the satellite branch fits the schedule at all.*
5. **Which IBTrACS column distinguishes observed from interpolated rows, and what is each agency
   column's population rate?** *Determines the intensity target and whether labels are genuine
   observations.*

Also unresolved: exact channel names, grid geometry, and brightness-temperature units; the storm
duration distribution (which sets feasible `L`); the class distribution of the chosen label set; the
scan-time offset distribution; and licensing/attribution requirements.

---

## 6. Scope Guard

**The MVP table in PROJECT_REQUIREMENTS.md §5 is a contract.** Anything not in it goes on the
Advanced list and waits. Three rules:

1. **No new capability is added until every MVP capability has passed its exit criterion.**
2. **No technology is added without a named problem it solves that an existing dependency cannot.**
3. **If a phase overruns, cut scope within that phase — never cut evaluation, baselines, or the
   methodology page.** Those are what make the results trustworthy, and a prototype with honest,
   modest numbers is worth far more than an impressive one nobody can verify.
