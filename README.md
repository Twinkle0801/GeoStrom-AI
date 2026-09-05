# GeoStrom AI

**AI/ML-based identification, classification, and prediction of tropical cyclone patterns from multi-source satellite and best-track data.**

> **Status: Phase 5 — Classification Label Analysis. COMPLETE.**
> Audited the real 627-sample Phase 4 dataset at both sample- and storm-level (12 storms
> total) and selected `scene_taxonomy_v1`: **CDO** (+IrrCDO), **CurvedBand**, **Eye**
> (+LargeEye), **Shear** — grounded in the ADT algorithm's own EyeScene/CloudScene code
> tables (`ml/scripts/verify_adt.py`, Phase 1), not an invented axis. `Land` (89 samples,
> not a genuine storm-pattern class) and `EmbCenter` (17 samples/5 storms, 1 in test) are
> excluded with explicit reasons, never silently dropped — 521/627 samples remain.
> Non-deep-learning baselines (majority-class, logistic regression, LightGBM) trained on
> deterministic image statistics from the canonical Zarr store, evaluated on the frozen
> storm-level split: best test **macro-F1 = 0.559** (logistic regression) vs. 0.079
> majority-class floor. `Eye` has zero test-split samples — a real, documented dataset
> limitation (only 3 test storms), not hidden. 10 adversarial leakage tests pass. See
> [docs/PHASE_5_CLASSIFICATION_LABEL_ANALYSIS.md](docs/PHASE_5_CLASSIFICATION_LABEL_ANALYSIS.md).

> **Phase 7 — Intensity Prediction: GRU (Sequence Model). COMPLETE — baseline retained.**
> Trained a GRU (1 layer, hidden 64) → shared 4-horizon dense head on the same frozen Phase 2
> split/features (no new split, no new download), per `docs/ML_ARCHITECTURE.md` §6's pre-existing
> spec: absolute-wind (deliverable) and Δwind (diagnostic) variants, Huber loss. **Neither variant
> beats the Phase 2 LightGBM baseline at any of the four horizons** — GRU (abs) scores 8.826 kt MAE
> at 24h vs. LightGBM's 8.535 kt (3.4% worse), and the gap is far larger at short horizons (46%
> worse at 6h) where LightGBM's direct access to the current wind value dominates. RI (rapid
> intensification) recall is 0.0 at every horizon with true RI cases — a genuine, honestly reported
> diagnostic limitation, not a new capability claim. Reproducibility verified bit-identical across
> two independent full training runs. 22 new tests pass (8 leakage/scientific-validation vectors +
> model/determinism tests). **Recommendation: keep shipping the Phase 2 LightGBM baseline for
> intensity.** See
> [docs/PHASE_7_INTENSITY_PREDICTION.md](docs/PHASE_7_INTENSITY_PREDICTION.md).

> **Phase 6 — Deep-Learning Cyclone Pattern Classification. COMPLETE — baseline retained.**
> Trained and evaluated a from-scratch small CNN and a grayscale-adapted, ImageNet-pretrained,
> mostly-frozen ResNet-18 (per `docs/ML_ARCHITECTURE.md`'s pre-existing recommendation) on the
> frozen `scene_taxonomy_v1` taxonomy and split. **Neither beats the Phase 5 logistic-regression
> baseline** (test macro-F1 0.559): small CNN scored 0.356, ResNet-18 scored 0.370 — both
> honestly reported, not suppressed. Root cause: real overfitting (train macro-F1 reaches
> 0.99–1.0) and validation-selection instability at only 7 train / 2 val / 3 test storms, not an
> implementation defect. Two real bugs were found and fixed while developing this phase: invalid
> pixels (stored as NaN in the canonical Zarr) bleeding into neighboring pixels when augmentation
> ran before the fill step, and model weights being randomly initialized before the seed was set
> (breaking reproducibility). Both fixed, tested, and documented. **Recommendation: ship the
> Phase 5 baseline**; the CNN path needs more labelled imagery, not more tuning, to be
> reconsidered. See
> [docs/PHASE_6_DEEP_LEARNING_CLASSIFICATION.md](docs/PHASE_6_DEEP_LEARNING_CLASSIFICATION.md).

> **Phase 4 — Satellite Data Pipeline. PIPELINE COMPLETE; MVP-scale processing PARTIAL.**
> The full HURSAT-B1 → IRWIN QC → dedup → IBTrACS/ADT-HURSAT fusion → Zarr + Parquet pipeline is
> implemented and verified end-to-end on real data at two scales. First, against the Phase 1
> verification sample (195 frames/3 storms), reproducing Phase 1's own numbers exactly
> (195 → 109 frames after VZA dedup, 100% IBTrACS join). Then, after a real, deterministic,
> seeded download of 9 more storms, a full production build over **12 storms / 1,097 raw HURSAT
> frames → 627 fused samples** spanning all three frozen splits (train 404 / val 128 / test 95),
> with 100% IBTrACS join, 0 spatial/temporal QC failures, and 100% ADT-HURSAT Scene-label
> coverage — QC gate **PASS (5/5)** at both scales. Real full-archive discovery (not
> extrapolation) found 531/547 (97.07%) frozen-split storms have HURSAT-B1 coverage (~11.6 GB).
> The remaining ~519-storm archive was **not** downloaded/processed this phase — measured
> per-file I/O cost on this workstation projects to ~18 hours of wall-clock for full coverage,
> reported honestly rather than faked. See
> [docs/PHASE_4_SATELLITE_PIPELINE.md](docs/PHASE_4_SATELLITE_PIPELINE.md) for exact numbers,
> the resume command, and what remains. No CNN/classification training — that is Phase 5.

> **Phase 3 — Vertical Slice. COMPLETE.**
> The first full, real, end-to-end path is built and verified: Phase 2 baseline predictions →
> PostgreSQL/PostGIS → read-only FastAPI → generated OpenAPI contract → Next.js + Leaflet map,
> with observed and predicted tracks visually distinct. 41,568 predictions / 2,084 observations /
> 88 storms ingested idempotently from the real Phase 2 artifact; 154/154 tests pass (61 backend +
> 80 Phase 2 regression + 13 frontend). See
> [docs/PHASE_3_VERTICAL_SLICE.md](docs/PHASE_3_VERTICAL_SLICE.md). This is an integration proof,
> not the final product UI — no CNN/GRU/LSTM, no Gemini, no auth, no live data.

> **Phase 2 — Forecasting Baselines. COMPLETE.** North Atlantic locked as the MVP basin.
> IBTrACS-only Persistence/Ridge/LightGBM (intensity) and Persistence/CLIPER-style/LightGBM (track)
> baselines, trained and evaluated on a frozen, storm-level, leak-tested split — see
> [docs/PHASE_2_FORECASTING_BASELINES.md](docs/PHASE_2_FORECASTING_BASELINES.md). Headline (24h):
> LightGBM beats persistence on intensity by 19.8%; CLIPER-style Ridge beats persistence on track
> by 11.4% (and edges out LightGBM on track — reported honestly, not adjusted).

> **Phase 1 — Foundation & Dataset Verification. COMPLETE.** The multi-source fusion premise
> (IBTrACS ↔ HURSAT-B1 ↔ ADT-HURSAT) was verified against real, small samples of official
> NOAA/NCEI data — see [docs/PHASE_1_DATASET_VERIFICATION.md](docs/PHASE_1_DATASET_VERIFICATION.md).

---

## What GeoStrom AI Is

GeoStrom AI is a **retrospective tropical cyclone (TC) intelligence system**. It ingests historical
storm-centric satellite infrared imagery and best-track records, fuses them into a single
spatio-temporal dataset, and trains four independent, swappable ML components:

| Capability | Question it answers |
|---|---|
| **Detection** | Is an organised tropical cyclone present in this satellite scene? |
| **Classification** | What structural pattern / intensity stage is this cyclone in? |
| **Intensity Prediction** | What will its wind speed and central pressure be over the next 24 h? |
| **Track Prediction** | Where will its centre be over the next 24 h? |

Results are served through a FastAPI backend to a Next.js geospatial dashboard, with a
**Gemini explanation layer** that verbalises — but never produces — the numerical forecasts.

## What GeoStrom AI Is *Not*

- **Not an operational forecasting product.** It replays historical storms. It has no live data feed.
- **Not a safety-critical system.** It must never be used for evacuation, routing, or emergency decisions.
- **Not a Gemini-powered forecaster.** Gemini explains model output; it never predicts.

## Architecture at a Glance

```
 HURSAT-B1 (IR imagery)   IBTrACS (best track)   ADT-HURSAT (intensity/scene)
            |                      |                        |
            +----------- Ingestion & Validation -------------+
                                   |
                        Alignment & Fusion (SID + synoptic time)
                                   |
                     Canonical Store (Parquet + Zarr/HDF5)
                                   |
                          Feature Engineering
                                   |
        +--------------+-----------+-----------+--------------+
        | Detection    | Classify  | Intensity | Track        |   <- swappable via ModelRegistry
        +--------------+-----------+-----------+--------------+
                                   |
                   Offline Batch Inference -> PostgreSQL/PostGIS
                                   |
                            FastAPI (read-mostly)
                                   |
                   Next.js + TypeScript + Tailwind frontend
                     (maps, charts, 3D globe, dashboards)
                                   |
                  Gemini Explanation Layer (grounded, backend-only)
```

The central architectural decision: **because the system is retrospective, all ML inference is
precomputed offline and stored.** The API is read-mostly. This removes GPU-in-production
requirements, removes inference latency from the request path, and lets the whole serving stack
run on free-tier CPU hosting.

## Documentation

| Document | Purpose |
|---|---|
| [docs/PROJECT_REQUIREMENTS.md](docs/PROJECT_REQUIREMENTS.md) | Problem analysis, capabilities, scope boundaries, success criteria |
| [docs/DATA_STRATEGY.md](docs/DATA_STRATEGY.md) | Dataset roles, join keys, alignment, fusion, labels, leakage control |
| [docs/ML_ARCHITECTURE.md](docs/ML_ARCHITECTURE.md) | Modular ML design, per-task models, losses, metrics, baselines |
| [docs/SYSTEM_ARCHITECTURE.md](docs/SYSTEM_ARCHITECTURE.md) | End-to-end system, geospatial layer, database schema, directory layout |
| [docs/API_ARCHITECTURE.md](docs/API_ARCHITECTURE.md) | FastAPI endpoint design, contracts, Gemini grounding architecture |
| [docs/UI_UX_ARCHITECTURE.md](docs/UI_UX_ARCHITECTURE.md) | Design language, frontend tech evaluation, page architecture |
| [docs/DEVELOPMENT_ROADMAP.md](docs/DEVELOPMENT_ROADMAP.md) | Phase plan, dependencies, risk register, MVP scope guard |
| [docs/PHASE_1_DATASET_VERIFICATION.md](docs/PHASE_1_DATASET_VERIFICATION.md) | Dataset verification against real IBTrACS/HURSAT/ADT samples |
| [docs/PHASE_2_FORECASTING_BASELINES.md](docs/PHASE_2_FORECASTING_BASELINES.md) | IBTrACS-only baseline models, frozen split, benchmark results |
| [docs/PHASE_3_VERTICAL_SLICE.md](docs/PHASE_3_VERTICAL_SLICE.md) | Database, API, ingestion, and frontend — the first working end-to-end slice |
| [docs/PHASE_4_SATELLITE_PIPELINE.md](docs/PHASE_4_SATELLITE_PIPELINE.md) | HURSAT-B1/ADT-HURSAT fusion pipeline, QC gate, Zarr/Parquet dataset |
| [docs/PHASE_5_CLASSIFICATION_LABEL_ANALYSIS.md](docs/PHASE_5_CLASSIFICATION_LABEL_ANALYSIS.md) | Scene-label audit, `scene_taxonomy_v1`, classification dataset, non-deep-learning baselines |
| [docs/PHASE_6_DEEP_LEARNING_CLASSIFICATION.md](docs/PHASE_6_DEEP_LEARNING_CLASSIFICATION.md) | CNN + transfer-learning training/evaluation, overfitting analysis, comparison against the Phase 5 baseline |
| [docs/PHASE_7_INTENSITY_PREDICTION.md](docs/PHASE_7_INTENSITY_PREDICTION.md) | GRU sequence model (absolute + Δwind), leakage validation, honest comparison against the Phase 2 LightGBM baseline |

## Recommended Stack (summary)

**Frontend** Next.js 15 (App Router) · TypeScript · Tailwind · Framer Motion · react-leaflet · Recharts · react-globe.gl
**Backend** Python 3.11 · FastAPI · Pydantic v2 · SQLAlchemy 2.0 · Alembic · ONNX Runtime
**ML** PyTorch · timm · scikit-learn · LightGBM · xarray/netCDF4 · Albumentations
**Data/DB** PostgreSQL + PostGIS (serving) · DuckDB + Parquet (analytics) · Zarr/HDF5 (imagery)
**AI Assistant** Gemini (backend-only, grounded, schema-validated)
**Deploy** Vercel (web) · Render/Fly.io Docker (API) · Neon/Supabase (Postgres+PostGIS)

## Hardware Baseline (this workstation)

`AMD Ryzen 7 7840HS (8C/16T)` · `15.3 GB RAM` · `RTX 4050 Laptop, 6 GB VRAM` · `639 GB free`

6 GB VRAM is the binding constraint on model selection — see
[docs/ML_ARCHITECTURE.md](docs/ML_ARCHITECTURE.md#9-compute-budget--rtx-4050-6-gb-vram).

> ⚠️ **Known environment risk:** this project directory is inside OneDrive. Before Phase 3,
> dataset and checkpoint storage must be relocated outside the synced folder via a `DATA_ROOT`
> environment variable. See the risk register in the roadmap.

## Conventions Used in These Docs

- **TO VERIFY** — a claim that must be confirmed against the real dataset before implementation.
- **ASSUMPTION** — a working assumption adopted to allow planning to proceed.
- **DECISION** — a locked architectural choice with stated rationale.

---

*Phase 0 deliverable. Do not begin Phase 1 implementation until this plan is reviewed and approved.*
