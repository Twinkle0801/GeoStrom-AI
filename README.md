# GeoStrom AI

**AI/ML-based identification, classification, and prediction of tropical cyclone patterns from multi-source satellite and best-track data.**

> **Status: Phase 3 — Vertical Slice. COMPLETE.**
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
