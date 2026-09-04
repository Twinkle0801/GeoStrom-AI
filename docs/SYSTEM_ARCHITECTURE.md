# SYSTEM ARCHITECTURE — GeoStrom AI

**Phase:** 0 (Architecture) · **Status:** Complete

---

## 1. Architectural Thesis

Three decisions shape everything else:

1. **The system is retrospective.** It replays historical storms. There is no live feed.
2. **Therefore all ML inference is precomputed offline** and stored as rows. The API is read-mostly.
3. **Therefore the serving stack is small, cheap, CPU-only, and stateless** — a database, a FastAPI
   process, and a static frontend.

This is the single most consequential architectural choice in the project. It eliminates, in one
move: GPU hosting cost, model-loading memory pressure, inference latency in the request path,
cold-start problems, autoscaling complexity, and the possibility of a model crashing the API.
It also converts "will it be fast enough?" into "is the index right?".

---

## 2. Complete System Architecture

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  OFFLINE  —  runs on the workstation / a GPU box.  Never in the request path ║
╚══════════════════════════════════════════════════════════════════════════════╝

  ┌────────────┐   ┌──────────────┐   ┌──────────────┐
  │  IBTrACS   │   │  HURSAT-B1   │   │  ADT-HURSAT  │     external sources
  │ best track │   │  IR imagery  │   │  ADT record  │
  └─────┬──────┘   └──────┬───────┘   └──────┬───────┘
        │                 │                  │
        ▼                 ▼                  ▼
  ┌──────────────────────────────────────────────────┐
  │ 1. INGESTION          idempotent, checksummed    │
  │    download · verify · store immutably in raw/   │
  └────────────────────────┬─────────────────────────┘
                           ▼
  ┌──────────────────────────────────────────────────┐
  │ 2. PREPROCESSING                                 │
  │    tabular: units, NA sentinels, agency choice,  │
  │             synoptic filter, dtype assertions    │
  │    imagery: decode NetCDF, dedup satellites,     │
  │             resample 224², quantise uint8 → Zarr │
  └────────────────────────┬─────────────────────────┘
                           ▼
  ┌──────────────────────────────────────────────────┐
  │ 3. FUSION            join on (storm id, ±90 min) │
  │    ── QC GATE ──  8 assertions; failure blocks   │
  └────────────────────────┬─────────────────────────┘
                           ▼
  ┌──────────────────────────────────────────────────┐
  │ 4. FEATURE ENGINEERING     strictly causal       │
  │    tendencies · motion · cyclic time · IR scalars│
  │    sliding windows (L=8, H=4)                    │
  └────────────────────────┬─────────────────────────┘
                           ▼
  ┌──────────────────────────────────────────────────┐
  │ 5. SPLIT   by storm, preferably by season        │
  │    → splits.json  (version-controlled, frozen)   │
  └────────────────────────┬─────────────────────────┘
                           ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │ 6. MODEL TRAINING            all behind ModelRegistry            │
  │  ┌───────────┐ ┌──────────────┐ ┌───────────┐ ┌───────────┐      │
  │  │ Detection │ │Classification│ │ Intensity │ │   Track   │      │
  │  └───────────┘ └──────────────┘ └───────────┘ └───────────┘      │
  │           each: baseline → MVP model → (advanced)                │
  └────────────────────────┬─────────────────────────────────────────┘
                           ▼
  ┌──────────────────────────────────────────────────┐
  │ 7. BENCHMARK HARNESS   one frozen split,         │
  │    one metric set, one comparison table          │
  └────────────────────────┬─────────────────────────┘
                           ▼
  ┌──────────────────────────────────────────────────┐
  │ 8. EXPORT + BATCH INFERENCE                      │
  │    PyTorch → ONNX; run every model over every    │
  │    storm-time; compute empirical error radii     │
  └────────────────────────┬─────────────────────────┘
                           │  ═══ the only handoff ═══
╔══════════════════════════▼═══════════════════════════════════════════════════╗
║  ONLINE  —  read-mostly, CPU-only, stateless                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
                           ▼
  ┌──────────────────────────────────────────────────┐
  │  PostgreSQL + PostGIS                            │
  │  storms · observations · predictions ·           │
  │  classifications · detections · model_versions   │
  │  + object storage / static dir for IR thumbnails │
  └────────────────────────┬─────────────────────────┘
                           ▼
  ┌──────────────────────────────────────────────────┐
  │  FastAPI          Pydantic v2 · SQLAlchemy 2.0   │
  │  ┌────────────────────────────────────────────┐  │
  │  │ /cyclones /tracks /detection /classification│ │
  │  │ /prediction /analytics /explain             │ │
  │  └────────────────────────────────────────────┘  │
  │  + response cache  + GeoJSON serialisation       │
  │  + EVIDENCE PACKET BUILDER ──┐                   │
  └────────────────────────┬─────┼───────────────────┘
                           │     ▼
                           │  ┌───────────────────────────────┐
                           │  │ GEMINI EXPLANATION LAYER      │
                           │  │ backend-only · grounded ·     │
                           │  │ schema-constrained            │
                           │  │      ↓                        │
                           │  │ GUARDRAIL VALIDATOR           │
                           │  │ numeric + claim verification  │
                           │  └───────────┬───────────────────┘
                           ▼              ▼
  ┌──────────────────────────────────────────────────┐
  │  Next.js 15 · TypeScript · Tailwind              │
  │  ┌──────────┬──────────┬──────────┬───────────┐  │
  │  │ Landing  │ Monitor  │ Analysis │ Prediction│  │
  │  ├──────────┴──────────┴──────────┴───────────┤  │
  │  │ Leaflet map · Recharts · 3D globe · panels  │  │
  │  └─────────────────────────────────────────────┘  │
  └──────────────────────────────────────────────────┘
```

### 2.1 Improvements over the brief's sketch

| Added | Why |
|---|---|
| **Explicit offline / online boundary** | The brief's linear flow implies the models sit in the request path. Separating them is the decision that makes the system deployable on free-tier CPU hosting |
| **QC gate after fusion** | A blocking, automated correctness checkpoint. Without it, a bad join silently produces a model that trains fine and means nothing |
| **Benchmark harness as a first-class stage** | The comparison table is the research deliverable, not a by-product |
| **Batch inference + model registry** | Predictions are data, versioned and queryable. Two model versions can coexist for comparison |
| **Evidence Packet Builder before Gemini** | Gemini is fed a structured, verified fact set — never raw database access and never free-form context |
| **Guardrail Validator after Gemini** | Output is machine-checked against the evidence packet before it reaches the user. Closes the hallucination path |
| **ONNX export step** | Keeps PyTorch out of the API container: smaller image, faster cold start, CPU-only deployment |

---

## 3. Component Responsibilities

**Strict rule: each concern lives in exactly one layer.**

| Layer | Owns | Explicitly does *not* |
|---|---|---|
| **ML (`ml/`)** | NetCDF decoding, fusion, feature engineering, training, evaluation, ONNX export, batch inference, uncertainty calibration | Serve HTTP · touch the frontend · call Gemini |
| **Backend (`backend/`)** | Querying stored results, geometry simplification, GeoJSON assembly, pagination, caching, evidence-packet construction, Gemini calls, guardrail validation, auth | Train models · run heavy inference · compute scientific quantities from scratch |
| **Frontend (`frontend/`)** | Rendering, interaction, animation, client-side view state, map/chart display, streaming Gemini text | Compute predictions · reproject geometry · hold API keys · do meteorological maths |
| **Database** | Durable storage of storms, observations, predictions, model metadata; spatial indexing | Business logic beyond spatial queries and aggregation |

**The load-bearing rule:** *every scientific number the user sees is computed offline in `ml/`,
stored, and merely transported by the backend.* The frontend never derives a physical quantity, and
Gemini never produces one.

---

## 4. Data Flow — a single user interaction

```
User opens Prediction Dashboard for storm S at time T
  │
  ├─▶ GET /api/cyclones/{sid}                  → metadata          (cached)
  ├─▶ GET /api/tracks/{sid}?format=geojson     → observed track    (cached, simplified)
  ├─▶ GET /api/prediction/{sid}?t=T            → 4 horizons + cone (precomputed row lookup)
  ├─▶ GET /api/classification/{sid}?t=T        → class + confidence(precomputed row lookup)
  └─▶ POST /api/explain  { sid, t, scope }
         │
         ├─ backend assembles EVIDENCE PACKET from the same stored rows
         ├─ calls Gemini with the packet + a strict system prompt
         ├─ GUARDRAIL: every number in the output must exist in the packet
         └─ streams validated text to the client
```

Every request is an indexed read. **No model executes.** p95 target < 300 ms is comfortably
achievable.

---

## 5. Online vs Offline — the boundary rule

| Runs offline (batch) | Runs online (request) |
|---|---|
| NetCDF decode, resample, quantise | Row lookup by `(sid, timestamp)` |
| Fusion and QC | Geometry simplification and GeoJSON assembly |
| Feature engineering | Aggregation for analytics (materialised where hot) |
| Training and evaluation | Evidence packet assembly |
| **All four models' inference** | Gemini call + guardrail validation |
| Error-radius / cone computation | Response caching |
| Thumbnail rendering | — |

**When this boundary must be revisited:** only if live/user-supplied imagery is added
(Advanced scope). At that point a single ONNX Runtime session is added to the API — which is exactly
why models are exported to ONNX now. The migration path is deliberately short.

---

## 6. Geospatial Architecture

### 6.1 Split of responsibility

| Concern | Where | Why |
|---|---|---|
| Track geometry construction (points → LineString) | **Backend, PostGIS** | Geometry is data; build it once, index it |
| Antimeridian handling (splitting a crossing LineString) | **Backend** | A rendering bug caused by a data property should be fixed where the data lives, not in every client |
| Distance / Haversine error | **Offline ML** (for metrics), **PostGIS geography** (for queries) | Must be identical everywhere; never recomputed in JavaScript |
| Track simplification for display | **Backend**, zoom-dependent (`ST_Simplify`) | Sending thousands of vertices to Leaflet kills frame rate |
| Uncertainty cone geometry | **Offline ML** produces radii; **backend** turns radii into a polygon (`ST_Buffer` on geography + convex hull) | Geodesically correct buffering is a PostGIS strength and a JavaScript weakness |
| Basemap tiles | **Frontend** (external tile provider) | Standard |
| Rendering, panning, hover, layer toggles | **Frontend** | Pure presentation |
| Colour scales, symbology | **Frontend**, from a shared token file | Design concern |

**Rule: the frontend receives GeoJSON that is already correct, simplified, and split.** It performs
no reprojection and no geodesic maths. This is the difference between a map that works and a map
with a mysterious line across the Pacific.

### 6.2 GeoJSON contract

One `FeatureCollection` per track request, with typed features:

| `properties.kind` | Geometry | Meaning |
|---|---|---|
| `observed_track` | LineString (possibly MultiLineString if split at ±180°) | Best-track history up to T |
| `observed_point` | Point | Per-timestep observation; properties carry wind, pressure, category |
| `current_position` | Point | The selected time T |
| `future_actual` | LineString | Ground truth after T — the comparison baseline |
| `predicted_track` | LineString | Model output from T |
| `predicted_point` | Point | +6/+12/+18/+24 h; properties carry wind, error radius, `model_version` |
| `uncertainty_cone` | Polygon | Envelope from empirical error radii |

Every feature carries `model_version` where model-derived, so the UI can always answer *"which model
said this?"*.

### 6.3 Technology choices

| Tool | Verdict | Reason |
|---|---|---|
| **PostGIS** | ✅ **Adopt** | Geodesic distance, buffering, simplification, and spatial indexing — all correct and server-side. The `geography` type handles the sphere properly |
| **GeoPandas / Shapely** | ✅ **Adopt (offline only)** | Track geometry construction, land intersection, and QC in the ML pipeline |
| **Leaflet + react-leaflet** | ✅ **Adopt (MVP)** | Small, stable, zero-cost, no API token, ideal for polylines/markers/polygons. Exactly our workload |
| **MapLibre GL JS** | ⚪ Consider later | Vector tiles and GPU rendering, better for very large datasets and 3D terrain. Not needed for a few hundred vertices |
| **Mapbox GL JS** | ❌ Reject | Requires an access token and a billing relationship. MapLibre is the open fork with no such requirement |
| **Turf.js** | ⚪ Minimal use only | Client-side geodesy invites divergence from the server's numbers. Permitted only for trivial view-state maths |

---

## 7. Database Architecture

### 7.1 Evaluation

| Option | Fit | Verdict |
|---|---|---|
| **PostgreSQL + PostGIS** | Data is strongly relational (storm → observations → predictions), numeric, time-ordered, and geospatial. PostGIS gives geodesic distance, buffering, simplification, and GiST spatial indexes. Schema is stable and known. Managed free tiers with PostGIS exist. | ✅ **RECOMMENDED** |
| **MongoDB** | Would require application-side joins for what is a naturally relational model, offers markedly weaker geospatial capability than PostGIS (no geodesic buffering/simplification), and lacks strong typing on numeric fields where physical-range validation matters. Its schema flexibility solves a problem this project does not have — the schema is known in advance and stable. | ❌ **Reject** |
| **SQLite / DuckDB + Parquet** | Excellent for local analytics and the ML pipeline; DuckDB queries Parquet with no ETL. But SpatiaLite is weaker than PostGIS and concurrent serving is not its strength. | ✅ **Adopt for the analytics/ML path**, not for serving |
| **TimescaleDB** | Time-series partitioning for a dataset in the 10⁵–10⁶ row range is optimising a problem we do not have. | ❌ Reject (unnecessary complexity) |

**DECISION: PostgreSQL + PostGIS for serving; DuckDB + Parquet for offline analytics.**
Both are accessed through a repository interface, so Phase 1 can develop against SQLite before a
Postgres instance exists.

### 7.2 Preliminary schema

> Field names marked ⚠ depend on dataset verification (DATA_STRATEGY.md §8) and are expected to change.

```sql
-- ═══ REFERENCE ═══════════════════════════════════════════════════════════
CREATE TABLE model_versions (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,             -- 'track_gru_v1'
    task            TEXT NOT NULL,             -- detection|classification|intensity|track
    version         TEXT NOT NULL,
    trained_at      TIMESTAMPTZ NOT NULL,
    dataset_build   TEXT NOT NULL,             -- links to the dataset manifest
    git_commit      TEXT,
    config          JSONB NOT NULL,            -- hyperparameters
    metrics         JSONB NOT NULL,            -- test-set metrics
    error_radii_km  JSONB,                     -- {6:.., 12:.., 18:.., 24:..} for cones
    is_active       BOOLEAN DEFAULT FALSE,     -- the version the UI shows by default
    UNIQUE (name, version)
);

-- ═══ CORE ════════════════════════════════════════════════════════════════
CREATE TABLE storms (
    sid             TEXT PRIMARY KEY,          -- IBTrACS serial id ⚠
    name            TEXT,
    season          SMALLINT NOT NULL,
    basin           TEXT NOT NULL,
    subbasin        TEXT,
    start_time      TIMESTAMPTZ NOT NULL,
    end_time        TIMESTAMPTZ NOT NULL,
    n_observations  INTEGER NOT NULL,
    max_wind        REAL,                      -- lifetime max — SUMMARY ONLY, never a model feature
    min_pressure    REAL,
    max_category    SMALLINT,
    made_landfall   BOOLEAN,
    split           TEXT,                      -- train|val|test — provenance for the methodology page
    track_geom      GEOGRAPHY(LINESTRING, 4326),
    bbox            GEOGRAPHY(POLYGON, 4326)
);
CREATE INDEX ON storms (season);
CREATE INDEX ON storms (basin, season);
CREATE INDEX ON storms USING GIST (track_geom);

CREATE TABLE observations (
    id              BIGSERIAL PRIMARY KEY,
    sid             TEXT NOT NULL REFERENCES storms(sid) ON DELETE CASCADE,
    ts              TIMESTAMPTZ NOT NULL,
    step_index      INTEGER NOT NULL,          -- ordinal position within the storm
    lat             DOUBLE PRECISION NOT NULL,
    lon             DOUBLE PRECISION NOT NULL, -- normalised convention, fixed at ingest
    geom            GEOGRAPHY(POINT, 4326) NOT NULL,
    wind            REAL,                      -- single chosen agency ⚠
    pressure        REAL,                      -- ⚠
    category        SMALLINT,                  -- ⚠
    nature          TEXT,                      -- ⚠
    storm_speed     REAL,
    storm_dir       REAL,
    dist2land       REAL,
    is_synoptic     BOOLEAN NOT NULL,          -- 00/06/12/18 UTC
    is_observed     BOOLEAN,                   -- not interpolated ⚠
    image_key       TEXT,                      -- Zarr/thumbnail key; NULL when no frame joined
    image_dt_min    SMALLINT,                  -- join offset, retained for audit
    UNIQUE (sid, ts)
);
CREATE INDEX ON observations (sid, ts);
CREATE INDEX ON observations USING GIST (geom);

-- ═══ MODEL OUTPUTS ═══════════════════════════════════════════════════════
CREATE TABLE detections (
    id              BIGSERIAL PRIMARY KEY,
    sid             TEXT NOT NULL REFERENCES storms(sid) ON DELETE CASCADE,
    ts              TIMESTAMPTZ NOT NULL,
    model_id        INTEGER NOT NULL REFERENCES model_versions(id),
    probability     REAL NOT NULL,
    is_positive     BOOLEAN NOT NULL,          -- at the frozen validation threshold
    UNIQUE (sid, ts, model_id)
);

CREATE TABLE classifications (
    id              BIGSERIAL PRIMARY KEY,
    sid             TEXT NOT NULL REFERENCES storms(sid) ON DELETE CASCADE,
    ts              TIMESTAMPTZ NOT NULL,
    model_id        INTEGER NOT NULL REFERENCES model_versions(id),
    predicted_class TEXT NOT NULL,
    confidence      REAL NOT NULL,
    probabilities   JSONB NOT NULL,            -- full distribution, for the UI bar chart
    true_class      TEXT,                      -- ground truth, retrospective mode
    UNIQUE (sid, ts, model_id)
);

-- One row per (origin time, lead time). Long form: simple to query, trivially
-- extensible to new horizons without a schema migration.
CREATE TABLE predictions (
    id              BIGSERIAL PRIMARY KEY,
    sid             TEXT NOT NULL REFERENCES storms(sid) ON DELETE CASCADE,
    origin_ts       TIMESTAMPTZ NOT NULL,      -- forecast issued for this observation time
    lead_hours      SMALLINT NOT NULL,         -- 6 | 12 | 18 | 24
    valid_ts        TIMESTAMPTZ NOT NULL,      -- origin_ts + lead_hours
    model_id        INTEGER NOT NULL REFERENCES model_versions(id),

    pred_lat        DOUBLE PRECISION,
    pred_lon        DOUBLE PRECISION,
    pred_geom       GEOGRAPHY(POINT, 4326),
    pred_wind       REAL,
    pred_pressure   REAL,

    error_radius_km REAL,                      -- empirical, from model metadata

    -- ground truth + error, populated because the system is retrospective
    true_lat        DOUBLE PRECISION,
    true_lon        DOUBLE PRECISION,
    true_wind       REAL,
    track_error_km  REAL,                      -- Haversine, computed offline
    wind_error_kt   REAL,

    UNIQUE (sid, origin_ts, lead_hours, model_id)
);
CREATE INDEX ON predictions (sid, origin_ts);
CREATE INDEX ON predictions (model_id, lead_hours);

-- Cached uncertainty polygons; cheaper than rebuilding per request
CREATE TABLE forecast_cones (
    id              BIGSERIAL PRIMARY KEY,
    sid             TEXT NOT NULL REFERENCES storms(sid) ON DELETE CASCADE,
    origin_ts       TIMESTAMPTZ NOT NULL,
    model_id        INTEGER NOT NULL REFERENCES model_versions(id),
    geom            GEOGRAPHY(POLYGON, 4326) NOT NULL,
    UNIQUE (sid, origin_ts, model_id)
);
```

**Design notes**

- `predictions` is **long form** (one row per horizon), not wide (`pred_lat_6h`, `pred_lat_12h`, …).
  Adding a 48 h horizon later becomes an insert, not a migration.
- `model_id` on every output row is what makes model swapping non-breaking: two versions coexist and
  the UI can compare them.
- `storms.max_wind` is a **display summary only**. It is a lifetime aggregate and must never enter a
  feature pipeline — the ML layer computes *max-so-far* from `observations`. The comment in the
  schema exists to prevent exactly that mistake.
- `observations.split` provenance lets the methodology page state honestly whether a demonstrated
  storm was in the training set.
- `geography` (not `geometry`) is used so distances and buffers are in metres on a sphere by default.

### 7.3 Imagery storage

IR frames stay **out of the database**. Postgres is a poor blob store and free tiers are small.

- **Offline:** Zarr store under `$DATA_ROOT`, keyed by `(sid, timestamp)`.
- **Serving:** pre-rendered PNG thumbnails (256², a perceptually-ordered colour map) on the API's
  static mount or object storage; `observations.image_key` holds the path.
- **Rationale:** the browser needs a picture, not a physical array. Rendering once offline is faster,
  smaller, and removes any array-processing dependency from the API.

---

## 8. Project Directory Structure

```
GeoStrom AI/
├── README.md
├── Makefile                      # one entry point for every pipeline stage
├── docs/                         # Phase 0 planning + living architecture docs
│   ├── PROJECT_REQUIREMENTS.md
│   ├── SYSTEM_ARCHITECTURE.md
│   ├── DATA_STRATEGY.md
│   ├── ML_ARCHITECTURE.md
│   ├── API_ARCHITECTURE.md
│   ├── UI_UX_ARCHITECTURE.md
│   └── DEVELOPMENT_ROADMAP.md
│
├── ml/                           # everything offline. No HTTP, no Gemini.
│   ├── pyproject.toml
│   ├── config/                   # YAML: dataset build, features, model hyperparams, class list
│   ├── geostrom_ml/
│   │   ├── data/                 # download · parse · convert · fuse · qc
│   │   ├── features/             # causal feature engineering, windowing, scalers
│   │   ├── datasets/             # torch Dataset / DataLoader over Zarr + Parquet
│   │   ├── models/
│   │   │   ├── base.py           # BaseModel contract — the modularity boundary
│   │   │   ├── registry.py       # name → class
│   │   │   ├── baselines/        # persistence, CLIPER-style, ridge, lightgbm
│   │   │   ├── vision/           # resnet, efficientnet, (vit)
│   │   │   └── temporal/         # gru, lstm, tcn, (transformer)
│   │   ├── training/             # loops, losses, schedulers, seeds
│   │   ├── evaluation/           # metrics, haversine, skill scores, benchmark harness
│   │   ├── uncertainty/          # empirical error radii, calibration
│   │   └── inference/            # onnx export, batch inference, DB writer
│   ├── scripts/                  # 00_download · 01_convert · 02_fuse · 03_features …
│   ├── notebooks/                # EDA + the blocking label-analysis notebook
│   └── tests/
│
├── backend/                      # FastAPI. Serves stored results. No training.
│   ├── pyproject.toml
│   ├── app/
│   │   ├── main.py
│   │   ├── core/                 # settings, logging, cache, errors
│   │   ├── api/v1/               # cyclones · tracks · detection · classification
│   │   │                         # prediction · analytics · explain
│   │   ├── schemas/              # Pydantic v2 — the source of truth for the API contract
│   │   ├── db/                   # SQLAlchemy models, session, repositories
│   │   ├── services/             # geojson assembly, simplification, aggregation
│   │   └── gemini/               # evidence packet · prompts · client · guardrail validator
│   ├── alembic/                  # migrations
│   └── tests/
│
├── frontend/                     # Next.js. Renders. Computes nothing scientific.
│   ├── package.json
│   ├── app/                      # App Router pages
│   ├── components/               # map · charts · panels · ui · motion
│   ├── lib/                      # api client, generated types, formatters
│   ├── styles/                   # design tokens, Tailwind config
│   └── public/
│
├── contracts/
│   └── openapi.json              # exported from FastAPI → generates frontend TS types
│
├── infra/
│   ├── docker/                   # api.Dockerfile, compose for local Postgres+PostGIS
│   └── deploy/
│
└── .env.example                  # DATA_ROOT, DATABASE_URL, GEMINI_API_KEY (names only)
```

**Not in the repository:** `datasets/`. Data lives at **`$DATA_ROOT`, outside OneDrive and outside
git**. The brief's sketch placed `datasets/` in the tree; with multi-GB satellite archives, a synced
folder, and git, that is actively harmful. The repository stores *manifests and checksums*, which is
what actually needs versioning.

### 8.1 Directory rationale

| Directory | Purpose | Key boundary |
|---|---|---|
| `ml/` | All offline computation: data, features, models, evaluation, batch inference | Never imports from `backend/`. Runnable with no server and no database |
| `backend/` | Transport, query, assembly, Gemini orchestration, guardrails | Never imports from `ml/`. Depends on the database schema and ONNX artefacts, not on training code |
| `frontend/` | Presentation and interaction | Types are *generated* from `contracts/openapi.json`, never hand-written |
| `contracts/` | The typed seam between backend and frontend | A schema change that breaks the client fails at compile time, not in a demo |
| `docs/` | Living architecture record | Updated as decisions change; TO-VERIFY items resolved in place |
| `infra/` | Reproducible local and deployed environments | Local Postgres+PostGIS via Docker — no manual install |
| `Makefile` | `make fuse`, `make train-track`, `make benchmark`, `make serve`… | Pipeline stages must be one command each, or they will not be re-run |

**Why `ml/` and `backend/` are separate packages, not one:** they have genuinely different dependency
sets. `ml/` needs torch, xarray, netCDF4, zarr — hundreds of megabytes. `backend/` needs FastAPI,
SQLAlchemy, and onnxruntime. Keeping them separate makes the deployed API image small and prevents
the API from ever importing training code by accident. It also enforces the layering rule in §3 at
the packaging level rather than by convention.

---

## 9. Deployment Architecture

```
  GitHub ──▶ CI (lint · type-check · tests · export openapi)
     │
     ├──▶ Vercel                    frontend (static + edge SSR)
     │
     ├──▶ Render / Fly.io           backend Docker (CPU, 512 MB–1 GB)
     │        └── env: DATABASE_URL, GEMINI_API_KEY, ALLOWED_ORIGINS
     │
     ├──▶ Neon / Supabase           PostgreSQL + PostGIS (managed, free tier)
     │
     └──▶ (workstation, manual)     ml/ pipeline → batch inference → writes to the DB
```

- **The ML pipeline is not deployed.** It runs on the workstation and writes results to the managed
  database. This is correct for a retrospective system and avoids all GPU hosting cost.
- **Thumbnails** ship either in the API image (if small) or in object storage.
- **Secrets** exist only as host environment variables. `GEMINI_API_KEY` is never present in the
  frontend build, in `NEXT_PUBLIC_*`, or in the repository.
- **Database seeding** is a one-command script so the deployment is reproducible from artefacts.
