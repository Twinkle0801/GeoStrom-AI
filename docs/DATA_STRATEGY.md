# DATA STRATEGY — GeoStrom AI

**Phase:** 1 (Foundation & Dataset Verification) · **Status:** Verified against real data.

> **Update (Phase 1):** every TO-VERIFY item below has been checked against a real, small sample
> of IBTrACS v04r01, HURSAT-B1 v06, and ADT-HURSAT (NCEI Accession 0307249). Verified claims are
> marked ✅ **VERIFIED**, partial findings ⚠️ **PARTIALLY VERIFIED**, failures ❌ **FAILED**, and
> anything not yet checked ⏳ **NOT YET VERIFIED**. The full evidence, methodology, and raw report
> files are in [docs/PHASE_1_DATASET_VERIFICATION.md](PHASE_1_DATASET_VERIFICATION.md). Original
> Phase 0 planning-grade text is left in place below and annotated rather than deleted, so the
> reasoning trail stays visible.

---

## 1. Dataset Roles

Three sources, three distinct jobs. Assigning each a single primary role prevents the classic
mistake of trying to make one dataset do everything.

| Dataset | Role | Feeds |
|---|---|---|
| **IBTrACS** | **The spine.** Authoritative storm identity, position, and intensity time series. Defines which storms exist, when, and where. | Track model, intensity model, all labels, all joins, the database |
| **HURSAT-B1** | **The eyes.** Storm-centric satellite infrared imagery, already re-gridded around the storm centre. | Detection model, classification model, image-derived features |
| **ADT-HURSAT** | **The interpreter.** Homogenised, algorithmically-derived intensity estimates from the HURSAT imagery, and potentially structural scene types. | Optional: pattern labels (Tier B), a satellite-only intensity cross-check, dataset QC |

### 1.1 Primary source per task

| Task | Primary | Secondary | Rationale |
|---|---|---|---|
| Detection | HURSAT-B1 | IBTrACS (`NATURE` for negative selection) | Imagery is the input; IBTrACS supplies the label context |
| Classification | HURSAT-B1 | IBTrACS (Tier A labels) / ADT-HURSAT (Tier B labels) | Imagery in, label from track or ADT |
| Intensity prediction | IBTrACS | HURSAT-derived image features; ADT as cross-check | Best-track wind is the target; imagery is an auxiliary predictor |
| Track prediction | IBTrACS | HURSAT-derived features | Position history is overwhelmingly the dominant signal |
| Visualisation / DB | IBTrACS | HURSAT (thumbnails) | Track geometry and metadata |

**Key consequence:** the two *prediction* tasks are primarily **IBTrACS problems** and can be built
end-to-end **without touching a single satellite file.** This is the highest-leverage scheduling fact
in the project — see DEVELOPMENT_ROADMAP.md §2.

---

## 2. IBTrACS — International Best Track Archive for Climate Stewardship

**What it is:** the consolidated global best-track archive maintained by NOAA NCEI, merging the
post-season reanalysed best tracks of the world's regional forecast agencies.

**Why it is the spine:** it is small, tabular, globally consistent, and carries a stable storm
identifier that everything else can be joined against.

### 2.1 Expected content — TO VERIFY

| Field group | Expected fields | Use |
|---|---|---|
| Identity | Storm serial ID, season, basin, sub-basin, name, agency cross-IDs | Join key, grouping, split-by-storm |
| Time | ISO timestamp | Temporal join key, sequence ordering |
| Position | Latitude, longitude | Track target, map geometry |
| Intensity | Maximum sustained wind, minimum central pressure (WMO-reported and per-agency variants) | Intensity target, category labels |
| Classification | Storm nature/type, Saffir–Simpson-style category | Label source, negative selection |
| Derived | Storm translation speed and direction, distance to land, landfall indicator | Features |
| Structure | Radius of maximum wind, wind radii by quadrant | Advanced features — expect heavy sparsity |

### 2.2 Known pitfalls — these will bite if not handled

| # | Pitfall | Why it matters | Mitigation |
|---|---|---|---|
| 1 | **Wind averaging period differs by agency.** US agencies conventionally report 1-minute sustained wind; most other agencies report 10-minute sustained. | Mixing them creates a systematic ~10–15% label inconsistency that the model will learn as noise or, worse, as a basin artefact. **This is the most commonly missed error in TC ML work.** | **DECISION: use a single agency's wind column for the whole training set** (US columns for the North Atlantic MVP). Do not fall back to the WMO column when the chosen agency is null — drop the row instead. Record the choice in the dataset manifest. Cross-agency harmonisation is Advanced scope. |
| 2 | **A units row follows the header** in the CSV distribution. | Naive `read_csv` yields an all-string dataframe and silently poisons every numeric column. | Skip the units row explicitly; assert dtypes after load. **TO VERIFY** exact structure. |
| 3 | **Multiple missing-value sentinels** — empty strings, whitespace, and numeric sentinels such as -999. | Sentinels silently become real numbers; a -999 kt wind will wreck a scaler. | Central `na_values` list plus a post-load range assertion per column (e.g. wind ∈ [0, 200] kt, pressure ∈ [850, 1050] hPa). |
| 4 | **Not every row is an observation.** Some rows are interpolated to sub-synoptic times. | Training on interpolated labels teaches the model an interpolation scheme, inflating apparent skill. | **DECISION: restrict supervised learning to 6-hourly synoptic times (00/06/12/18 UTC)** and, where a track-type/source indicator exists, to observed rows only. **TO VERIFY** which indicator column expresses this. |
| 5 | **Antimeridian crossing.** Longitude conventions and storms crossing ±180°. | Breaks distance maths, breaks map polylines, breaks any absolute-coordinate regression. | Normalise longitude convention once at ingest; predict *displacements* not absolutes; split rendered LineStrings at the antimeridian; compute distance with Haversine, never Euclidean. |
| 6 | **Pre-satellite-era storms** have far lower quality and coverage. | Contaminates training with unreliable labels. | Restrict to the satellite era; the HURSAT join naturally enforces this. |
| 7 | **Pressure is sparser than wind.** | A pressure model may have far fewer samples than expected. | Verify population rate before committing pressure as an MVP target — currently scoped as secondary. |

### 2.3 Scale

Order of 10⁵–10⁶ rows globally across all seasons — **TO VERIFY**. Comfortably fits in memory and in
a single Parquet file. No big-data tooling required for the track/intensity path.

---

## 3. HURSAT-B1 — Storm-Centric Satellite Imagery

**What it is:** geostationary and polar-orbiter infrared observations, re-gridded and **re-centred on
the tropical cyclone centre**, at roughly 8 km resolution on an approximately 301×301 grid, at
3-hourly intervals, distributed as NetCDF. **TO VERIFY:** exact grid size, resolution, channel names,
coverage period, and per-file packaging.

**Expected channels** — **TO VERIFY**: an infrared window channel (roughly 11 µm), a water-vapour
channel, a visible channel (daylight only, therefore ~50% missing), and possibly additional
near-IR/split-window channels.

### 3.1 Decisions driven by its structure

| Property | Consequence | Decision |
|---|---|---|
| Already storm-centred | Spatial matching to IBTrACS is a **QC check, not a registration problem**. Huge simplification. | Use the centre agreement as a validation gate, not as a processing step |
| Only contains storms | **No negative class for detection** | See PROJECT_REQUIREMENTS.md §2.A |
| Visible channel is daylight-only | ~50% missing, non-random (correlated with local solar time) | **MVP uses the IR window channel only.** A model fed VIS would learn a day/night artefact |
| Multiple satellites may view one storm simultaneously | Duplicate frames at the same timestamp; if split carelessly, the same scene lands in train *and* test | **Deduplicate to one frame per (storm, timestamp)** before splitting. Selection rule: prefer the smallest satellite view-zenith angle if available, else a fixed satellite priority list, else first-by-name for determinism. **TO VERIFY** which fields support this |
| 3-hourly cadence | Twice the IBTrACS synoptic cadence | Join at 6-hourly synoptic times only for supervised learning; retain 3-hourly frames for visualisation |
| NetCDF, one file per storm/satellite/time | Tens of thousands of small files; Windows filesystem and OneDrive both suffer badly | **Convert once to a consolidated store** (§4.5) |

### 3.2 Volume management

The full multi-basin, multi-channel archive is large — plausibly on the order of 10²–10³ GB
(**TO VERIFY**). The 639 GB of free disk is not a comfortable margin, and the directory is inside
OneDrive.

**DECISION — aggressive subsetting for MVP:**

1. **One basin** (North Atlantic) — the best-observed and best-analysed basin.
2. **One channel** (IR window).
3. **~20–25 seasons**, chosen to sit inside the modern geostationary era.
4. **6-hourly synoptic frames** for training (3-hourly retained only for display of demo storms).
5. **Downsample and quantise** at conversion time (§4.5).

Estimated resulting training store: single-digit GB. **TO VERIFY** once real file sizes are known.

---

## 4. ADT-HURSAT

**What it is:** the Advanced Dvorak Technique applied retrospectively and homogeneously to the
HURSAT imagery, producing a satellite-only intensity record that is internally consistent across
decades and satellites (unlike operational Dvorak estimates, which reflect changing analyst practice
and technology).

### 4.1 Why it matters here

1. **Potential source of true pattern labels.** The ADT algorithm internally determines a *scene
   type* (eye, embedded centre, central dense overcast, curved band, shear, and similar). If the
   distributed product exposes this field, it is **the scientifically correct label set for
   "cyclone pattern classification"** and directly satisfies the problem statement's wording.
   **TO VERIFY — this is the single highest-value verification task in Phase 3.**
2. **A satellite-only intensity benchmark.** Our CNN estimates intensity from imagery; ADT-HURSAT is
   an established algorithm doing the same thing. Comparing against it is far more meaningful than
   comparing against nothing.
3. **A QC signal.** Large disagreement between ADT-HURSAT intensity and IBTrACS best-track intensity
   at the same time flags a suspect record.

### 4.2 Cautions

- Coverage is expected to be **6-hourly and a subset of IBTrACS storms** — plan for a partial join,
  and never assume ADT rows exist for every fused sample. **TO VERIFY.**
- It is a *derived* product. If we train on ADT intensity and evaluate against ADT intensity, we are
  measuring agreement with an algorithm, not with nature. Best-track remains the ground truth.
- Column naming varies between releases. **TO VERIFY** against the actual file.

### 4.3 If ADT scene types are absent

Fall back to Tier A labels (intensity category) as the classification target, and describe the model
honestly as *intensity-stage classification from satellite imagery* rather than *Dvorak pattern
classification*. Plan the language of the methodology page for both outcomes so no rewrite is needed.

### 4.4 Join Quality-Control Gate

Before any model training, the fused dataset must pass these automated assertions. Failure blocks
Phase 5.

| # | Assertion | Threshold |
|---|---|---|
| 1 | HURSAT frame centre agrees with the IBTrACS position at the joined time | Great-circle separation below a small tolerance (target < 50 km) for > 99% of rows |
| 2 | One and only one image row per (storm, timestamp) | 100% — enforced by the dedup step |
| 3 | Joined timestamp offset within tolerance | \|Δt\| ≤ 90 min for 100% of rows |
| 4 | Wind and pressure inside physical ranges | 100% |
| 5 | No storm appears in more than one split | 100% |
| 6 | Monotonic non-duplicated timestamps within a storm | 100% |
| 7 | Image not all-NaN / not constant | > 99% |
| 8 | Class distribution reported and non-degenerate | Manual review |

### 4.5 Canonical intermediate format

**DECISION:** convert once from NetCDF into a consolidated store; never read raw NetCDF during
training.

- **Imagery → a single chunked array store** (Zarr, or HDF5 with one dataset per storm), keyed by
  `(sid, timestamp)`.
  - Resample to a fixed square grid — **target 224×224**, matching standard pretrained-CNN input and
    fitting comfortably in 6 GB VRAM.
  - Store infrared brightness temperature **quantised to `uint8`** over a fixed physical range
    (roughly 180–310 K, exact range **TO VERIFY** from the data histogram). This is a **4× size
    reduction versus float32** and the quantisation step is far finer than sensor noise, so no
    meaningful information is lost. The mapping constants are stored as array attributes so the
    physical values are always recoverable.
- **Tabular → Parquet**, partitioned by season, with a snappy-compressed columnar layout that DuckDB
  and pandas both read natively.
- **Manifests → JSON/CSV**, version-controlled: the split assignment, the dedup decisions, and the
  QC report for every build.

**Why not read NetCDF directly during training?** Tens of thousands of small-file opens per epoch on
Windows is pathologically slow, it defeats OS page caching, and OneDrive may attempt to sync or
dehydrate files mid-epoch. One conversion pass removes all three problems permanently.

---

## 5. Multi-Source Fusion

### 5.1 The join

```
IBTrACS row                          HURSAT file
  storm serial ID  ────── key 1 ──────  storm identifier  (needs a crosswalk, TO VERIFY)
  ISO timestamp    ────── key 2 ──────  observation time  (needs tolerance, see below)
  lat / lon        ────── QC only ────  frame centre lat / lon
```

**Key 1 — identity.** IBTrACS carries agency cross-reference IDs; HURSAT files are expected to be
named or attributed with a storm identifier and name. A **crosswalk table** must be built and
validated once. **TO VERIFY:** which identifier HURSAT actually exposes, and whether name+season+basin
is needed as a fallback. Name matching alone is unsafe — names are reused across seasons and basins.

**Key 2 — time.** Satellite scan times do not land exactly on synoptic hours (a scan starting at
17:45 belongs to the 18:00 slot). Therefore:

- Join **nearest-in-time within a ±90 minute tolerance**, preferring exact matches.
- **Rationale for ±90 min:** half of the 3-hourly image cadence. Wider risks assigning a frame to the
  wrong synoptic slot; narrower discards valid frames whose scan time is merely offset.
- Record the actual Δt on every fused row so the sensitivity of results to the tolerance can be
  tested later.
- **TO VERIFY:** the real distribution of scan-time offsets. Adjust the tolerance to the data.

**Key 3 — space.** Because HURSAT is already storm-centred, position is a **validation gate**, not a
join key (QC assertion 1 above). A frame whose centre disagrees with best track indicates a bad
crosswalk or a shifted centre-fixing, and is dropped.

### 5.2 Anatomy of one fused training sample

**Nothing below is assumed to exist until Phase 3 verification.** This is the target schema.

```
FusedSample
├── identity
│   ├── sid                  storm serial identifier
│   ├── timestamp            UTC, synoptic
│   ├── basin, season, name
│   └── step_index           position within the storm's sequence
├── image
│   ├── ir_window            224 × 224, uint8-quantised brightness temperature
│   ├── satellite_id         provenance of the frame
│   └── dt_offset_minutes    join Δt, retained for auditing
├── state (present)
│   ├── lat, lon
│   ├── wind, pressure
│   ├── storm_speed, storm_dir
│   ├── nature, category
│   └── dist2land
├── derived features (all strictly causal)
│   ├── d_wind_6h / 12h / 24h        intensity tendency
│   ├── d_pres_6h / 12h / 24h        pressure tendency
│   ├── d_lat_6h, d_lon_6h           recent motion
│   ├── max_wind_so_far              NEVER lifetime max
│   ├── hours_since_genesis
│   ├── doy_sin, doy_cos             seasonality without a January discontinuity
│   ├── abs_lat                      Coriolis / latitude effect proxy
│   └── ir_stats                     min BT, mean BT in radial rings, cold-cloud fraction, BT std
├── labels
│   ├── y_detect                     constructed, see requirements §2.A
│   ├── y_class                      Tier A / B / C — chosen after verification
│   ├── y_wind_{+6,+12,+18,+24}
│   └── y_dlat/y_dlon_{+6,+12,+18,+24}
└── provenance
    ├── split                        train / val / test
    └── build_version
```

### 5.3 Fusion depth — three levels, chosen deliberately

| Level | Description | Verdict |
|---|---|---|
| **L1 — Scalar fusion** | Extract handcrafted IR statistics (min BT, cold-cloud fraction, radial profile) and append to the tabular state vector | **MVP.** Cheap, interpretable, no coupling between vision and temporal training, and immediately usable by GBM baselines |
| **L2 — Embedding fusion** | Take the penultimate-layer embedding from the trained classification CNN, reduce it, and append to the sequence model input | **Stretch goal.** Genuinely "multi-source", still decoupled: the CNN is trained first, then frozen. Highest value-per-effort upgrade |
| **L3 — End-to-end joint** | CNN and sequence model trained jointly on image sequences | **Advanced only.** VRAM cost scales with sequence length × image size; on 6 GB this is not realistic |

**DECISION: build L1, design the interface so L2 is a config flag, defer L3.** The sequence model
takes a feature vector per timestep; whether the trailing dimensions are handcrafted statistics or a
CNN embedding is a data-layer concern the model never needs to know about.

### 5.4 Temporal sequence construction

```
Storm timeline (6-hourly synoptic steps)
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11
└───────── input window L=8 ─────────┘└── horizon H=4 ──┘
     └───────── next sample (stride 1) ─────────┘
```

- **Sliding window with stride 1** over each storm, maximising sample count.
- A window is emitted only if all `L + H` steps exist with a **contiguous 6-hour cadence**. Gaps are
  not interpolated — an interpolated target is a fabricated label.
- Windows **never cross storm boundaries**.
- Overlapping windows share timesteps, so **windows from one storm must all land in the same split**
  — this is exactly why the split is by storm ID.
- Storms shorter than `L + H` steps are dropped from the forecasting datasets (they remain available
  for detection and classification, which are per-frame tasks).
- **TO VERIFY:** the distribution of storm durations, which determines how much data survives the
  `L + H` requirement. If too little survives, reduce `L` — never `H`, since `H` is the deliverable.

---

## 6. Proposed Data Architecture

```
                    EXTERNAL SOURCES  (downloaded once, in Phase 3)
      ┌──────────────┬─────────────────────┬───────────────────┐
      │   IBTrACS    │     HURSAT-B1       │    ADT-HURSAT     │
      │   CSV/NetCDF │     NetCDF          │    CSV/NetCDF     │
      └──────┬───────┴──────────┬──────────┴─────────┬─────────┘
             │                  │                    │
   ══════════▼══════════════════▼════════════════════▼══════════
   RAW ZONE     $DATA_ROOT/raw/ — immutable, never edited, outside OneDrive
   ══════════┬══════════════════┬════════════════════┬══════════
             │                  │                    │
        parse+clean        decode+dedup         parse+align
        units, NA,         resample 224,        partial join
        agency choice      uint8 quantise
             │                  │                    │
   ══════════▼══════════════════▼════════════════════▼══════════
   INTERIM ZONE  $DATA_ROOT/interim/
     tracks.parquet        images.zarr           adt.parquet
     (one row per          (sid,time)->224x224   (partial coverage)
      storm-time)          uint8
   ══════════════════════════┬═════════════════════════════════
                             │  FUSION  (join keys 1+2, QC gate)
   ══════════════════════════▼═════════════════════════════════
   PROCESSED ZONE  $DATA_ROOT/processed/
     fused.parquet          + image index      + qc_report.json
   ══════════════════════════┬═════════════════════════════════
                             │  FEATURE ENGINEERING (causal only)
                             │  SPLIT (by storm / by season)
   ══════════════════════════▼═════════════════════════════════
   DATASET ZONE  $DATA_ROOT/datasets/<build_version>/
     detection/  classification/  intensity/  track/
     + splits.json  + scalers.pkl  + manifest.json
   ══════════════════════════┬═════════════════════════════════
                             ▼
                   MODEL TRAINING  (see ML_ARCHITECTURE.md)
```

**Zone rules**

1. **Raw is immutable.** Any bug is fixed by re-running a transform, never by editing raw files.
2. **Every zone transition is a single idempotent, re-runnable script.**
3. **`$DATA_ROOT` is an environment variable pointing outside OneDrive.** Non-negotiable — see the
   risk register.
4. **Each dataset build is versioned** and carries a manifest recording source versions, subsetting
   parameters, the agency wind column chosen, join tolerance, dedup rule, split seed, and QC results.
   Without this, no model comparison is trustworthy.

---

## 7. Format & Tooling Decisions

| Need | Choice | Why |
|---|---|---|
| NetCDF/HDF reading | `xarray` + `netCDF4` (+ `h5netcdf`) | Standard geoscience stack; labelled dimensions prevent axis-order bugs |
| Out-of-core conversion | `dask` **only if needed** | Chunked per-storm processing on 16 GB RAM likely suffices; do not add Dask pre-emptively |
| Image store | `zarr` (fallback: HDF5) | Chunked, compressed, parallel-readable, avoids the small-files problem |
| Tabular store | Parquet | Columnar, compressed, typed, universally readable |
| Analytical queries | DuckDB | Queries Parquet directly with zero ETL; ideal for label analysis and EDA |
| Geo operations (offline) | GeoPandas / Shapely | Track geometry construction, land intersection |
| Serving store | PostgreSQL + PostGIS | See SYSTEM_ARCHITECTURE.md §7 |

**Deliberately not used:** Spark, Hadoop, Airflow, Kafka, a feature-store service, or a data-versioning
service. At this data scale each would add operational burden without solving a problem we have.
A small set of numbered, idempotent Python scripts driven by a Makefile is the correct tool.

---

## 8. Consolidated TO-VERIFY Register

**Status: all 23 items checked against real data in Phase 1** (small verified samples — 2 IBTrACS
basin files, 3 HURSAT-B1 storms/195 frames across 3 seasons, 7 seasons of archive listings, 31 ADT
storms/1,727 records). Full methodology, evidence, and raw report JSON are in
[PHASE_1_DATASET_VERIFICATION.md](PHASE_1_DATASET_VERIFICATION.md). **Items marked ⛔ were blocking**
— every blocking item resolved favourably; none forced an architecture change beyond what is noted.

### IBTrACS

| # | Question | Status | Finding |
|---|---|---|---|
| 1 | ⛔ Exact wind/pressure column names, population rate per agency | ✅ **VERIFIED** | `USA_WIND`/`USA_PRES` are the columns. On usable NA synoptic rows (1980–2015, observed+main): wind present **99.76%**, pressure **79.51%**. NA carries **1** wind-averaging convention (1-min, US agencies); the North Indian file carries **4** (1-min/3-min/10-min mixed) — decisive evidence for the basin choice, see §Basin Decision below. |
| 2 | ⛔ Which column distinguishes observed vs interpolated rows | ✅ **VERIFIED** | `IFLAG` (15-char per-agency string; char 1 = USA agency: `O`=original, `P`=position interpolated, `I`=intensity interpolated, `V`=partial, `_`=missing) combined with `TRACK_TYPE` (`main`/`spur-*`/`PROVISIONAL`). At synoptic hours, **99.99%** of NA rows are already `IFLAG[0]=='O' & TRACK_TYPE=='main'** — the observed-only filter costs almost nothing. |
| 3 | Exact CSV header/units-row structure, missing-value sentinels | ✅ **VERIFIED — plus one CRITICAL bug found and fixed** | Row 2 is a units row as expected; skip it explicitly. **Sentinels in v04r01 are blank cells, not `-999`** (the Phase 0 assumption was over-cautious but harmless to keep). **Critical:** `pandas.read_csv` with default NA handling parses the *basin code* `"NA"` (North Atlantic) as `NaN` — this silently corrupted 126,586/127,188 rows (99.5%) on first run. Fixed by passing `keep_default_na=False` with an explicit sentinel list. This is now a standing rule for all IBTrACS loading code. |
| 4 | Longitude convention, antimeridian behaviour | ⚠️ **PARTIALLY VERIFIED** | Convention confirmed as −180..180 (NA range −136.9..63.0; NI range −87.7..163.7). **Neither sample basin crosses ±180°**, so the antimeridian-splitting code path itself remains untested — carry forward as NOT YET VERIFIED and test explicitly against a Western/Central Pacific storm before shipping the map layer. |
| 5 | Storm-duration distribution in 6-hourly steps | ✅ **VERIFIED** | NA: median 24 steps, p90 = 52, **84.6%** of storms have ≥12 steps (the `L=8,H=4` requirement). NI: median 15, p90 = 32, only 66.5% clear the same bar. Confirms `L=8` is usable for NA without reduction. |
| 6 | Category/nature field values and distribution | ✅ **VERIFIED** | `NATURE` values confirmed: `TS, ET, DS, SS, MX, NR, DB`. `USA_SSHS` confirmed range −5..5. On the NA 1980–2015 usable subset: TS 7,558 · ET 1,259 · DS 1,121 · SS 216 rows — see §21. |
| 7 | Population rate of distance-to-land / wind-radii fields | ⚠️ **PARTIALLY VERIFIED** | `DIST2LAND` is **100%** populated (measured, NA). Wind radii are sparse as expected: `USA_R34_NE` 13.9%, `USA_R50_NE` 7.2%, `USA_R64_NE` 4.1% — confirms these are Advanced-scope-only features, not MVP-usable. |

### HURSAT-B1

| # | Question | Status | Finding |
|---|---|---|---|
| 8 | ⛔ Storm identifier exposed, clean IBTrACS crosswalk | ✅ **VERIFIED — the single most important result of Phase 1** | The IBTrACS SID is embedded **three times per file** (filename token, global attribute `TC_serial_number`, and a `sid` data variable) and all three agree on **100%** of 195 sampled frames. Across a 7-season, 745-archive listing scan, **100% of HURSAT filenames carry a well-formed IBTrACS SID**, zero malformed, zero duplicates. The crosswalk is not a heuristic — it is a direct, exact key. |
| 9 | ⛔ On-disk size of the chosen basin/period subset | ✅ **VERIFIED (measured + extrapolated)** | Measured NA archive sizes (7 sampled seasons): mean **26.3 MB/storm** (compressed), rising from ~6 MB/storm (1985) to ~56 MB/storm (2010) as channel coverage improved. **Estimated NA 1980–2015 full subset: ~13.8 GB compressed download, ~24 GB peak extracted, collapsing to well under 1 GB after conversion to quantised Zarr** (raw deleted post-conversion). Comfortably fits the 639 GB free disk. |
| 10 | ⛔ Field to choose between simultaneous satellite views | ✅ **VERIFIED** | `VZA` (view zenith angle) is present on **100%** of sampled frames. **78.9%** of (storm, timestamp) pairs in the sample had 2 simultaneous satellite views (max observed: 2); the documented dedup rule (`min(VZA)`) resolved every case (195 → 109 frames after dedup, 0 remaining collisions). |
| 11 | Exact channel names, grid size, resolution, projection | ✅ **VERIFIED** | Grid is **301×301** on **100%** of sampled frames, projection = Mercator. Channels confirmed: `IRWIN` (10.3–11.0 µm, **100%** of frames), `IRWVP` (6.45–7.02 µm), `IRNIR` (3.79–4.04 µm), `IRSPL` (11.6–12.5 µm), `VSCHN` (0.53–0.77 µm visible), plus variability channels. **Channel completeness varies by era/satellite**: only 58% of sampled frames carry the full 7-channel set; older satellites (GOES-7/8) carry as few as `IRWIN` alone. MVP's IR-window-only decision is therefore also the *safest* choice for coverage, not only for the day/night-artefact reason in §3.1. |
| 12 | Brightness-temperature range and units | ✅ **VERIFIED** | Units = Kelvin (confirmed via variable metadata). Documented fill value −1.0 was never observed exactly in the sample (0.0000%); **~0.05% of pixels fall in an unphysical 0–150 K band** (edge/interpolation artefacts) — these must be masked by a `<150 K` physical floor, not by fill-value equality alone. Physical range measured: p0=190.5 K … p100=318.0 K. **Native data is already quantised at a ~0.4–0.7 K step** (118–180 distinct values per frame) — confirms the planned `uint8` over [180, 320] K (0.549 K/level) is lossless relative to source precision. |
| 13 | Distribution of scan-time offsets from synoptic hours | ✅ **VERIFIED — better than assumed** | HURSAT-B1's `htime` is a **nominal**, already-synchronised slot: measured offset to IBTrACS `ISO_TIME` was **exactly 0.0 minutes for 100% of the 109 deduplicated sample frames** (median = mean = max = 0.0). The ±90 min tolerance is not merely safe, it is generous for HURSAT-B1 itself. (ADT-HURSAT, which records true scan times, does show a realistic offset distribution — see #17/#19.) |
| 14 | Missing-data / fill-value / edge patterns | ✅ **VERIFIED** | `IRWIN` valid-pixel fraction: mean **99.93%**, minimum **96.26%** across the sample; **zero** constant/empty frames (0/195). Missing pixels are non-finite at grid edges beyond the storm-centred extent, as expected for a fixed-radius re-projected grid. |
| 15 | Temporal coverage vs geostationary era | ✅ **VERIFIED** | HURSAT-B1 v06 confirmed 1978–2015 (NCEI product page). Sample satellites span GOES-7/8 (1990s) through GOES-15/Meteosat-10 (2015) — full multi-satellite geostationary constellation is represented across the archive. |

### ADT-HURSAT

| # | Question | Status | Finding |
|---|---|---|---|
| 16 | ⛔ **Does the release expose an ADT scene-type / pattern field?** | ✅ **VERIFIED — YES.** | **This resolves the single highest-value open question from Phase 0.** ADT-HURSAT contains a `Scene` field (string) derived from `EyeScene` (Eye / Pinhole Eye / Large Eye / No Eye) × `CloudScene` (CDO / Embedded Center / Irregular CDO / Curved Band / Shear) — the exact Dvorak taxonomy Tier B specified. On a 2005 NA-season sample (31 storms, 1,727 records): `CurvedBand` 31.0% · `Shear` 27.6% · `CDO` 17.1% · `Land` 12.9% · `Eye` 6.0% · `IrrCDO` 2.3% · `EmbCenter` 1.9% · `LargeEye` 0.75% · `PinholeEye` 0.29%. **Tier B pattern classification is promoted from "if available" to the recommended headline classification target** — see §Architecture Changes in the Phase 1 report. Severe class imbalance (107× max/min) must be handled per ML_ARCHITECTURE.md §5.3. |
| 17 | Fraction of fused samples with an ADT record | ✅ **VERIFIED** | **100%** of the 31 sampled 2005 NA storms had an ADT file (files are named directly by IBTrACS SID, e.g. `2005236N23285.nc`). Record density is lower than IBTrACS synoptic cadence (mean 55.7 ADT records/storm, at *true* scan times, not the 6-hourly grid) — a temporal join is required, not a direct row match. |
| 18 | Column names, units, time convention | ✅ **VERIFIED** | Full 38-variable schema extracted (`CI`, `FinalT`, `Scene`, `EyeScene`, `CloudScene`, `WindSpeed`, `MSLP`, `EyeT`, `CloudT`, `Lat`, `Lon`, `VZA`, `RMW`, and Dvorak-rule diagnostic flags). Time is `Date` (`YYYYMonDD`) + `Time` (`HHMMSS`), both string-typed; parses cleanly. |
| 19 | Magnitude of ADT-vs-best-track intensity disagreement | ✅ **VERIFIED** | On 1,504 matched records (2005 NA sample, ±90 min join): mean bias **−6.24 kt** (ADT reads low vs. `USA_WIND`), MAE **11.3 kt**, correlation **r = 0.87**. Confirms NCEI's own caveat that ADT-HURSAT "should not be used to determine actual storm intensities" — it is usable as a QC cross-check and a structural-label source, never as an intensity ground truth, exactly as Phase 0 planned. |

### Cross-cutting

| # | Question | Status | Finding |
|---|---|---|---|
| 20 | ⛔ Final joined sample count per task | ✅ **VERIFIED (measured funnel + estimated fusion step)** | NA, 1980–2015, exact from IBTrACS: 29,995 raw rows → 15,116 usable (observed, synoptic, wind-labelled) rows, 547 storms → **9,398 sequence windows** (`L=8,H=4`) from 445 storms for the IBTrACS-only track/intensity path (P2). With imagery fused (measured 96% HURSAT coverage rate applied): **≈14,500 fused frames, ≈9,000 windows with imagery**. This comfortably clears the "too few samples" failure mode — sufficient for the recommended GBM baselines and a small pretrained CNN/GRU, though still modest for training a large vision model from scratch (reinforces the Phase 0 ResNet-18-pretrained recommendation). |
| 21 | Class distribution of the chosen label set | ✅ **VERIFIED for both candidate tiers** | **Tier A** (`USA_SSHS`, NA usable rows, 1980–2015): TD 1,689 · TS 3,457 · Cat1 1,305 · Cat2 458 · Cat3 300 · Cat4 306 · Cat5 43 · non-tropical stages (−2..−4) 2,596 rows. Cat5 is critically rare (43 rows) — the pre-declared class-merge rule (ML_ARCHITECTURE.md §5.1) will trigger. **Tier B** (ADT `Scene`): see #16, 107× imbalance. Both tiers need the layered imbalance strategy already specified; neither is a surprise. |
| 22 | Licensing/attribution requirements | ✅ **VERIFIED** | All three sources are NOAA/NCEI or NOAA-affiliated (UW–CIMSS) US Government works; attribution requested, not legally required. Citations captured in `ml/manifests/datasets.json`: Knapp et al. 2010 (IBTrACS), Knapp & Kossin 2007 (HURSAT-B1), Kossin et al. 2020 *PNAS* (ADT-HURSAT). ADT-HURSAT's NCEI caveat (#19) is recorded verbatim for the methodology page. |
| 23 | Download size and time for the chosen subset | ✅ **VERIFIED (measured throughput + estimate)** | IBTrACS NA (57 MB) and NI (28 MB) downloaded in well under a minute each on this connection. Sample HURSAT archives (6–24 MB each) downloaded in seconds. At the demonstrated throughput, the estimated 13.8 GB NA 1980–2015 subset (#9) is a low-risk, low-single-digit-hours download — not a Phase 4 schedule threat. |

---

## 9. Summary of Locked Decisions

1. IBTrACS is the spine; HURSAT is the eyes; ADT is an optional interpreter and cross-check.
2. Track and intensity models are built **first**, from IBTrACS alone, with no satellite dependency.
3. Supervised learning uses **6-hourly synoptic observed rows only**.
4. **One agency's wind column** for the whole training set; no cross-agency fallback.
5. Join on identity + nearest time within **±90 min**; position agreement is a QC gate.
6. Deduplicate to **one frame per (storm, timestamp)** before splitting.
7. Convert NetCDF once into **Zarr (uint8, 224×224) + Parquet**; never read raw NetCDF in a training loop.
8. Split **by storm**, preferably **by season**; manifests are version-controlled.
9. All engineered features are **strictly causal**.
10. Fusion is **L1 (scalar)** for MVP, with L2 (embedding) behind a config flag.
11. `$DATA_ROOT` lives **outside OneDrive**.

### 9.1 Phase 1 additions to the locked-decision set

12. **MVP basin = North Atlantic (NA), confirmed by evidence, not by default carry-over.**
    Measured on real data: NA has **1** wind-averaging convention present vs **4** in the North
    Indian comparison basin; NA storm-duration coverage for `L=8,H=4` is 84.6% vs 66.5% for NI;
    NA HURSAT coverage in the sample was 93–100% per season vs 27–71% for NI. See
    [PHASE_1_DATASET_VERIFICATION.md §Basin Decision](PHASE_1_DATASET_VERIFICATION.md) for the
    full comparison. **This was a Phase 1 verification task, not a Phase 0 assumption — it is
    now evidence-backed and ready for explicit user sign-off before Phase 2.**
13. **IBTrACS CSV loading MUST use `keep_default_na=False`** with an explicit sentinel list.
    Pandas' default NA-value list contains the literal string `"NA"`, which collided with the
    North Atlantic basin code and silently corrupted 99.5% of rows on first load. This is now a
    standing rule for every IBTrACS reader in the codebase, not a one-off fix.
14. **ADT-HURSAT `Scene` is promoted to the recommended primary classification target**, ahead of
    the Tier A intensity-category fallback, now that TO-VERIFY #16 has confirmed the field exists
    with a usable (if imbalanced) class distribution. Tier A remains the guaranteed fallback if
    ADT coverage proves too sparse once the full fusion join is built in Phase 4.
15. **IRWIN validity masking uses a `< 150 K` physical floor**, not fill-value (`-1.0`) equality
    alone — Phase 1 measured a small population (~0.05%) of unphysical near-zero pixels that the
    documented sentinel does not catch.
