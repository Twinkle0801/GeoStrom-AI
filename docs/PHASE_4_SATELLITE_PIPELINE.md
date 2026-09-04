# Phase 4 — Satellite Data Pipeline

**Status:** Pipeline implementation COMPLETE and verified end-to-end on real data at two
scales: the original 3-storm/109-sample Phase 1 verification set, and an expanded, real,
downloaded 12-storm/627-sample multi-split dataset (QC gate PASS at both scales).
Full MVP-scale (all 531 available NA 1980–2015 archive storms) processing NOT performed —
see §16/§19 for the measured reason and the exact resume commands.

---

## 1. Objective

Convert the Phase 1 verified HURSAT-B1 + ADT-HURSAT + IBTrACS sources into a clean,
reproducible, ML-ready satellite sample index (Parquet) plus a canonical imagery store
(Zarr). This phase prepares data only — it does not train, classify, or detect anything.
Nothing in `backend/`, `frontend/`, or the Phase 2 IBTrACS-only baselines was modified.

Pipeline implemented, matching the task's required stages exactly:

```
RAW SATELLITE DATA (HURSAT-B1 .tar.gz)
        |
HURSAT ingestion (discover + extract)          ml/geostrom_ml/satellite/download.py
        |
metadata extraction                            ml/geostrom_ml/satellite/hursat.py
        |
IRWIN quality control (<150K / >350K physical floor)
        |
spatial normalization (native 301x301 grid preserved -- see §9)
        |
temporal alignment + IBTrACS join               ml/geostrom_ml/satellite/alignment.py
        |
ADT-HURSAT Scene join                            ml/geostrom_ml/satellite/adt.py
        |
duplicate-frame resolution (VZA-based)           ml/geostrom_ml/satellite/dedup.py
        |
QC gate (18-point report)                        ml/geostrom_ml/satellite/qc.py
        |
dataset manifest                                 ml/geostrom_ml/satellite/manifest.py
        |
Zarr (imagery) + Parquet (sample metadata index)  ml/geostrom_ml/satellite/imagery.py
```

Orchestration: `ml/geostrom_ml/satellite/pipeline.py::run_pipeline`, invoked by
`ml/scripts/build_satellite_dataset.py`.

---

## 2. Data Sources

| Source | Version | Role | Never used for |
|---|---|---|---|
| IBTrACS | v04r01 | Best-track spine; `USA_WIND` (1-minute convention) is the only intensity ground truth | -- |
| HURSAT-B1 | v06 | Storm-centric IR imagery, IRWIN channel, native 301x301 grid | -- |
| ADT-HURSAT | ADT v9.0 over HURSAT V07b (NCEI Accession 0307249, v1.1) | `Scene` classification label only | Intensity ground truth (NCEI's own caveat: "should not be used to determine actual storm intensities") |

No new data source was introduced. `keep_default_na=False` and the `USA_WIND`-only,
no-cross-agency-fallback rule (Phase 1/2) are unchanged and untouched by this phase.

---

## 3. Architecture Decision: Reuse vs. New Code

Per the task's explicit reuse instruction, three modules consolidate already-validated
Phase 1 logic into importable library code rather than re-deriving it (same precedent as
`ml/geostrom_ml/data/ibtracs.py` in Phase 2):

| New module | Consolidates | Historical script (untouched) |
|---|---|---|
| `satellite/discovery.py` | Season listing fetch/parse/cache | `ml/scripts/verify_crosswalk.py` |
| `satellite/hursat.py` | NetCDF metadata + IRWIN extraction | `ml/scripts/verify_hursat_join.py`, `ml/scripts/qc_gate.py` |
| `satellite/adt.py` | ADT NetCDF parsing + time format | `ml/scripts/verify_adt.py` |

`satellite/alignment.py` imports `haversine_km` directly from
`ml/geostrom_ml/features/geo.py` (Phase 2's validated implementation) — this module lives
inside `ml/`, so, unlike `backend/`, there is no "never import from ml/" boundary to
respect; a third re-implementation would be pure duplication.

**Directory placement note.** `docs/SYSTEM_ARCHITECTURE.md`'s original Phase 0 sketch
placed satellite ingestion under `geostrom_ml/data/`. Phase 2 had already repurposed that
module for IBTrACS-only loading (`ibtracs.py`), so Phase 4 code lives in a new
`geostrom_ml/satellite/` package instead of overloading `data/` with two unrelated
concerns — a minimal, documented deviation from the Phase 0 sketch, not a redesign of the
storage/service architecture (still Zarr + Parquet, still a `ml/scripts/*.py` CLI
convention).

---

## 4. HURSAT-B1 Pipeline

- **Discovery** (`satellite/discovery.py`): fetches real NCEI season directory listings
  (small HTML, no imagery), cached to `$DATA_ROOT/raw/hursat_listings/`.
- **Download** (`satellite/download.py`): idempotent per-storm downloader. Storms are
  selected FROM the frozen Phase 2 split manifest only (`--storm-ids`, `--season`, or
  `--sample-storms N` with a deterministic seeded stratified sample) — never an arbitrary
  IBTrACS storm, so every downloaded storm is guaranteed split-compatible.
- **Extraction**: `.tar.gz` -> `$DATA_ROOT/interim/hursat/<archive-name>/*.nc`, with a
  path-traversal guard on archive members.
- **Metadata extraction** (`satellite/hursat.py::parse_frame_metadata`): identity
  (`TC_serial_number`), satellite name, `htime`, VZA, centre lat/lon, channel list, IRWIN
  valid-pixel percentage. A two-pass design (metadata-only inventory, then a second,
  separate read for pixel arrays) keeps memory bounded at MVP scale.
- **IRWIN physical QC**: `IRWIN_VALID_RANGE_K = (150.0, 350.0)`, reused verbatim from
  `ml/scripts/qc_gate.py::IRWIN_PHYSICAL_K` — a real physical floor/ceiling, not fill-value
  (`-1.0`) equality. A pixel outside this range (whether the `-1.0` sentinel, a NaN, or a
  physically-impossible value like 500 K) is masked invalid via a **separate boolean
  `valid_mask`**, never inferred from the value alone. See `test_satellite_hursat.py`.

---

## 5. Duplicate-Frame Resolution

Phase 1 found simultaneous multi-satellite views at the same `(storm_id, timestamp)`.
`satellite/dedup.py` resolves each group deterministically:

1. lowest VZA (most direct view) wins;
2. ties, or all-missing VZA, fall back to lowest `source_file` path string.

Every non-selected candidate is logged with a machine-readable `rejection_reason`
(`duplicate_frame_higher_vza` or `duplicate_frame_no_vza_tiebreak_filename`). Selection is
proven reproducible under randomised input row order (`test_satellite_dedup.py`).

---

## 6. Temporal Alignment + Spatial QC (IBTrACS join)

`satellite/alignment.py::join_frames_to_ibtracs` joins each canonical frame to the
**nearest IBTrACS row within an explicit, configurable tolerance** (default 90 min, the
Phase 1 verified value), against the **full** per-storm track (every `IFLAG`/`TRACK_TYPE`
value) — not the Phase 2 "usable rows only" subset. This preserves the
observed/interpolated/missing distinction the task requires: `is_observed` is `True` only
when the matched row's primary-agency `IFLAG` character is `'O'`; any other value (e.g.
`'I'`, interpolated) sets `is_interpolated = True`. A frame that matches only a
non-original row is still recorded, never discarded for that reason alone.

Spatial QC gate: `SPATIAL_QC_KM = 50.0` (locked, Phase 1 value; the task explicitly
forbids loosening it). Every join failure — no IBTrACS row in tolerance, storm unknown to
IBTrACS, or spatial separation >= 50 km — sets `qc_status = "rejected"` with a
machine-readable `qc_reason`; nothing is silently dropped.

---

## 7. ADT-HURSAT Integration

`satellite/adt.py::join_adt_scene` attaches `scene_label` + `adt_timestamp` on the nearest
true ADT scan time within tolerance. **Never blocking**: a sample with no ADT match keeps
its (IBTrACS-sourced) intensity fields and is simply `adt_qc_status = "unmatched"`. The
join function never writes `usa_wind` or any field Phase 2/3 treat as intensity ground
truth — enforced by `test_satellite_adt.py::test_never_writes_an_intensity_field`.

---

## 8. Sample Schema

`ml/geostrom_ml/satellite/schema.py::SAMPLE_COLUMNS` — the task's required minimum
schema plus four justified additions (`split`, `season`, `zarr_index`,
`preprocessing_version`; rationale documented in the module docstring). No meteorological
feature beyond identity/QC fields is duplicated into the image dataset.

---

## 9. Storage Architecture: Zarr Layout (a locked "TO VERIFY" resolved)

`docs/DATA_STRATEGY.md` §4.5 (Phase 0 planning) marked the canonical resolution and
quantization range as **ASSUMPTION / TO VERIFY** ("target 224x224... exact range TO VERIFY
from the data histogram"). Phase 4 resolves both, and **revises the resize assumption**
with a documented reason:

- **Resolution: native 301x301, unresized.** Downsampling to 224x224 requires either
  interpolation (alters physical pixel values) or an asymmetric crop (discards
  field-of-view) — both are transformations a future model-input pipeline can apply
  deterministically FROM the canonical grid, but neither belongs in canonicalization
  itself per this task's explicit "preserve raw/physical values in the canonical dataset"
  instruction. Storage cost at 301x301 remains modest (§13).
- **Quantization range: 150-350 K**, not the placeholder 180-310 K — this is the same
  physically-verified IRWIN floor/ceiling already locked in Phase 1, reused rather than
  re-derived.
- **Values**: canonical `irwin_k` array is float32 Kelvin, invalid pixels NaN, governed by
  a separate `valid_mask` boolean array. A companion `irwin_u8` array stores a
  deterministic, fully documented linear quantization of the same range for compact
  storage (`quantize_irwin` / `dequantize_irwin`, round-trip error <= 1 quantization step,
  see `test_satellite_imagery.py`). `irwin_u8` is a convenience copy; `irwin_k` is the
  source of truth.
- **Chunking**: `(32, 301, 301)` per array (~11.6 MB/chunk for float32) — large enough to
  avoid a small-files problem at MVP scale, small enough that reading one training batch
  does not pull unrelated frames.

This is a refinement of a placeholder the Phase 0 docs explicitly deferred to this phase,
not a redesign of the storage architecture (still Zarr + Parquet, still `$DATA_ROOT`,
still raw NetCDF deleted-after-conversion-eligible).

---

## 10. Split Strategy

**No new split was created.** `satellite/pipeline.py::load_split_map` reads
`ml/manifests/splits_v1.json` (Phase 2, frozen) and looks up each fused sample's
`storm_id`. A storm absent from that manifest is excluded from the final index (logged as
`split_rejected`, reason `storm_not_in_frozen_split_manifest`) rather than silently
assigned a split — this is QC gate check **Q2**, which by construction can never fail on a
correctly-behaving pipeline (see `test_satellite_leakage.py`).

---

## 11. Leakage Controls

`ml/tests/test_satellite_leakage.py` covers all six vectors the task names:

| # | Vector | Protection | Test |
|---|---|---|---|
| 1-2 | Frame time > label time / future IBTrACS leaking in | Explicit, bounded temporal tolerance; matches beyond it are rejected, not silently accepted | `TestVector1And2TemporalOffsetBounded` |
| 3 | Future ADT scene leaking in | Same tolerance mechanism, explicit parameter (not hard-coded/unbounded) | `TestVector3AdtNeverExceedsTolerance` |
| 4, 6 | Duplicate/representation crossing splits | Split is looked up from the frozen manifest, never recomputed | `TestVector4And6SplitIntegrity` |
| 5 | Same physical frame appearing twice | Dedup guarantees <= 1 canonical frame per `(storm_id, timestamp)`; QC gate check Q1 independently re-verifies on the final index | `TestVector5NoDuplicatePhysicalFrame` |

---

## 12. QC Gate (18-point report)

`ml/geostrom_ml/satellite/qc.py::build_qc_report` emits exactly the 18 counts the task
requires (files discovered/parsed/rejected, frames discovered/valid-IRWIN, duplicate/
unique frames, successful/failed IBTrACS joins, spatial/temporal QC failures, ADT matched/
unmatched, final fused samples, Scene class distribution, missing-value statistics,
per-season counts, per-storm counts) plus 5 blocking/informational checks (Q1-Q5:
no duplicate `(storm_id, timestamp)` in the final index; every storm belongs to the frozen
split; no sample exceeds the 50 km spatial gate; every final row has `qc_status == "ok"`;
zero parse errors). A blocking failure fails the gate (`gate_status: "FAIL"`, non-zero
script exit) — see `test_satellite_leakage.py::TestVector5...test_a_pipeline_that_skipped_dedup_would_be_caught_by_qc_gate_check_Q1`
for proof the gate is not vacuous.

**Image Quality Report** (the task's separate, explicit "Image Quality Report" section):
`ml/scripts/satellite_image_quality_report.py` complements the QC gate with what pass/fail
counts alone do not show — reads the already-built Parquet + Zarr and adds: full
distribution statistics (min/max/mean/median/p95, not just counts) for temporal offset and
spatial separation; a real IRWIN pixel-value histogram sampled from the canonical Zarr
store; invalid-pixel-fraction statistics; and rendered sample thumbnails
(`ml/reports/figures/satellite_sample_thumbnails.png`) plus a 3-panel distributions figure
(`ml/reports/figures/satellite_distributions.png`). Writes
`ml/reports/satellite_image_quality.json`. This script is a thin reporting layer over
already-tested pipeline output (`alignment.py`'s columns, the canonical Zarr) and is not
separately unit-tested — documented here, not hidden.

---

## 13. Dataset Manifest

`ml/geostrom_ml/satellite/manifest.py::build_manifest`, written to
`ml/manifests/satellite_dataset_v1_manifest.json` (committed, small). Contains dataset/
preprocessing version, source dataset versions, basin/seasons/storm selection, the full QC
counts, QC thresholds, scene-labeled sample count, gate status, and **DATA_ROOT-relative**
(never absolute/machine-specific) storage paths — enforced by
`test_satellite_manifest.py::test_storage_paths_are_relative_to_data_root_not_absolute`.

---

## 14. Reproducibility

Given the same files on disk, the same `splits_v1.json`, and the same configuration, the
pipeline is deterministic end to end: sample selection (dedup tie-breaks), `sample_id`
generation, `zarr_index` assignment (sorted `(storm_id, satellite_timestamp)` order), and
QC outcomes never depend on randomness, dict/set iteration order, or wall-clock time.
Proven on real data by `test_satellite_pipeline_integration.py::test_pipeline_is_reproducible_on_a_second_run`
(independent second run of the real 3-storm sample; identical `sample_id` sets).

---

## 15. Testing

| File | Covers |
|---|---|
| `test_satellite_hursat.py` | NetCDF parsing, IRWIN physical QC, grid-shape assertion, filename regex, discovery |
| `test_satellite_dedup.py` | VZA selection, deterministic tie-break, reproducibility under shuffled input |
| `test_satellite_alignment.py` | Temporal join, observed/interpolated classification, spatial QC gate, tolerance rejection |
| `test_satellite_adt.py` | ADT parsing/time format, Scene join, non-blocking on missing ADT, never writes intensity fields |
| `test_satellite_imagery.py` | Quantization round-trip, Zarr write/read round-trip, chunking, store metadata |
| `test_satellite_manifest.py` | Manifest content, DATA_ROOT-relative paths, split-source provenance |
| `test_satellite_leakage.py` | All six named leakage vectors, adversarial (proves protections are not vacuous) |
| `test_satellite_data_safety.py` | DATA_ROOT safety, `.gitignore` coverage of every new artifact type |
| `test_satellite_pipeline_integration.py` | Real 3-storm Phase 1 sample, full pipeline, reproducibility (skips cleanly if sample data absent) |

Synthetic fixtures (`ml/tests/conftest.py`: `synthetic_hursat_nc`, `synthetic_ibtracs_full_track`)
are used everywhere a real HURSAT file is unnecessary, per the task's explicit instruction.

---

## 16. Real, Measured Validation Results

### 16.1 Small validation: the original Phase 1 sample (3 storms, no download)

Ran for real against the actual Phase 1 verification sample (3 storms, 195 real HURSAT-B1
frames: Gabrielle 1995 [train], Katrina 2005 [val], Claudette 2015 [test] — real files
already on this workstation, no new download needed for this run):

| Metric | Value |
|---|---|
| Frames discovered / parsed | 195 / 195 (0 parse errors) |
| Canonical frames after dedup | 109 (86 rejected duplicates) |
| IBTrACS join success | 109/109 (100%) |
| Spatial QC failures | 0 |
| Temporal QC failures | 0 |
| Final fused samples | 109 |
| QC gate status | PASS (5/5 checks) |
| Reproducibility (2nd independent run) | Identical `sample_id` set |

These 195 -> 109 numbers are an **exact match** to Phase 1's own
`ml/reports/hursat_join_verification.json` finding (109 deduplicated frames from the same
sample) — independent confirmation the pipeline reproduces, rather than silently
recomputes differently from, the already-verified Phase 1 result.

**Measured I/O cost** (real, on this workstation): opening one real HURSAT NetCDF file via
`xarray.open_dataset` costs **~1.8-2.0 seconds**, regardless of xarray backend engine
tested (h5netcdf could not be benchmarked -- `h5py` is not installed -- but the default
netCDF4-engine cost was consistent and reproducible across every file measured). The
pipeline opens each surviving frame twice (once for metadata, once for pixels), so
wall-clock cost is roughly `(frames_discovered + frames_in_final_index) x 1.9s`. For this
195-frame/109-sample run: measured **~6.5 min** for the metadata pass, consistent with the
formula.

### 16.2 Expanded validation: 12 storms, real download, all three splits

Beyond the 3-storm sample, this phase additionally: (a) ran real archive discovery across
**all 36 seasons** (1980-2015, not a 7-season sample) via
`ml/scripts/discover_hursat_archive.py`, finding **531/547 (97.07%)** frozen-split storms
have a HURSAT-B1 archive, totalling **~11.6 GB** for full coverage (refines Phase 1's
96%/13.8 GB extrapolation with a real, complete census — `ml/reports/hursat_archive_discovery.json`);
and (b) downloaded and processed a real, deterministic, seeded (`seed=42`) stratified
sample of **9 additional storms** (`ml/scripts/download_hursat_sample.py --sample-storms 9`,
proportional across train/val/test) on top of the original 3, then ran the full production
build (`ml/scripts/build_satellite_dataset.py`) over all **12 storms / 1,097 raw frames**:

| Metric | Value |
|---|---|
| Files discovered / parsed | 1,097 / 1,097 (0 parse errors) |
| Unique frames after dedup | 627 (from 1,097 raw, across 470 duplicate-bearing groups) |
| IBTrACS join success | 627/627 (100%) |
| Spatial QC failures | 0 (max separation observed: 48.88 km, under the 50 km gate) |
| Temporal QC failures | 0 (offset was exactly 0 minutes for all 627 samples) |
| ADT-HURSAT Scene matches | 627/627 (100%) |
| **Final fused samples** | **627** |
| QC gate status | **PASS (5/5 checks)** |
| Split breakdown | train 404 (7 storms) / val 128 (2 storms) / test 95 (3 storms) |
| Scene class distribution | CDO 177, CurvedBand 154, Land 89, Shear 83, Eye 82, EmbCenter 17, LargeEye 14, IrrCDO 11 |
| `pressure_if_valid` missing | 19/627 (matches IBTrACS's ~94% USA_PRES coverage) |
| IRWIN pixel value (sampled 50 frames, 4.5M pixels) | mean 272.8 K, std 24.2 K |
| Invalid-pixel fraction per frame | mean 0.3%, max 4.7% |

Sample count is genuinely a small dataset for CNN training (not this phase's job), but is
large enough to exercise every pipeline stage at real, non-trivial, multi-storm,
multi-decade, multi-split scale — not just the original 3-storm toy case. Full reports:
`ml/reports/satellite_qc_gate.json`, `ml/reports/satellite_image_quality.json`,
`ml/manifests/satellite_dataset_v1_manifest.json`, and rendered figures at
`ml/reports/figures/satellite_sample_thumbnails.png` /
`ml/reports/figures/satellite_distributions.png`.

**Measured wall-clock for this run**: ~46 minutes total (12:27:46-13:13, timestamps from
the actual run), consistent with the §16.1 per-file formula scaled to 1,097 discovered +
627 written frames.

---

## 17. Frontend / API / Backend

**Not touched.** Phase 4 is a data-preparation phase only. No new frontend page, no new
API route, no backend schema change. `docs/SYSTEM_ARCHITECTURE.md`'s eventual `image_key`
field on the `predictions`/observations table (Phase 3's DB) is unpopulated — wiring
satellite imagery into the API/map is explicitly out of this phase's scope.

---

## 18. How to Run Locally

```bash
# 1. Determine real archive coverage before downloading anything (small HTML listings only)
python ml/scripts/discover_hursat_archive.py --seasons 1980 2015 --basin NA

# 2. Download a bounded, real sample (idempotent; --dry-run to preview first)
python ml/scripts/download_hursat_sample.py --dry-run --sample-storms 20
python ml/scripts/download_hursat_sample.py --sample-storms 20 --seed 42
# or: --storm-ids <SID> [<SID> ...] / --season <YEAR>

# 3. Build the dataset (Zarr + Parquet + manifest + QC report)
python ml/scripts/build_satellite_dataset.py --dry-run   # counts only, writes nothing
python ml/scripts/build_satellite_dataset.py

# 4. Inspect the QC gate
python ml/scripts/satellite_qc_gate.py

# 5. Image quality report (sample thumbnails, IRWIN histogram, offset/separation
#    distributions, invalid-pixel stats) -- reads the Parquet + Zarr from step 3
python ml/scripts/satellite_image_quality_report.py

# Tests
pip install -r ml/requirements.txt -r ml/requirements-satellite.txt
pytest ml/tests/test_satellite_*.py -v
```

---

## 19. Known Limitations

- **Full 1980-2015 NA archive (531 storms confirmed available by real discovery, §16.2;
  ~14,500 estimated fused frames per the Phase 1 extrapolation) was NOT downloaded or
  processed — only a real, deterministic 12-storm/627-sample subset was.** Measured
  wall-clock across TWO real runs at different scales (195 frames/109 samples in ~10.6
  min; 1,097 frames/627 samples in ~46 min) gives a calibrated rate of ~1.6s per
  file-open-equivalent, projecting to **roughly 18 hours** of wall-clock for the full
  archive (~25,961 raw + ~14,511 final frames, Phase 1's extrapolation) — infeasible
  within this session. This is a scope decision made explicit here, not a silently
  incomplete claim. Resume with:
  `python ml/scripts/download_hursat_sample.py --season <YEAR>` per season (or
  `--sample-storms N` for a larger seeded sample), then re-run
  `python ml/scripts/build_satellite_dataset.py` (it will pick up and re-process the 12
  storms already on disk plus whatever is newly downloaded — idempotent, no code changes
  needed to scale up). Running it unattended overnight, or in smaller per-season batches
  across multiple sessions, is the realistic path to full-archive coverage.
- No thumbnail pre-rendering (PNG) for API serving — deferred; not required by this
  phase's explicit exclusion list (no frontend/API changes).
- ADT-HURSAT coverage is not guaranteed for every storm (Phase 1: NCEI does not publish
  ADT records for every IBTrACS storm); `adt_qc_status = "unmatched"` samples keep their
  IBTrACS-sourced fields and are retained, not dropped.
- The reproducibility test doubles the real-data integration suite's runtime (two full
  109-frame pipeline runs). Deliberately isolated to one test, not paid per-assertion.
- IRWIN is the only channel processed (IR-window-only MVP, the Phase 1-selected strategy).
  `IRWVP`/`IRNIR`/`IRSPL`/`VSCHN` etc. are inventoried (channel list recorded) but not
  extracted into the Zarr store.

---

## 20. Next Phase

Per the roadmap, **P5 — Classification** depends on P4. Before starting it: either (a) run
a larger real download (§19's resume command) to raise the fused-sample count materially
above the current 627, since that is enough to prove the pipeline end-to-end but still far
short of the ~14,500-frame full-archive estimate a CNN would need, or (b) begin P5's
blocking label-analysis step (Scene class distribution, merge-rule decisions) against the
current real 627-sample, 8-class dataset as a genuine methodology dry run — the class
imbalance already visible (CDO 177 vs. IrrCDO 11, a ~16x ratio) is real signal, not
noise — deferring the compute-heavy full build to run unattended (e.g. overnight) before
actual model training starts.
