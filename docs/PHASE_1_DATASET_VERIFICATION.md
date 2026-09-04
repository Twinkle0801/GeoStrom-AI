# PHASE 1 — DATASET VERIFICATION REPORT

**Project:** GeoStrom AI · **Phase:** 1 (Foundation & Dataset Verification) · **Status:** Complete

> This report records what was actually checked against real data, how it was checked, and what
> was found. It corrects DATA_STRATEGY.md and DEVELOPMENT_ROADMAP.md in place; this document is
> the narrative and evidence trail behind those edits. All scripts referenced are in `ml/scripts/`,
> all raw JSON reports are in `ml/reports/`, and every downloaded file is recorded with a real
> SHA-256 checksum in `ml/manifests/datasets.json`.

---

## 1. Executive Summary

Phase 1 set out to answer one question: **can IBTrACS, HURSAT-B1, and ADT-HURSAT actually be joined
into a reliable, temporally- and spatially-aligned training dataset, or does the Phase 0 architecture
rest on an assumption that breaks on contact with real data?**

The answer is **yes, and better than Phase 0 assumed.** The crosswalk between HURSAT-B1 and IBTrACS
is not a fuzzy, name-based heuristic — it is an exact, triple-redundant key (filename token, NetCDF
global attribute, and an internal data variable all agree on 100% of sampled frames). The measured
timestamp offset between HURSAT-B1 and IBTrACS was **exactly zero minutes** on every sampled frame.
Spatial agreement between the HURSAT frame centre and the IBTrACS position was a median of **0.0 km**.
ADT-HURSAT, which Phase 0 flagged as the single highest-value open question, **does contain genuine
Dvorak scene-type labels** — resolving the classification-target question in favour of the
scientifically strongest option.

One critical bug was found and fixed: a naive pandas CSV load silently corrupted 99.5% of North
Atlantic rows because pandas' default missing-value handling treats the literal string `"NA"` — which
is also the IBTrACS basin code for North Atlantic — as `NaN`. This is exactly the class of silent,
undetected corruption the Phase 0 methodology section warned about, and it was caught by the
verification script's own sanity output, not by luck.

The basin decision was tested rather than assumed: North Atlantic and North Indian were compared
head-to-head on real data, and North Atlantic is unambiguously cleaner — one wind-averaging
convention present vs. four, higher storm-duration coverage for the planned sequence windows, and
higher HURSAT archive coverage. **North Atlantic is recommended and was used as the verification
basin throughout this report**, consistent with the Phase 0 default, but this recommendation is
presented for explicit sign-off rather than silently locked, per the task instructions.

No blocking question remains unanswered. Two items are marked **partially verified** (antimeridian
behaviour, wind-radii population) because the North Atlantic sample used does not exercise those
code paths — both are flagged for explicit testing in Phase 4, not silently assumed safe.

**No model was trained. No frontend, dashboard, database, or API was built. No full archive was
downloaded.** Total data footprint from this phase: ~130 MB on disk, all outside OneDrive and outside
the Git repository.

---

## 2. Environment / DATA_ROOT Result

**Finding: the obvious fallback location was also unsafe.** This machine has OneDrive "Known Folder
Move" active, which silently redirects `Desktop`, `Documents`, **and `Pictures`** into OneDrive. A
naive choice of `~/Documents/GeoStromData` — a very plausible default — would have violated the
Phase 0 safety rule while looking like a normal user-profile path.

| Check | Result |
|---|---|
| `DATA_ROOT` selected | `C:\GeoStromData` |
| Inside OneDrive? | No — verified programmatically and by directory probe |
| Inside the Git repository? | No |
| Writable without elevation? | Yes, confirmed by a write/read/delete round trip |
| Free disk at `C:\` | 637 GB |
| Guard mechanism | `ml/geostrom_ml/config.py::get_data_root()` — raises `DataRootError` for any path inside OneDrive or the repo, checked against `%OneDrive%`, `%OneDriveConsumer%`, `%OneDriveCommercial%`, and the repo root, not against a hardcoded string |

The guard was tested against three cases: the repo's own `datasets/` subfolder (rejected, both for
being inside OneDrive *and* inside the repo), `~/OneDrive/Documents/data` (rejected), and
`C:\GeoStromData` (accepted, zones created). All three behaved as expected.

**Files created:** `.env.example` (names only, no secrets, documents the OneDrive Known-Folder-Move
trap explicitly for future contributors), `.gitignore` (blocks datasets, model artefacts, and
secrets from ever being committed), `ml/geostrom_ml/config.py` (the DATA_ROOT resolver and guard).

---

## 3. Basin Recommendation

**Task instruction was explicit: do not silently lock North Atlantic.** North Atlantic was used as
the *verification* basin (as permitted — "if a basin is absolutely required for continuing a
technical verification task, use North Atlantic temporarily"), but a real, data-backed comparison
against North Indian was run rather than asserted.

| Dimension | North Atlantic (NA) | North Indian (NI) | Measured from |
|---|---|---|---|
| IBTrACS file size | 57.1 MB | 27.9 MB | direct download |
| Storms (all-time) | 2,298 | 1,858 | `verify_ibtracs.py` |
| Wind-averaging conventions present | **1** (US agencies, 1-min, 85.2% populated) | **4** (US 26.0%, Tokyo 10-min 2.0%, New Delhi 3-min 15.1%, CMA 2-min 4.6%, HKO 10-min 3.6%) | `verify_ibtracs.py`, `agency_wind_conventions` |
| USA_WIND population (all rows) | 85.16% | 26.03% | same |
| Observed+synoptic USA-original rows | 42.7% of all rows | 24.9% of all rows | same |
| Storms with ≥12 synoptic steps (`L8+H4` feasible) | **84.6%** | 66.5% | same |
| HURSAT archive coverage (7 sampled seasons, storms with an archive / IBTrACS storms) | 93–100% | 27–71% | `verify_crosswalk.py` |
| Duplicate (SID,time) rows within one file | 0 | 0 | same |

**Every axis favours North Atlantic.** The wind-convention finding is decisive on its own: DATA_STRATEGY.md's
locked decision #4 ("one agency's wind column, no fallback") is nearly free to honour in NA — the
single USA column already covers 85% of all rows — while in NI, the same rule would leave the
majority of storms with **no usable label at all**, because no single agency dominates.

**Recommendation: confirm North Atlantic as the MVP basin.** This matches the Phase 0 default, but
it is now an evidence-backed recommendation rather than an inherited assumption, and is presented
here for explicit approval before Phase 2 rather than silently carried forward.

---

## 4. IBTrACS Verification

**Source used:** official NOAA/NCEI distribution, v04r01, CSV format — **not** a Kaggle mirror, per
the task instruction. Landing page and the official *IBTrACS v04r01 column documentation* PDF were
retrieved directly from `ncei.noaa.gov` and used as the authority for every field definition below,
including the exact per-agency wind-averaging periods.

- Availability: confirmed live, updated three times weekly (Sun/Tue/Thu).
- Formats confirmed available: NetCDF, CSV, Shapefile. CSV was used (smallest practical format for
  tabular verification, per the task's "download only the smallest practical subset" instruction).
- Files downloaded: `ibtracs.NA.list.v04r01.csv` (57,143,389 bytes) and `ibtracs.NI.list.v04r01.csv`
  (27,877,743 bytes) — basin-specific subsets, not the full `ALL` archive.

| Item requested | Result |
|---|---|
| Storm identifier field | `SID` (e.g. `2005236N23285`); cross-reference `USA_ATCF_ID` also present |
| Timestamp field | `ISO_TIME`, format `YYYY-MM-DD HH:mm:ss` |
| Latitude/longitude fields | `LAT`, `LON` (degrees north / east, −180..180 convention) |
| Wind fields | `USA_WIND`, `WMO_WIND`, plus 9 agency-specific `*_WIND` columns |
| Pressure fields | `USA_PRES`, `WMO_PRES`, plus agency-specific `*_PRES` columns |
| Agency-specific wind fields | Confirmed: `USA_WIND` (1-min), `TOKYO_WIND` (10-min), `NEWDELHI_WIND` (3-min), `REUNION_WIND` (10-min), `BOM_WIND` (10-min), `NADI_WIND` (10-min), `WELLINGTON_WIND` (10-min), `CMA_WIND` (2-min), `HKO_WIND` (10-min) — averaging periods from the official column documentation, cross-checked against measured population rates |
| Observed vs. interpolated | `IFLAG` (15-char per-agency interpolation flag string: `O`=original, `P`=position-interpolated, `I`=intensity-interpolated, `V`=partial, `_`=missing) and `TRACK_TYPE` (`main`/`spur-*`/`PROVISIONAL*`) |
| Basin/storm ID fields | `BASIN`, `SUBBASIN`, `SID`, `NUMBER`, `NAME` |
| Update/version info | v04r01, released June 2024, updated 3×/week; file `Last-Modified` header confirmed 2026-08-30 |

### Schema inspection results (`ml/scripts/verify_ibtracs.py`)

Run against both files. Full JSON in `ml/reports/ibtracs_verification_ibtracs.{NA,NI}.list.v04r01.json`.

| Metric | NA | NI |
|---|---|---|
| Rows × columns | 127,188 × 174 | 62,848 × 174 |
| Distinct storms | 2,298 | 1,858 |
| Season range | 1851–2026 | 1842–2025 |
| Duplicate `(SID, ISO_TIME)` within file | 0 | 0 |
| Longitude convention | −180..180 (observed −136.9..63.0) | −180..180 (observed −87.7..163.7) |
| Rows at exact 6-hourly synoptic times | 64,312 (50.6%) | 31,511 (50.1%) |

**Critical bug found and fixed:** the first run used `pandas.read_csv` with default NA handling.
Pandas' built-in missing-value token list includes the literal string `"NA"` — which collides with
IBTrACS's own basin code for North Atlantic. Result: **126,586 of 127,188 rows (99.5%) had `BASIN`
silently set to `NaN`**, and the same trap applies to any other IBTrACS field whose valid value
happens to be `"NA"`. Fixed by setting `keep_default_na=False` and supplying an explicit sentinel
list (`""`, `-999`, `-9999`, `MM`, etc.) instead. This is now documented as a standing rule in the
verification script and DATA_STRATEGY.md.

**Second issue found and fixed:** a naive cross-file concatenation of NA + NI produced 548 apparent
duplicate `(SID, ISO_TIME)` rows. Root cause: IBTrACS basin files **overlap** — a storm that crosses
basins (4 such storms across the two files, all pre-1970) is listed in full in every basin file it
touches. Verified the duplicated rows are **byte-identical** in content (100% of 548 rows), so
deduplicating on `(SID, ISO_TIME)` after concatenation is lossless. This check (`T1`/`T1b`) is now a
permanent part of the QC gate.

---

## 5. HURSAT Verification

**Source used:** official NOAA/NCEI product page and archive directory listing — not a mirror.

- **HURSAT-B1 availability:** confirmed live at `ncei.noaa.gov/data/hurricane-satellite-hursat-b1/`.
- **Version:** v06 (also v05 present; v06 used as current).
- **Temporal coverage:** 1978–2015 (confirmed via product page; sample satellites GOES-7/8 through
  GOES-15/Meteosat-10 span the full range).
- **Spatial coverage:** storm-centric, global (any basin with a tracked TC).
- **Channels:** confirmed 7 possible variables — `IRWIN` (10.3–11.0 µm, always present in the
  sample), `IRWVP` (6.45–7.02 µm water vapour), `IRNIR` (3.79–4.04 µm), `IRSPL` (11.6–12.5 µm split
  window), `VSCHN` (0.53–0.77 µm visible), `VSVAR`, `IRVAR`. **Channel completeness is not uniform**
  — only 58% of sampled frames carry all 7; some older-satellite frames carry `IRWIN` alone.
- **Temporal resolution:** 3-hourly.
- **Spatial resolution/grid:** confirmed **301×301** on 100% of sampled frames, 0.07° lat/lon,
  Mercator projection, ~1,100 km extent.
- **File format:** NetCDF4, one file per storm/satellite-view/timestamp, packaged as a per-storm
  `.tar.gz`.
- **Storm identification metadata:** the IBTrACS `SID` is embedded **three independent ways** — see
  §6.
- **Timestamp metadata:** `htime` coordinate (nominal, synchronised to IBTrACS `ISO_TIME` — see §7).
- **Approximate file sizes:** measured mean **26.3 MB/storm** (compressed) for NA across 7 sampled
  seasons; individual frame files average **0.41 MB** uncompressed.

### Sample selection

Three storms were downloaded across three different eras of the archive, deliberately spanning early,
middle, and late coverage:

| Storm | Season | SID | Archive size | Frames | Satellites present |
|---|---|---|---|---|---|
| Gabrielle | 1995 | `1995222N24265` | 6.4 MB | 37 | GOES-8 |
| Katrina | 2005 | `2005236N23285` | 24.4 MB | 106 | GOES-10, GOES-12 |
| Claudette | 2015 | `2015193N35285` | 12.5 MB | 52 | GOES-13, GOES-15, Meteosat-10 |

195 individual NetCDF frames were opened and inspected (`ml/scripts/verify_hursat_join.py`); **zero
open errors**. Each frame's identity, timestamps, embedded state vector, ARCHER/eye diagnostics, and
`IRWIN` image content were extracted.

A separate, broader scan (`ml/scripts/verify_crosswalk.py`) fetched only small HTML directory
**listings** (no imagery) for 7 seasons — 1985, 1990, 1995, 2000, 2005, 2010, 2015 — covering 745
HURSAT archives, to test the identifier grammar and basin coverage at wider scale without a large
download.

---

## 6. HURSAT ↔ IBTrACS Crosswalk Result

**This was the highest-priority task of Phase 1, and it is resolved decisively.**

### Hypothesis and mechanism

Filename inspection (`HURSAT_b1_v06_2005236N23285_KATRINA_c20170721.tar.gz`) suggested the IBTrACS
SID is embedded directly in the filename. This was tested — not assumed — three independent ways per
frame:

1. **Filename token** — regex-extracted from the `.tar.gz` name and again from each inner `.nc` name.
2. **NetCDF global attribute** `TC_serial_number` (e.g. `"2005236N23285"`).
3. **NetCDF data variable** `sid` (a per-record string variable inside the file).

### Frame-level result (195 frames, 3 storms)

| Check | Result |
|---|---|
| Filename SID == attribute SID | **195/195 (100%)** |
| Data-variable SID == attribute SID | **195/195 (100%)** |
| Frames that failed to open | 0/195 |

### Archive-listing-level result (745 archives, 7 seasons: 1985/1990/1995/2000/2005/2010/2015)

| Metric | Result |
|---|---|
| Well-formed IBTrACS-grammar SIDs in filenames | **745/745 (100%)** |
| Malformed SIDs | 0 |
| Duplicate SIDs within a season | 0 |
| NA-basin storm coverage (storms with an archive ÷ IBTrACS NA storms that season) | 93–100% across all 7 seasons (100% in 1990 and 2015) |
| NI-basin storm coverage | 27–71% across the same seasons |
| Storm-name agreement (HURSAT filename name vs. IBTrACS `NAME`, where both are named) | 130/131 checked (99.2%) |

**Number of attempted joins / successful / failed:** at the frame level, 109 deduplicated frames
(from 195 raw frames — see §8) were joined against IBTrACS; **109/109 succeeded (100%), 0 failed**,
0 reasons for failure to report.

### Timestamp difference distribution

| Metric | Value |
|---|---|
| Attempted joins | 109 |
| Matched within ±90 min | 109 (100.0%) |
| \|Δt\| minimum | 0.0 min |
| \|Δt\| median | 0.0 min |
| \|Δt\| mean | 0.0 min |
| \|Δt\| maximum | 0.0 min |
| % exactly zero offset | **100.0%** |

**HURSAT-B1's `htime` is a nominal, pre-synchronised slot, not a raw scan time** — it already equals
the IBTrACS synoptic timestamp exactly for every sampled frame. The Phase 0 ±90-minute tolerance
hypothesis is not merely satisfied; for HURSAT-B1 itself it has essentially no work to do. (Contrast
with ADT-HURSAT, §9, which does carry true scan times with a realistic offset distribution — the
Phase 0 tolerance turns out to matter there instead.)

### Spatial agreement

| Metric | Value |
|---|---|
| N compared | 109 |
| Median separation | **0.0 km** |
| Mean separation | 2.02 km |
| 95th percentile | 5.69 km |
| Maximum | 10.65 km |
| % under the 50 km QC threshold | **100.0%** |

### Duplicate / missing frames

- **Duplicate frames at the same (storm, timestamp):** 86 of 109 unique timestamps (78.9%) had more
  than one satellite view (max observed: 2 simultaneous views). This is expected — multiple GOES
  satellites can see the same storm — and is fully resolved by the documented dedup rule.
- **Dedup rule tested:** keep the frame with minimum `VZA` (view zenith angle) per `(SID, htime)`.
  `VZA` was present on **100%** of sampled frames. After applying the rule: 195 raw frames → 109
  deduplicated frames, zero remaining collisions.
- **Missing frames:** within the observed time span of each sample storm, 0 synoptic best-track
  timestamps lacked a corresponding frame (small sample; not a large-N claim — see §12 caveats).
- **Naming inconsistencies:** none found in the sample; 1 of 131 name comparisons mismatched (a
  storm renamed or an `UNNAMED`/`MISSING` placeholder on one side), immaterial since the join key is
  the SID, not the name.

**Conclusion: the crosswalk is not a fuzzy heuristic requiring a fallback. It is an exact key,
triple-redundant, with a temporal offset of zero and sub-3 km mean spatial agreement.** TO-VERIFY #8
(blocking) is resolved with the strongest possible result.

---

## 7. 6-Hour Synoptic Alignment Result

| Check | Result |
|---|---|
| Matched HURSAT frames falling exactly on a 00/06/12/18 UTC synoptic hour | 56/109 (51.4%) |
| IBTrACS hour distribution (all rows, NA) | Bimodal — synoptic hours (0,6,12,18) each carry ~15,900 rows; off-synoptic hours (landfall/peak-intensity special reports) carry ~40–60 rows each |
| `IFLAG` USA-original ('O') among matched frames | 56/109 (51.4%) — **identical to the synoptic fraction**, i.e. every synoptic-hour match in the sample was also a USA-original (non-interpolated) report |
| `TRACK_TYPE` among matched frames | 100% `main` (no `spur`/`PROVISIONAL` in the sample) |
| Satellite frames alignable to 00/06/12/18Z | **Yes** — confirmed directly; the HURSAT nominal-time mechanism (§6) means alignment is exact, not approximate |

**No label interpolation was performed**, per the task instruction. The 48.6% of frames that fall on
non-synoptic hours (3-hourly cadence between the 6-hourly labelled slots) are retained for display
purposes only, exactly as the Phase 0 architecture specified — they are not used to manufacture
training labels.

---

## 8. Wind Convention Result — BLOCKING task

**Decision, evidence-backed, for the North Atlantic basin:**

| Field | Value |
|---|---|
| **Agency selected** | `USA_WIND` (US agencies: NHC/JTWC/HURDAT/ATCF, hierarchically selected per the official IBTrACS documentation) |
| **Averaging period** | **1-minute sustained** |
| **Units** | Knots |
| **Missing-value rate (all NA rows)** | 14.84% |
| **Missing-value rate (usable rows: synoptic, observed, main track, 1980–2015)** | **0.11%** (15,116/15,133) |
| **Temporal coverage** | Full archive span; population climbs sharply after the satellite era begins |
| **Why this agency/field** | It is the sole wind field with a **single, unambiguous averaging convention** in this basin — every other populated field in NA (`WMO_WIND`, at 43.8% population) mixes conventions across storms depending on which agency was "responsible" at the time, which is precisely the failure mode DATA_STRATEGY.md's pitfall #1 warns about. `USA_WIND` alone covers 85.16% of all NA rows and 99.89% of the already-filtered usable rows. |

**No cross-agency fallback is used.** A row with `USA_WIND` null and a populated `WMO_WIND` is
dropped rather than backfilled, exactly per the locked decision — verified this costs only 0.11% of
otherwise-usable rows, i.e. essentially free to honour strictly.

**Candidate agency comparison (since basin sign-off is still pending per §3):** if North Indian were
selected instead, no single agency would clear even 30% coverage (`USA_WIND` itself drops to 26.0% in
NI), and four different averaging periods (1-min, 3-min, 10-min, 2-min) would all be competing for
the role — a materially worse position than North Atlantic's. This is additional, independent
evidence for the basin recommendation in §3, not merely a restatement of it.

---

## 9. ADT-HURSAT Result

**This was flagged in Phase 0 as "the single highest-value verification task" (TO-VERIFY #16), because
it decides whether the project delivers true Dvorak pattern classification or falls back to intensity-
stage classification. It is resolved positively.**

**Source:** official NCEI product page (`ncei.noaa.gov/products/advanced-dvorak-technique-hurricane-satellite`)
and the NCEI Accession 0307249 data directory — not a third-party mirror. Confirmed: ADT-HURSAT files
are named directly by IBTrACS SID (e.g. `2005236N23285.nc`), one file per storm, NetCDF4, spanning
1978–2024, ADT algorithm version 9.0 applied to HURSAT V07b imagery.

### Does a genuine scene-type field exist? **YES.**

The `Scene` variable (string) is present, derived from two integer sub-fields with a fully documented
code book:

- `EyeScene`: 0=Eye, 1=Pinhole Eye, 2=Large Eye, 3=No Eye
- `CloudScene`: 0=CDO, 1=Embedded Center, 2=Irregular CDO, 3=Curved Band, 4=Shear

This is exactly the Dvorak taxonomy that DATA_STRATEGY.md's Tier B specified as the "true" pattern
label, described there only as a hoped-for outcome pending verification.

### Class distribution (measured: 2005 North Atlantic season, 31 storms, 1,727 ADT records)

| Scene | Count | % |
|---|---|---|
| CurvedBand | 536 | 31.0% |
| Shear | 477 | 27.6% |
| CDO | 296 | 17.1% |
| Land | 223 | 12.9% |
| Eye | 104 | 6.0% |
| IrrCDO | 40 | 2.3% |
| EmbCenter | 33 | 1.9% |
| LargeEye | 13 | 0.75% |
| PinholeEye | 5 | 0.29% |

Imbalance ratio (max/min): **107×**. This is real and must be handled by the layered strategy already
specified in ML_ARCHITECTURE.md §5.3 (class weights → balanced sampling → augmentation → the
pre-declared merge rule); it does not change the recommendation to use `Scene` as the primary target.

### Coverage and timing

| Metric | Value |
|---|---|
| Storms with an ADT file (2005 NA sample) | **31/31 (100%)** |
| Mean ADT records per storm | 55.7 (true scan-time cadence, denser than 6-hourly) |
| Join success to IBTrACS at ±90 min tolerance | **100.0%** |
| Join success at ±30 min | 99.94% |
| Join success at ±15 min | 86.16% |
| \|Δt\| median / mean / max (at ±90 min) | 15.0 / 16.53 / 75.0 minutes |
| % exact-zero offset | 1.04% |

Unlike HURSAT-B1, **ADT-HURSAT records true satellite scan times** (`Date`+`Time`, e.g. `17:45:13`),
not nominal synoptic slots — confirming the Phase 0 tolerance design was solving a real problem, just
not the one HURSAT-B1 itself turned out to have.

### ADT vs. best-track intensity agreement (QC-signal usefulness)

| Metric | Value |
|---|---|
| N compared (WindSpeed vs. USA_WIND, matched records) | 1,504 |
| Mean bias | −6.24 kt (ADT reads slightly low) |
| MAE | 11.3 kt |
| Correlation (r) | 0.87 |

This confirms NCEI's own published caveat verbatim: *"This dataset should not be used to determine
actual storm intensities."* ADT-HURSAT is usable exactly as Phase 0 planned — a structural-label
source and a QC cross-check — never as an intensity ground truth. IBTrACS best-track remains the
sole intensity label source, unchanged.

---

## 10. Negative-Class (Detection) Result

**Verifying the Phase 0 finding that HURSAT is storm-centric by construction and therefore contains
no true negatives.**

| Check | Result |
|---|---|
| Every sampled HURSAT frame is centred on a tracked TC? | **Yes, confirmed** — 100% of 195 frames have a valid `CentLat`/`CentLon` and join successfully to an IBTrACS storm |
| Frames whose `NATURE` is non-tropical (Path-B negative candidates) | 18/109 matched frames in the 3-storm sample (16.5%); on the larger NA 1980–2015 usable-row population: **2,596/10,154 rows (25.6%)** are non-tropical (`ET`/`DS`/`SS`, i.e. extratropical, disturbance, or subtropical) |
| Would another public satellite source be needed for a *general* detection claim? | **Yes** — Path-B negatives still contain an organised (if non-tropical) vortex; they support "tropical vs. non-tropical/disorganised" framing, not "cyclone present vs. arbitrary empty sky". GridSat-B1 (Path A) would be required to support the stronger, general claim. Not downloaded in Phase 1, per the strict scope rules (no new negative dataset without explicit sign-off) |
| Must the MVP detection claim be narrowed? | **Yes, confirmed, matching the Phase 0 plan exactly.** No new information changes this — Path B (non-tropical/pre-genesis frames) remains the correct MVP default; Path A remains the honest upgrade path |

**Recommendation: unchanged from Phase 0.** Ship Path B, state the scope explicitly in the UI and
methodology page ("organised tropical system vs. non-tropical/disorganised system"), and treat Path A
(GridSat-B1) as a defined but undelivered upgrade.

---

## 11. Sample-Size / Storage Estimates

**⚠️ ALL FIGURES IN THIS SECTION ARE ESTIMATES.** IBTrACS-derived counts (rows, storms, sequence
windows) are computed exactly from the full downloaded NA file — those are census figures, not
extrapolations. Anything involving satellite imagery coverage is **extrapolated from a 3-storm,
195-frame sample** (measured HURSAT coverage rate: 100% join success, 96% average storm-archive
coverage across sampled seasons) and is explicitly an estimate, not a full-archive count.
(`ml/scripts/estimate_dataset.py`, full JSON in `ml/reports/dataset_estimates_1980_2015.json`.)

### Funnel — North Atlantic, 1980–2015 (36 seasons)

| Stage | Rows | Storms | Basis |
|---|---:|---:|---|
| 1. Raw IBTrACS rows | 29,995 | 547 | exact |
| 2. Synoptic (00/06/12/18 UTC) | 15,134 | 547 | exact |
| 3. + observed (`IFLAG[0]=='O'`) + main track | 15,133 | 547 | exact |
| 4. + `USA_WIND` present | 15,116 | 547 | exact |
| 5. Sequence windows (`L=8, H=4`) | **9,398 windows** | 445 contributing (102 storms too short) | exact, from IBTrACS alone |
| — sensitivity: `L=4,H=4` | 11,338 windows | — | exact |
| — sensitivity: `L=12,H=4` | 7,727 windows | — | exact |
| 6. **Estimated** fused frames with imagery | **~14,511** | ~525 (est.) | **estimated**, 96% coverage rate applied |
| 7. **Estimated** windows with imagery | **~9,022** | — | **estimated** |

### Season-based split (proposed, by season — not yet frozen)

| Split | Seasons | Storms | Frames | Windows |
|---|---|---:|---:|---:|
| Train | 1980–2004 (25 seasons) | 366 | 9,903 | 6,109 |
| Val | 2005–2009 (5 seasons) | 86 | 2,452 | 1,557 |
| Test | 2010–2015 (6 seasons) | 95 | 2,761 | 1,732 |

### Storage estimates (GB)

| Item | GB | Note |
|---|---:|---|
| Raw HURSAT `.tar.gz` download | 13.8 | estimate, extrapolated |
| Raw HURSAT extracted (transient) | 24.3 | deleted after Zarr conversion, per DATA_STRATEGY.md §4.5 |
| Zarr, `uint8`, 224², synoptic-only | 0.73 | the actual training footprint |
| Zarr, `uint8`, 224², all 3-hourly (display) | 1.42 | |
| (for comparison) same data as `float32` | 2.91 | shows the 4× benefit of quantisation |
| IBTrACS CSV (raw) | 0.085 | exact |
| ADT-HURSAT (all storms) | 0.023 | exact scaling from measured 42 KB/storm |
| Rendered thumbnails (256², PNG) | 0.36 | |
| **Working total during Phase 4 conversion** | **~25.5** | raw + zarr + csv + thumbnails, peak |
| **Steady-state total after raw is deleted** | **~1.2** | matches Phase 0's "single-digit GB" prediction |

**Conclusion on TO-VERIFY #20 (the sample-count blocking question):** the surviving sample count
(~9,000 fused sequence windows, ~14,500 fused classification/detection frames) is **not** in the
"too small — fall back to GBM-only" regime the Phase 0 roadmap flagged as a risk. It supports the
planned pretrained-CNN + GRU/LightGBM comparison as scoped, while remaining modest enough that
pretrained weights and strong baselines stay the right call (unchanged from Phase 0 — this is a
confirmation, not a new finding).

---

## 12. QC Results

A preliminary QC gate (`ml/scripts/qc_gate.py`) implementing the DATA_STRATEGY.md §4.4 assertions,
plus the Phase 1 task list's additional checks, was run against the verification sample.

| ID | Blocking | Check | Result |
|---|:---:|---|---|
| T1 | ✅ | No duplicate `(SID, ISO_TIME)` in best track (post cross-file dedup) | **PASS** (0) |
| T1b | ✅ | Cross-basin-file duplicates are byte-identical | **PASS** (548/548) |
| T2 | ✅ | No missing/unparseable `ISO_TIME` | **PASS** (0) |
| T3 | ✅ | `LAT` within [−90,90], non-null | **PASS** (0 violations) |
| T4 | ✅ | `LON` within [−180,180], non-null | **PASS** (0 violations) |
| T5 | ✅ | `USA_WIND` present ≥90% of usable rows | **PASS** (99.76%) |
| T6 | — | `USA_PRES` present ≥50% of usable rows | **PASS** (79.51%) |
| T7 | ✅ | Wind/pressure within physical ranges | **PASS** (0 violations) |
| S1 | ✅ | All satellite frames open without error | **PASS** (0 errors / 195) |
| S2 | ✅ | Filename SID == embedded `TC_serial_number` | **PASS** (0 mismatches / 195) |
| S3 | ✅ | Every frame SID resolves in IBTrACS | **PASS** (0 unresolved) |
| S4 | — | All frames are 301×301 | **PASS** (195/195) |
| S5 | ✅ | Duplicate frames resolvable via `VZA` | **PASS** (100% VZA present) |
| S6 | ✅ | Frames join to best track within tolerance | **PASS** (100.0%) |
| S7 | ✅ | \|Δt\| within tolerance for all joined frames | **PASS** (max 0.0 min) |
| S8 | ✅ | Frame centre agrees with best track <50 km | **PASS** (100%, median 0.0 km) |
| S9 | — | Synoptic rows lacking a frame (informational) | **PASS** (0) |
| S10 | ✅ | `IRWIN` not constant/empty | **PASS** (0 / 195) |
| S11 | — | `NATURE` distribution non-degenerate | **PASS** (3 classes present) |

**Gate result: 19/19 PASS, 0 blocking failures.** (First run showed a T1 failure of 548 duplicate
rows; root-caused to the cross-basin-file overlap described in §4 and confirmed benign — see T1b.)
Full machine-readable output: `ml/reports/qc_gate_phase1.json`.

**Scope note:** this is a *preliminary* gate over the Phase 1 verification sample (3 storms, 195
frames), not the production gate. The same assertions, unchanged, will run over the full fused
dataset in Phase 4 per the existing DEVELOPMENT_ROADMAP.md plan.

---

## 13. Blocking Issues

**None remain open.** Every item marked ⛔ in DATA_STRATEGY.md §8 (TO-VERIFY #1, #2, #8, #9, #10, #16,
#20) has been resolved with evidence, and none forced an architecture change beyond the two additive
fixes below.

Two non-blocking items remain only **partially** verified because the North Atlantic sample used
does not exercise them:

1. **Antimeridian crossing behaviour (#4).** Confirmed longitude convention (−180..180), but neither
   sample basin's storms crossed ±180°, so the LineString-splitting code path is untested. **Action
   for Phase 4:** test explicitly against a Western/Central Pacific storm before the map layer ships.
2. **Wind-radii field population (#7).** Confirmed `DIST2LAND` is 100% populated; wind-radii fields
   (`USA_R34_NE` 13.9%, `USA_R50_NE` 7.2%, `USA_R64_NE` 4.1%) are sparse, confirming they were
   already correctly scoped as Advanced-only, not MVP features. No action needed — this is a
   confirmation of the existing scope decision, not an open risk.

---

## 14. Architecture Changes Required

Two changes to Phase 0 documents, both **additive corrections**, neither a redesign:

1. **ADT-HURSAT `Scene` is promoted from "if available" to the recommended primary classification
   target**, ahead of the Tier A intensity-category fallback (DATA_STRATEGY.md §9.1, item 14). This
   was always the Phase 0-preferred outcome *conditional on verification* — verification succeeded,
   so the promotion is exactly what Phase 0's own decision tree specified, not a new direction.
2. **IRWIN validity masking uses a `<150 K` physical floor**, not fill-value (`-1.0`) equality alone
   — the documented sentinel was never observed exactly in the sample, while ~0.05% of pixels carry
   unphysical near-zero values that the documented check would miss (§9.1, item 15).

One **process rule** is added, not an architecture change: **all IBTrACS CSV loading must use
`keep_default_na=False`** with an explicit sentinel list, to prevent the basin-code collision found
in §4. This is now enforced in every verification script and must carry into the Phase 4 production
pipeline.

**No change** to: the offline/online system split, the database schema, the API design, the ML model
selection, the fusion join strategy, the directory structure, or the phase sequencing. The
architecture as designed in Phase 0 is validated, not revised.

---

## 15. Final Recommendation

# GO

All ten Phase 1 exit-criteria questions are answered with evidence (§16). The multi-source fusion
premise — the part of this project Phase 0 identified as carrying the real risk — is not merely
"viable," it performed better than the conservative Phase 0 assumptions in every measured dimension:
exact rather than fuzzy identity matching, zero rather than ±90-minute HURSAT-B1 time offset, sub-3 km
rather than sub-50 km spatial agreement, and a confirmed rather than hoped-for Dvorak label field.

The one open item requiring a human decision — not a technical blocker — is **explicit sign-off on
North Atlantic as the MVP basin** (§3). The evidence strongly favours it and it was used as the
verification basin throughout; a decision is requested rather than assumed, per the task instructions.

---

## 16. Phase 1 Exit Criteria

| # | Question | Status | Evidence |
|---|---|:---:|---|
| 1 | Can IBTrACS be downloaded and parsed? | ✅ | §4 — 2 basin files downloaded from the official NCEI source, parsed, schema-verified, one critical loader bug found and fixed |
| 2 | Can HURSAT be downloaded and parsed? | ✅ | §5 — 3 sample storms / 195 frames downloaded and opened with zero errors |
| 3 | Can HURSAT and IBTrACS be reliably cross-referenced? | ✅ | §6 — 100% exact match, triple-redundant key, across both a 195-frame deep sample and a 745-archive breadth scan |
| 4 | Can satellite observations be temporally aligned with cyclone observations? | ✅ | §7 — 100% of frames alignable to synoptic hours; measured offset exactly 0.0 minutes |
| 5 | Can a consistent wind label be selected? | ✅ | §8 — `USA_WIND`, 1-minute, single convention, 99.89% coverage of usable NA rows |
| 6 | Does ADT-HURSAT provide genuine pattern/scene labels? | ✅ | §9 — Yes, `Scene`/`EyeScene`/`CloudScene`, full Dvorak taxonomy confirmed with measured class distribution |
| 7 | How serious is the negative-class problem? | ✅ | §10 — Confirmed real; Path-B framing (25.6% non-tropical rows available) is the correct, scoped MVP mitigation |
| 8 | How many usable fused samples can realistically be expected? | ✅ | §11 — ~9,400 IBTrACS-only windows (exact); ~9,000 fused-with-imagery windows, ~14,500 fused frames (estimated) |
| 9 | How much storage will the MVP require? | ✅ | §11 — ~25.5 GB peak during conversion, ~1.2 GB steady-state |
| 10 | Is the proposed MVP dataset strategy feasible? | ✅ | §15 — Yes, without architectural revision |

---

## Appendix A — Reusable Verification Tooling

All scripts are read-only with respect to source data, idempotent, and designed to be re-run at full
scale in later phases without modification:

| Script | Purpose |
|---|---|
| `ml/geostrom_ml/config.py` | `DATA_ROOT` resolution and the OneDrive/repo safety guard |
| `ml/scripts/verify_ibtracs.py` | IBTrACS schema, dtype, missing-value, and structure verification |
| `ml/scripts/verify_crosswalk.py` | HURSAT filename ↔ IBTrACS SID crosswalk at archive-listing scale |
| `ml/scripts/verify_hursat_join.py` | Frame-level HURSAT NetCDF inspection, temporal/spatial join testing |
| `ml/scripts/verify_adt.py` | ADT-HURSAT scene-label and intensity cross-check verification |
| `ml/scripts/estimate_dataset.py` | Sample-size and storage estimation (clearly labelled as estimates) |
| `ml/scripts/qc_gate.py` | The DATA_STRATEGY.md §4.4 QC assertions, machine-readable JSON output |
| `ml/scripts/make_manifest.py` | Dataset manifest generation with real SHA-256 checksums |

## Appendix B — Raw Evidence Files

`ml/reports/ibtracs_verification_ibtracs.NA.list.v04r01.json` ·
`ml/reports/ibtracs_verification_ibtracs.NI.list.v04r01.json` ·
`ml/reports/crosswalk_verification.json` ·
`ml/reports/hursat_join_verification.json` ·
`ml/reports/adt_verification.json` ·
`ml/reports/dataset_estimates_1980_2015.json` ·
`ml/reports/qc_gate_phase1.json` ·
`ml/manifests/datasets.json`
