# PHASE 2 — FORECASTING BASELINES (IBTrACS Only)

**Project:** GeoStrom AI · **Phase:** 2 · **Status:** Complete · **Scope:** North Atlantic, IBTrACS only

> **Data legend used throughout this document:**
> **OBSERVED DATA** — a value read directly from IBTrACS, unmodified.
> **DERIVED FEATURE** — a value computed from observed data by a documented, causal transformation (e.g. `max_wind_so_far`, Haversine error).
> **MODEL PREDICTION** — an output of a fitted model (Persistence, Ridge, LightGBM, or CLIPER-style).
> These three categories are never conflated in this document, in the code's column names (`ref_*`/`y_*_true` vs `<model>__*`), or in the plots (every legend entry is tagged with its category).

---

## 1. Objective

Establish scientifically defensible, honestly-evaluated baseline performance for two IBTrACS-only
forecasting tasks — **cyclone intensity prediction** and **cyclone track prediction** — on the North
Atlantic basin, so that any future deep-learning model (GRU, LSTM, etc.) has a real, leak-free number
to beat. This phase builds no deep model; it builds the yardstick.

Per `DEVELOPMENT_ROADMAP.md` P2, this phase is also the project's schedule insurance policy: it
depends only on the IBTrACS portion of Phase 1, not on the satellite pipeline, and produces a
complete, deployable forecasting result independent of any future HURSAT work.

---

## 2. Data Used

**Source:** IBTrACS v04r01, North Atlantic basin file, exactly as downloaded and verified in Phase 1
(`ml/reports/ibtracs_verification_ibtracs.NA.list.v04r01.json`). No new data was downloaded in Phase 2.

**Basin:** North Atlantic (NA) — locked per the explicit Phase 2 instruction, superseding the
Phase 1 recommendation-pending-signoff.

**Wind field:** `USA_WIND` (1-minute sustained, US agencies) — the single agency column selected and
verified in Phase 1 (`docs/PHASE_1_DATASET_VERIFICATION.md` §8). No cross-agency fallback is used
anywhere in Phase 2; a row with `USA_WIND` null is dropped, never backfilled from `WMO_WIND` or
another agency.

**Season range:** 1980–2015, identical to the range Phase 1 used for its sample-size and storage
estimates (`docs/PHASE_1_DATASET_VERIFICATION.md` §11) — chosen for continuity, not re-derived.

### 2.1 Filtering rules (unchanged from Phase 0/1, reused verbatim)

Implemented once in `ml/geostrom_ml/data/ibtracs.py::filter_usable_rows`, reusing the exact loading
logic validated in Phase 1's `qc_gate.py` (see §12, "Reuse of Phase 1 components"):

1. **Synoptic times only** — 00/06/12/18 UTC, on the hour.
2. **Observed only** — `IFLAG` character 1 (USA agency) `== 'O'` (original report; excludes
   position- or intensity-interpolated rows).
3. **Main track only** — `TRACK_TYPE == 'main'` (excludes `spur-*`/`PROVISIONAL*`).
4. **`USA_WIND` present** — no fallback.
5. **Season ∈ [1980, 2015]**.

No interpolation is performed anywhere in Phase 2 — neither to fill a missing synoptic observation
nor to bridge a gap in a storm's sequence.

### 2.2 Volumes (exact, from the frozen split — `ml/manifests/splits_v1.json`)

| Stage | Rows/windows | Storms |
|---|---:|---:|
| Raw IBTrACS rows loaded (NA, 1980–2015, after cross-file dedup) | 126,888 | 2,292 |
| Usable rows (all 5 filters applied) | 15,116 | 547 |
| **Train** (seasons 1980–2004) | 9,903 observations → **6,109 windows** | 366 storms → 288 contributing |
| **Val** (seasons 2005–2009) | 2,452 observations → **1,557 windows** | 86 storms → 69 contributing |
| **Test** (seasons 2010–2015) | 2,761 observations → **1,732 windows** | 95 storms → 88 contributing |
| **Total windows** | **9,398** | 445 contributing (102 of 547 storms too short for `L+H=12` steps) |

These figures are an **exact match** to Phase 1's pre-registered estimate
(`docs/PHASE_1_DATASET_VERIFICATION.md` §11: "9,398 windows... from 445 storms", train/val/test
6,109/1,557/1,732) — obtained independently, by running the actual pipeline rather than
extrapolating from a sample. This cross-check gives strong confidence the implementation is correct.

---

## 3. Split Methodology

**File:** `ml/manifests/splits_v1.json` (committed, frozen). **Built by:** `ml/scripts/build_splits.py`.

- **Unit of split: storm ID**, never a row or window. Enforced by construction — the split assigns a
  `split` label to each storm, and every window (which spans up to 12 consecutive observations of one
  storm) inherits its storm's label. A window can never straddle two splits, because a storm cannot.
- **Method: season-block temporal split**, not a random storm-level split. `train` = seasons
  1980–2004 (25 seasons), `val` = 2005–2009 (5 seasons), `test` = 2010–2015 (6 seasons). This tests
  generalisation to **future, unseen storms** — the actual deployment condition — per
  `PROJECT_REQUIREMENTS.md` §4.1 rule 2.
- **Cross-check:** every storm's SID-encoded season (`SID[:4]`) is verified to agree with its
  IBTrACS `SEASON` field before the split is frozen; 547/547 storms checked, 0 mismatches.
- **Integrity check, computed and stored in the manifest itself:**
  `intersection(train, val) = []`, `intersection(train, test) = []`, `intersection(val, test) = []`,
  `all_disjoint = True`. Independently re-verified by `ml/tests/test_splits.py` (8 tests) against
  the frozen file on disk, and by `ml/tests/test_leakage.py` against the *materialised* Parquet
  dataset (not just the JSON manifest).
- **Determinism:** the split rule has no randomness. `ml/tests/test_splits.py::TestDeterminism`
  rebuilds the split from scratch and asserts the storm lists are identical to the frozen file.
- **Feature/dataset versioning:** `feature_version: "v1"`, `dataset_version: "v1"` are recorded in
  both the split manifest and the dataset manifest (`ml/manifests/dataset_v1_manifest.json`), so any
  future change to feature engineering is required to bump the version rather than silently
  invalidate old benchmark results.

---

## 4. Causal Feature Engineering

**Implementation:** `ml/geostrom_ml/features/engineering.py`. **Rule enforced:** every feature at
row *t* uses only that row and rows strictly before it, within the same storm. Verified by
`ml/tests/test_features.py` and `ml/tests/test_leakage.py` (see §9).

### 4.1 Per-timestep features (20), explicit temporal window per feature

| Feature | Temporal window | Category |
|---|---|---|
| `lat`, `abs_lat`, `lon_sin`, `lon_cos` | row *t* only | OBSERVED / DERIVED (trig of observed) |
| `USA_WIND`, `USA_PRES` | row *t* only | OBSERVED |
| `storm_speed_kt`, `storm_dir_sin`, `storm_dir_cos` | *t*−6h → *t* (one step back) | DERIVED (Haversine + bearing of the prior 6h) |
| `d_wind_6h`, `d_pres_6h` | *t*−6h → *t* | DERIVED (exact-lag difference; `NaN` if the row 6h back doesn't exist) |
| `d_wind_12h`, `d_pres_12h` | *t*−12h → *t* | DERIVED (exact-lag; `NaN` on any gap) |
| `d_wind_24h`, `d_pres_24h` | *t*−24h → *t* | DERIVED (exact-lag; `NaN` on any gap) |
| `max_wind_so_far` | [storm genesis .. *t*] inclusive, **expanding**, recomputed per row | DERIVED — **never** the whole-storm lifetime maximum; monotonically non-decreasing by construction and verified by test |
| `hours_since_genesis` | *t* vs. storm's first observation | DERIVED |
| `doy_sin`, `doy_cos` | *t* only | DERIVED (calendar, no lookback needed) |
| `dist2land` | row *t* only | OBSERVED (IBTrACS `DIST2LAND`) |

**On `max_wind_so_far` vs. the prohibited "storm summary statistic":** the Phase 2 instructions
explicitly ban using a whole-storm lifetime aggregate (e.g. a `storms` table's `max_wind`, computed
once over the entire storm and therefore constant and future-leaking at early timesteps).
`max_wind_so_far` is a different, causal object: an **expanding** maximum, recomputed independently
at every row from only that row and its predecessors. `ml/tests/test_features.py::
test_max_wind_so_far_is_expanding_not_lifetime` asserts an early-storm row's value is strictly below
the storm's eventual lifetime maximum — i.e. it does **not** silently become the lifetime aggregate.

### 4.2 Sequence windows: `L=8` input steps, `H∈{6,12,18,24}h` horizons

Matches `ML_ARCHITECTURE.md` §6.1/§7.1/§6.3 exactly (L=8 steps = 48h of input history; horizons
+6/+12/+18/+24h). A window is emitted **only if** all `L + max(H)/6 = 8+4 = 12` steps exist at an
**exact, contiguous 6-hour cadence** within one storm — verified via a run-length ID that breaks on
any non-6h gap (`ml/geostrom_ml/features/engineering.py::_contiguous_run_id`). No gap is ever
bridged by interpolation; `ml/tests/test_features.py::test_gap_breaks_window_eligibility` confirms a
storm with one missing observation yields strictly fewer windows than the same storm without the gap.

**Flattened representation for the tabular baselines (Ridge, LightGBM):** each window's model input
is the 20 per-timestep features × 8 lags = **160 flattened columns** (`x__<feature>__lag<k>`, `k=0`
= reference time *t*, `k=7` = *t*−42h), matching `ML_ARCHITECTURE.md` §6.5's "LightGBM on flattened
window features" design. The **CLIPER-style** baseline (§6 below) deliberately uses a much smaller,
hand-picked feature set instead of the full flattened block — this is what differentiates it from
the LightGBM baseline, per the architecture's own tiered design.

**Targets**, all computed from OBSERVED future rows relative to the reference row:

- `y_wind_abs_{h}h` — absolute future wind (OBSERVED, read directly from the target row)
- `y_wind_delta_{h}h` — `wind(t+h) - wind(t)` (DERIVED)
- `y_dlat_{h}h`, `y_dlon_{h}h` — displacement targets; **`y_dlon` always goes through the wrap-safe
  `wrap_lon_diff`**, never raw subtraction (§5)
- `y_lat_future_{h}h`, `y_lon_future_{h}h` — absolute future position, retained for evaluation only

### 4.3 Leakage controls, summarised

| Control | Where enforced | Verified by |
|---|---|---|
| Split by storm, not row | `storm_to_split_map` + window inheriting storm's label | `test_splits.py`, `test_leakage.py` |
| No cross-agency wind fallback | `filter_usable_rows` drops null `USA_WIND` | code path; §2.1 |
| No interpolated rows | `IFLAG` + `TRACK_TYPE` filter | code path; §2.1 |
| No future information in any feature | per-timestep features use only backward shifts | `test_features.py::test_causality_...`, `test_leakage.py` (mutation-based regression tests) |
| No lifetime aggregates | `max_wind_so_far` is expanding, not whole-storm | `test_max_wind_so_far_is_expanding_not_lifetime` |
| No silently bridged gaps | contiguous-run windowing | `test_gap_breaks_window_eligibility` |
| Scalers/medians fit on train only | Ridge/CLIPER median-imputation fit in `.fit()`, applied (not refit) in `.predict()` | code path, `models/*.py` |

---

## 5. Longitude Wrapping

**Implementation:** `ml/geostrom_ml/features/geo.py`. Per the task instruction, longitude
differences are **never** computed by raw subtraction anywhere in this codebase — every displacement,
bearing, and distance calculation routes through `wrap_lon_diff`, which returns the shortest signed
angular difference in `(-180, 180]`.

**The two required regression tests, verified passing:**

```
wrap_lon_diff(179, -179)  ==  2.0    (not -358.0)
wrap_lon_diff(-179, 179)  == -2.0    (not  358.0)
```

Additional coverage: `wrap_lon_deg` (canonicalisation), `haversine_km` across the antimeridian (a
0.2°-separated pair at 179.9°/−179.9° correctly returns ~22 km, not ~40,000 km),
`initial_bearing_deg` across the antimeridian, `destination_point` (the inverse operation, used by
the track persistence baseline), and `displace` (reconstructing absolute position from a predicted
displacement, wrapping the result).

**Scope note carried over from Phase 1:** North Atlantic storms in the 1980–2015 sample do not
actually cross ±180° (confirmed range: −136.9°..63.0°, Phase 1 §4), so this code path is not
exercised by the *real* benchmark data — exactly as Phase 1 flagged ("antimeridian behaviour ...
NOT YET code-path-tested"). Phase 2 closes that gap at the **unit-test level** with synthetic
storms constructed specifically to cross the antimeridian (`ml/tests/test_features.py::
test_dlon_uses_wrap_safe_difference`, `ml/tests/test_metrics.py::
test_antimeridian_prediction_not_penalised_incorrectly`), so the implementation is proven correct
even though no North Atlantic storm in this dataset currently exercises it end-to-end.

---

## 6. Intensity Baselines

**Implementation:** `ml/geostrom_ml/models/intensity_baselines.py`. All three predict **absolute
wind** at each horizon; one model is fit per horizon for the learned baselines, per
`ML_ARCHITECTURE.md` §6.5.

| Baseline | Definition |
|---|---|
| **Persistence** | `future_wind = current_wind` at every horizon. Stateless — no `.fit()` step. |
| **Ridge** | `sklearn.linear_model.Ridge(alpha=1.0)` on the 160-column flattened window, one model per horizon. Median-imputed on train-only statistics (defensive; no `NaN`s occur in practice since window construction requires a full L=8 history). |
| **LightGBM** | `LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10)` on the same 160-column input, one model per horizon. |

No MAPE anywhere (rejected per `ML_ARCHITECTURE.md` §6.2 — penalises errors on weak storms far more
than identical errors on intense storms). Metrics: **MAE, RMSE, bias**, all in knots.

---

## 7. Track Baselines

**Implementation:** `ml/geostrom_ml/models/track_baselines.py`. All three predict **displacement**
(`dlat`, `dlon`) at each horizon, never absolute coordinates, per `PROJECT_REQUIREMENTS.md` §2.D.

| Baseline | Definition |
|---|---|
| **Persistence** | Constant-velocity extrapolation: the causal `storm_speed_kt`/`storm_dir` at the reference time (computed from the prior 6h only) is held constant and projected forward via `destination_point` (great-circle reckoning), matching `ML_ARCHITECTURE.md` §7.4 Tier 1a exactly ("constant velocity from the last two positions"). A storm with no prior motion persists at zero displacement rather than propagating `NaN`. |
| **CLIPER-style** | `Ridge(alpha=1.0)` per horizon per axis (`dlat`, `dlon`), on a **compact, hand-picked** feature set: current position (lat, lon sin/cos), motion (speed, bearing sin/cos), intensity (wind, pressure), day-of-year (sin/cos), storm age, **plus five explicit interaction terms** (lat×doy, speed×bearing, wind×|lat|) — matching `ML_ARCHITECTURE.md` §7.4's specification of "current position, motion, intensity, day-of-year, and their interactions." Deliberately **not** the full 160-column flattened window, which is what differentiates it from LightGBM below. |
| **LightGBM** | Same architecture as the intensity LightGBM, on the full 160-column flattened window, one model per horizon per axis (`dlat`, `dlon`). |

### 7.1 Evaluation

Absolute predicted position is reconstructed via `displace(ref_lat, ref_lon, pred_dlat, pred_dlon)`
(wrap-safe) — the same reconstruction a served forecast would use. Error is **always geodesic**:

- **Mean / median / p90 / max / RMSE great-circle error (km)** — `haversine_km`, never a flat-plane
  or raw-degree distance.
- **Mean lat error (deg)**, **mean lon error (deg)** — the wrap-safe signed longitude error, reported
  alongside distance, not instead of it (per the task instruction not to evaluate lat/lon
  independently as the *only* metric).
- **Along-track / cross-track decomposition (km)** — projects the (predicted, actual) error onto the
  bearing of the storm's actual motion, separating "wrong speed" from "wrong direction"
  (`ML_ARCHITECTURE.md` §7.2).

---

## 8. Benchmark Harness

**Implementation:** `ml/geostrom_ml/evaluation/benchmark.py` + `ml/scripts/run_phase2_benchmark.py`.

- **One call path for every model**, task-typed only at the harness level
  (`evaluate_intensity_model` / `evaluate_track_model`) — adding a new baseline means registering an
  instance in the model list, not writing new evaluation code, per `ML_ARCHITECTURE.md` §1.1.
- **Test set touched exactly once per model** — a single `model.predict(test_df)` call per model,
  per `ML_ARCHITECTURE.md` §8. The validation split is loaded and available (for any future model
  that needs it for hyperparameter selection) but Phase 2's fixed-hyperparameter baselines do not use
  it for that purpose — this is stated plainly, not hidden.
- **Every `BenchmarkResult` record contains**, exactly as required: `model_name`, `model_version`,
  `dataset_version`, `split_version`, `feature_version`, `forecast_horizon_h`, `sample_count`,
  `metrics` (task-appropriate dict), `timestamp_utc`, and `config` (a JSON-safe snapshot of the
  model's hyperparameters).
- **Output format:** JSON (`ml/reports/phase2_benchmark_results.json`, 24 records = 6 models × 4
  horizons) + Parquet (`ml/reports/phase2_test_predictions.parquet`, raw per-window predictions for
  every model, used by the plotting script so plots never re-fit a model). **Not** written to any
  database or API — Phase 2 is explicitly out of scope for that per the task brief.

---

## 9. Results

**Test set: 1,732 windows, 88 storms, seasons 2010–2015. Every number below is on this held-out set,
touched once.** Full precision in `ml/reports/phase2_benchmark_results.json`; comparison tables also
exported as `ml/reports/phase2_comparison_*.md`.

### 9.1 Intensity — MAE (kt) by horizon

| Model | +6h | +12h | +18h | **+24h (headline)** |
|---|---:|---:|---:|---:|
| Persistence | 3.21 | 6.09 | 8.53 | **10.64** |
| Ridge | 3.08 | 5.49 | 7.71 | **9.59** |
| **LightGBM** | **2.88** | **4.97** | **6.82** | **8.54** |

**Skill vs. persistence (headline +24h): LightGBM +19.8%, Ridge +9.9%.** LightGBM wins at every
horizon tested (+10.3% at 6h, +18.4% at 12h, +20.0% at 18h, +19.8% at 24h) — a consistent, growing
margin, not a fluke at one horizon.

**Bias (kt, positive = over-forecast) at +24h:** Persistence +1.85, Ridge +2.03, LightGBM +1.13.
All three models systematically over-forecast wind at longer horizons (intensity tends to weaken
over the window more often than the models predict) — LightGBM's bias is the smallest but not zero.
This is reported because it is true, not because it flatters LightGBM.

### 9.2 Track — mean great-circle error (km) by horizon

| Model | +6h | +12h | +18h | **+24h (headline)** |
|---|---:|---:|---:|---:|
| Persistence | 30.5 | 80.6 | 146.7 | **226.2** |
| **CLIPER-style (Ridge)** | 30.0 | **75.5** | **133.3** | **200.4** |
| LightGBM | 30.7 | 76.5 | 134.9 | 203.3 |

**Skill vs. persistence (headline +24h): CLIPER +11.4%, LightGBM +10.1%.**

**Reported as instructed, without adjusting the narrative: CLIPER-style Ridge regression beats
LightGBM on track prediction at every single horizon tested** (200.4 vs. 203.3 km at 24h; 133.3 vs.
134.9 km at 18h; 75.5 vs. 76.5 km at 12h; 30.0 vs. 30.7 km at 6h). The margin is small (1–3 km, on
the order of 1–2% of the error itself) but consistent across all four horizons, so it is unlikely to
be pure noise. **At the shortest horizon (+6h), LightGBM is marginally *worse* than persistence**
(30.7 vs. 30.5 km, a −0.6% "skill" — i.e. a loss): at very short lead times, storms have enough
momentum that naive constant-velocity extrapolation is already close to optimal, and a
higher-capacity model has more room to add noise than signal. See §11 for interpretation.

Median errors track the same ordering (CLIPER best, then LightGBM, then persistence) and are, as
expected for a right-skewed error distribution, meaningfully lower than the means at every horizon
(e.g. median 153.9 km vs. mean 200.4 km for CLIPER at 24h) — see `track_error_vs_horizon.png` for
both curves.

### 9.3 Along-track / cross-track decomposition (+24h, CLIPER)

Mean |along-track| = 135.7 km, mean |cross-track| = 119.5 km — error is somewhat more concentrated in
the direction of motion (speed error) than perpendicular to it (direction error), consistent with the
intuition that a linear model extrapolating a smoothed motion vector tracks *direction* changes
(recurvature) worse than it tracks *speed* changes.

### 9.4 Sample sizes

Every result above is computed on the same **1,732** test windows from **88** storms — a fixed,
frozen, disjoint-from-training test set. No metric in this document mixes sample counts across models.

---

## 10. Baseline Comparison — Which Wins, and Where Each Fails

**Best baseline for intensity: LightGBM**, by a clear and horizon-consistent margin (+19.8% over
persistence at 24h, and the best or tied-best baseline at every horizon). This is a real,
multi-horizon effect, not a one-metric artifact — LightGBM also has the lowest bias of the three.

**Best baseline for track: CLIPER-style Ridge regression**, marginally but consistently ahead of
LightGBM at every horizon, and by a larger, consistent margin ahead of Persistence. **LightGBM does
not win on track**, despite winning decisively on intensity — this is reported as a real, honest
finding, not smoothed over. A plausible reading (offered as interpretation, not as an additional
verified fact) is that track displacement over 6–24h is close to a **smooth, low-order function** of
motion and position — exactly what a compact linear/interaction model is suited to — while LightGBM's
extra flexibility (the full 160-column flattened window) does not find additional exploitable
structure and may be fitting noise in the wind/pressure lag features that are not actually
informative for *where* the storm goes next.

**Where each baseline fails:**

- **Persistence** (both tasks) fails hardest on storms that **recurve** — see the top-left panel of
  `track_examples.png`: a storm turning sharply northeast is predicted by every model (including
  persistence) to continue northwest, because none of these baselines model atmospheric steering
  flow, only the storm's own recent motion. This is an intrinsic limitation of an IBTrACS-only
  baseline, not a bug.
- **Ridge/CLIPER** (intensity) under-performs LightGBM at every horizon — a linear model cannot
  capture the (likely non-linear) relationship between the tendency/max-wind-so-far features and
  future intensity change as well as gradient-boosted trees can.
- **LightGBM** (track) is beaten by the simpler CLIPER model, and is the *only* baseline that loses
  to persistence at the shortest horizon (+6h) — evidence that, for this specific target and sample
  size, its extra capacity is not paying for itself.

**Are the differences meaningful?** The intensity margin (LightGBM vs. persistence, ~2 kt MAE at
24h, ~20%) is large relative to the metric's own scale and consistent across four horizons — this
reads as a real, useful effect. The track margin (CLIPER vs. LightGBM, ~1–3 km, ~1–2%) is much
smaller and could plausibly narrow or reverse with a different random seed or a larger sample; it is
reported as a genuine but modest finding, not a decisive one. No formal significance test (e.g. a
paired bootstrap over storms) was run in Phase 2 — this is listed as a limitation in §12, not
glossed over.

---

## 11. Visualizations

All in `ml/reports/figures/`, generated by `ml/scripts/make_phase2_plots.py` from the artifacts
`run_phase2_benchmark.py` already wrote (no model is re-fit to produce a plot).

| File | Content |
|---|---|
| `intensity_actual_vs_predicted.png` | Scatter, observed vs. predicted wind @ +24h, one panel per model, y=x reference line |
| `intensity_error_distribution.png` | Overlaid histograms of (predicted − observed) wind error @ +24h, all three models |
| `track_examples.png` | 4 randomly sampled test storms: observed history, forecast origin, each model's +6/+12/+18/+24h predicted track, and the actual future track — including one storm (top-left) that recurves sharply, illustrating the shared failure mode of all three baselines |
| `track_error_vs_horizon.png` | Mean and median great-circle error vs. horizon, one line per model |
| `model_comparison.png` | Headline (+24h) bar chart, both tasks side by side |
| `error_by_storm.png` | Per-storm mean error (best model per task), sorted — shows the spread across individual storms, not just the aggregate |

Every plotted quantity is either read directly from IBTrACS (OBSERVED DATA) or produced by one of the
six fitted baselines (MODEL PREDICTION) or a documented deterministic transform of the two (DERIVED
FEATURE, e.g. an error). No wind field, rainfall, or other unmeasured quantity is visualized anywhere.

---

## 12. Reproducibility

| Item | Value |
|---|---|
| Random seed | `42`, fixed in `ml/geostrom_ml/splits/split.py::RANDOM_SEED` and passed to every `Ridge`/`LGBMRegressor` constructor |
| Split determinism | The split rule itself is deterministic (season-block, no randomness); `ml/tests/test_splits.py::TestDeterminism` proves rebuilding it reproduces the frozen file exactly |
| Dataset version | `v1` (`ml/manifests/dataset_v1_manifest.json`) |
| Split version | `v1` (`ml/manifests/splits_v1.json`) |
| Feature version | `v1` |
| Model version | `v1` for all six baselines |
| Environment | Python 3.11.9; pinned package versions in `ml/requirements.txt` (pandas 2.3.3, numpy 2.4.6, scikit-learn 1.9.0, lightgbm 4.7.0, matplotlib 3.11.0, pyarrow 24.0.0, pytest 9.1.1, tabulate 0.10.0) |
| **Empirical reproducibility check** | `run_phase2_benchmark.py` was run twice, independently, end-to-end (fresh model fit both times). **Maximum absolute difference across all 24 benchmark records × all metric fields: 0.0** — bit-for-bit identical, not merely "within tolerance." |

**Running the pipeline from scratch:**

```
python ml/scripts/build_splits.py         # writes ml/manifests/splits_v1.json (frozen)
python ml/scripts/build_dataset.py        # writes $DATA_ROOT/datasets/v1/{train,val,test}.parquet
python ml/scripts/run_phase2_benchmark.py # writes ml/reports/phase2_benchmark_results.json + predictions
python ml/scripts/make_phase2_plots.py    # writes ml/reports/figures/*.png
python -m pytest ml/tests/ -v             # 80 tests
```

---

## 13. Testing

**80 tests, all passing** (`ml/tests/`, run via `pytest`). Coverage against every item the task
brief lists:

| Requirement | Test file / class |
|---|---|
| Storm-level split integrity | `test_splits.py::TestFrozenSplitIntegrity` (8 tests), `test_leakage.py::TestMaterialisedDatasetSplitLeakage` |
| Temporal causality | `test_features.py::TestPerTimestepFeatures::test_causality_future_mutation_does_not_change_past_features`, `test_leakage.py::TestWindowLevelCausality` |
| Feature leakage (explicit, adversarial) | `test_leakage.py` — includes a "does the test methodology actually detect a leak" self-check using a deliberately leaky construction |
| Missing values | `test_features.py::test_first_row_has_no_motion_or_tendency`, `test_tendency_requires_exact_lag_else_nan` |
| Longitude wrapping | `test_geo.py::TestWrapLonDiff`/`TestWrapLonDeg` — includes the two exact cases the task brief specifies (179→−179, −179→179) |
| Haversine calculation | `test_geo.py::TestHaversine` — zero-distance, known 1°-longitude value, antimeridian short-distance regression, symmetry, a real-world sanity check (NYC–London) |
| Prediction horizon construction | `test_features.py::TestSequenceWindows` (window counts, storm-boundary respect, gap handling, target correctness) |
| Baseline output shape | `test_baselines.py` (all 6 models, fit/predict shape and finiteness checks) |
| Metric calculation | `test_metrics.py` (MAE/RMSE/bias on hand-computed values, skill sign convention, track metrics incl. antimeridian and along/cross-track sanity) |
| Deterministic splitting | `test_splits.py::TestDeterminism` |

**A leakage test would fail if future information entered the feature set:** demonstrated directly
by `test_leakage.py::test_a_deliberately_leaky_construction_is_actually_detected`, which constructs
an intentionally leaky feature (`shift(-1)`, reading the future row) and asserts it changes when a
future row is mutated — proving the causality tests are not vacuously passing.

---

## 14. Limitations and Known Issues

- **No hyperparameter tuning.** Ridge `alpha=1.0` and LightGBM's hyperparameters are fixed,
  reasonable defaults, not selected via the validation set. A tuned model might close or widen the
  CLIPER-vs-LightGBM track gap in either direction. This is a deliberate Phase 2 scope decision (the
  goal is an honest baseline, not a maximally-tuned one) but is stated explicitly as a limitation.
- **No statistical significance testing.** The track skill differences (1–3 km) are small enough that
  a proper significance test (e.g. paired bootstrap over storms, not windows, to respect the
  clustering) would be needed before treating "CLIPER beats LightGBM" as a strong claim rather than a
  suggestive one. Not run in Phase 2.
- **Wind and pressure predictors only** — no SST, wind shear, or ocean heat content (unavailable in
  IBTrACS), so intensity skill is fundamentally bounded regardless of model choice, exactly as
  `PROJECT_REQUIREMENTS.md` §2.C anticipated.
- **Recurving storms are a shared blind spot.** All three track baselines rely on extrapolating
  recent motion; none has any representation of atmospheric steering flow, so all three fail
  similarly on sharp recurvature (see `track_examples.png`, top-left panel).
- **North Atlantic only**, 1980–2015 — no claim is made about generalisation to other basins or more
  recent seasons.
- **Antimeridian handling is unit-tested but not exercised by the real benchmark data** (no NA storm
  in this sample crosses ±180°) — correct by construction and by synthetic test, not by an
  end-to-end real-data proof. Flagged, not hidden, exactly as Phase 1 left it.
- **The `val` split is loaded but not used for model selection** in Phase 2 (all baselines use fixed
  hyperparameters). It is correctly disjoint from `train`/`test` and available for a future phase
  that does need it.

---

## 15. Reuse of Phase 1 Components

Per the task instruction to inspect and reuse validated Phase 1 work rather than duplicate it:

- **`ml/geostrom_ml/config.py`** (DATA_ROOT resolution and safety guard) — used unchanged.
- **The `keep_default_na=False` IBTrACS-loading rule**, found and fixed in Phase 1
  (`docs/PHASE_1_DATASET_VERIFICATION.md` §4), is reused verbatim in
  `ml/geostrom_ml/data/ibtracs.py::load_ibtracs_raw` — the single canonical implementation Phase 2
  code imports from.
- **The cross-basin-file duplicate-row handling**, also found and fixed in Phase 1 (verified
  byte-identical duplicates from storms crossing basin files), is reused verbatim, with the same
  identical-content assertion re-checked at every load (now raising an error rather than silently
  proceeding if a future IBTrACS release ever violates that assumption).
- **`ml/scripts/qc_gate.py`** (the Phase 1 artifact containing the original inline version of this
  loading logic) was **left unmodified** — it remains a correct, standalone historical record of the
  Phase 1 verification run. Phase 2 does not import from it; instead, the validated *logic* was
  extracted into the new canonical `ml/geostrom_ml/data/ibtracs.py` module, extended with the
  additional columns (`NUMBER`, `NAME`, `STORM_SPEED`, `STORM_DIR`, `DIST2LAND`, `LANDFALL`, etc.)
  Phase 2 feature engineering needs but Phase 1's QC-only loader did not select. This avoids both
  duplicating the validated logic and risking a regression in a working Phase 1 script.
- **No bug in Phase 1 code was found to be relevant to Phase 2** beyond the two already fixed and
  documented in Phase 1 (the `"NA"`-as-`NaN` sentinel collision, and the cross-file duplicate rows) —
  both are re-verified as still holding by every Phase 2 dataset build (`load_ibtracs_raw` raises if
  the duplicate-identity assumption is ever violated).

---

## 16. Recommended Next Step

Per `DEVELOPMENT_ROADMAP.md`, Phase 2's exit criterion — "a committed benchmark table showing at
least one model beating persistence at 24h for both track and intensity, on held-out storms, with a
leakage audit documented" — is met: **LightGBM beats persistence on intensity by 19.8%, and
CLIPER-style Ridge beats persistence on track by 11.4%, both at the headline 24h horizon, both on the
frozen, disjoint, 1,732-window test set**, with the leakage audit in §9/§13 above.

The recommended next phase is **P3 — Vertical Slice**: write these baseline predictions to storage
with `model_version` tags, stand up the minimal FastAPI endpoints, and deploy a bare-bones map view
showing predicted vs. actual track for a selected storm. This is the next item in the roadmap and
does **not** require the satellite pipeline (P4) to be complete first — consistent with the
parallel-branch scheduling rationale in `DEVELOPMENT_ROADMAP.md` §2.1.

**This phase does not proceed to Phase 3.** Per the task instructions, Phase 2 stops here.
