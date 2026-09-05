# Phase 8 — Track Prediction: GRU (Sequence Model)

**Status: GRU trained and evaluated honestly. It does NOT beat the Phase 2 CLIPER-style Ridge
baseline (the best Phase 2 track model) at any of the four horizons (6/12/18/24h), though it does
beat plain Persistence at the two longer horizons (18h/24h).** CLIPER-style Ridge remains the model
to ship for track. Both the GRU's win over persistence at long horizons and its loss to CLIPER
everywhere are documented here as real, evaluated results, not suppressed or reframed — consistent
with `docs/ML_ARCHITECTURE.md` §7.3's own stated expectation that "a deep model that does not beat
[persistence] has learned nothing" and its broader "baseline before deep learning" principle.

---

## 1. Objective

Add the next track-prediction capability the existing roadmap calls for on top of Phase 2's
validated tabular baselines (Persistence, CLIPER-style Ridge, LightGBM): a GRU sequence model that
consumes the same causal 48h input window and predicts (Δlat, Δlon) displacement at +6/+12/+18/+24h,
with the longitude component of its training loss weighted by cos(latitude) per
`docs/ML_ARCHITECTURE.md` §7.2. This phase does not recreate Phase 2 — it extends it with exactly
one new model, reusing every other piece of Phase 2's frozen infrastructure (split, features,
evaluation harness) completely unmodified.

## 2. Dataset

Identical to Phase 2/Phase 7 — no new split, no new download, no re-materialization. Verified from
the repository, not assumed:

| Split | Windows | Storms |
|---|---:|---:|
| Train | 6,109 | 288 |
| Val | 1,557 | 69 |
| Test | 1,732 | 88 |

Source: `$DATA_ROOT/datasets/v1/{train,val,test}.parquet` (Phase 2's own materialised IBTrACS North
Atlantic build, USA_WIND-based, 1980–2015). `train_track_gru.py` re-verifies (does not re-derive)
that every storm in each parquet file maps to that same split in the frozen manifest before training
begins.

## 3. Frozen split verification

Reused exactly as Phase 2/Phase 7 froze it (`ml/manifests/splits_v1.json`, `SPLIT_VERSION="v1"`):
storm/season-block split, basin=NA, train seasons 1980–2004, val 2005–2009, test 2010–2015, seed 42.
**The file was not modified, not regenerated, not reopened during Phase 8.**
`validate_split_integrity()` (Phase 2's own function, reused not reimplemented) is called at the
start of the training script and in Phase 8's own leakage tests
(`test_phase8_leakage.py::TestVector4And5And8NoSplitOverlapOrContamination`), confirming zero storm
overlap across splits on the real materialised data — train/val/test storm sets are pairwise
disjoint and each storm maps to the split the frozen manifest assigns it.

## 4. Input sequence definition

Unchanged from Phase 2/Phase 7: causal sliding windows of `L=8` historical 6-hourly observations
(48h), the same 20 per-timestep features (`ml/geostrom_ml/features/engineering.py::
PER_TIMESTEP_FEATURES`), already flattened by Phase 2 into 160 `x__<feature>__lag{k}` columns
(`flattened_feature_columns()` — the exact same feature set `LightGBMTrack` already uses, and the
same one `IntensityGRU` uses in Phase 7, per §7.2's own instruction: "Same window as intensity; the
two models share the feature pipeline"). `reshape_to_sequence()` — Phase 7's un-flattening transform
— is imported and reused unmodified (not duplicated) to build the chronological `(N, 8, 20)` input
tensor, since it is a pure reindexing of already-computed, already-causality-verified columns.
Missing values: train-only median imputation; normalization: `StandardScaler` fit on the training
split only — both mirroring `CliperTrack`'s and `IntensityGRU`'s established conventions.

## 5. Target definition

Displacement, never absolute coordinates, per `docs/PROJECT_REQUIREMENTS.md` §2.D (the same
project-wide decision `PersistenceTrack`/`CliperTrack`/`LightGBMTrack` already follow): 8 outputs —
`(Δlat, Δlon)` at each of +6/+12/+18/+24h, relative to the last observed position (`ref_lat`,
`ref_lon`), reusing the pre-existing `y_dlat_{h}h`/`y_dlon_{h}h` columns Phase 2 already
materialised (`dlat_col`/`dlon_col` from `ml/geostrom_ml/models/track_baselines.py`, imported not
redefined). `predict()` returns these same raw, unweighted degree displacements — identical output
format to every Phase 2 track baseline — so absolute-position reconstruction
(`ml/geostrom_ml/features/geo.py::displace`) and every downstream metric are computed identically
for all four models, with zero changes to `ml/geostrom_ml/evaluation/benchmark.py`.

## 6. Geospatial representation

Longitude degrees and latitude degrees are never treated as equivalent, per the explicit
requirement. Every longitude difference in this phase's code — inside the training loss, in the
early-stopping validation metric, and in the final test evaluation — goes through the project's
existing, already-tested geodesic utilities (`ml/geostrom_ml/features/geo.py`), never a raw
subtraction:

- **Longitude wrapping**: `wrap_lon_diff`/`wrap_lon_deg` (unmodified, Phase 2) — the *only*
  sanctioned way to compute a longitude difference in this codebase; already used to build the
  `y_dlon_{h}h` training targets themselves.
- **Antimeridian-safe reconstruction**: `displace()` (unmodified, Phase 2) reconstructs absolute
  position from a predicted displacement, wrapping longitude correctly across ±180°. Re-verified at
  the TrackGRU-integration level (not just the pre-existing `test_geo.py` unit tests) in
  `test_track_gru.py::TestAntimeridianHandling`: a synthetic storm at `ref_lon=179.5°` with a
  2°-eastward predicted displacement is confirmed to wrap to `−178.5°`, and the resulting great-circle
  error against a nearby true antimeridian-crossing position comes out under 50 km — not the
  ~40,000 km a naive unwrapped subtraction would compute. A paired adversarial test confirms the
  naive (wrong) computation really would blow up (`358.0°` raw difference vs. the correct `2.0°`),
  proving the check is not vacuous.
- **Error metric**: `haversine_km` (unmodified, Phase 2), `R=6371.0088 km` — used identically for the
  training-time validation selection metric (§9) and the final test evaluation (§11), never a flat
  Euclidean degree distance.
- **cos(latitude)-weighted loss**: implemented exactly as `ML_ARCHITECTURE.md` §7.2 specifies — see
  §8.

## 7. Model architecture

`ml/geostrom_ml/models/track_gru.py::TrackGRU`, conforming to the existing `BaselineModel` ABC
contract (`fit`/`predict`/`name`/`task`), extended (not modified) with an optional `val_df` kwarg on
`fit()` for early stopping — the same pattern Phase 7's `IntensityGRU` established:

- `nn.GRU(input_size=20, hidden_size=64, num_layers=1, batch_first=True, dropout=0.2)` → final
  layer's last hidden state `h_n[-1]` → `nn.Dropout(0.2)` → `nn.Linear(64, 8)`, reshaped to
  `(batch, 4, 2)` — one shared dense head producing all 4 horizons' `(Δlat, Δlon)` pairs at once, per
  `ML_ARCHITECTURE.md` §7.4's Tier-2 MVP specification ("GRU encoder → dense multi-output head
  producing 8 displacement values").
- One model, `track_gru_v1` — track's architecture is displacement-only by design (§7.2), so there is
  no absolute/delta variant split the way intensity has (§6.4); every Phase 2 track baseline already
  predicts displacement only.
- No Transformer/LSTM/TCN/attention/ensemble/hyperparameter sweep was used anywhere in this phase —
  the GRU is the only new architecture introduced, exactly matching the roadmap's Tier-2
  authorization and nothing beyond it (Tier-3 "Advanced" architectures in §7.4 remain explicitly out
  of scope).

## 8. Loss function

`CosLatWeightedHuberLoss` (`ml/geostrom_ml/models/track_gru.py`), implementing
`ML_ARCHITECTURE.md` §7.2 exactly: **"Huber on scaled displacements, with longitude displacement
weighted by cos(latitude)."** Exact mathematical formulation, per sample *i* and horizon *h*:

```
w_i          = cos(radians(ref_lat_i))                     -- fixed per window, from the
                                                                ALREADY-OBSERVED reference
                                                                position (last input timestep)
e_lat_{i,h}  = pred_dlat_{i,h}  - true_dlat_{i,h}
e_lon_{i,h}  = w_i * (pred_dlon_{i,h} - true_dlon_{i,h})

loss = mean_{i,h}[ Huber(e_lat_{i,h}) ]  +  mean_{i,h}[ Huber(e_lon_{i,h}) ]
```

`w_i` is computed from `ref_lat` — part of the causal input, never a future value — so this
introduces no leakage (verified: `test_phase8_leakage.py::TestRefPositionIsCausalNotFuture`).
**Why this and not a different geographic loss:** one degree of longitude is ≈111 km at the equator
but ≈55 km at 60°N; without the `cos(latitude)` weight, the loss over-penalises high-latitude
longitude error and under-penalises tropical longitude error — the opposite of physical reality —
exactly the failure mode §7.2 warns against. The weight is applied to the **loss only**; `predict()`
always returns raw, unweighted `(Δlat, Δlon)` degrees (§5), so no metric or stored prediction is ever
itself latitude-weighted.

**"Scaled displacements"**: the model's *input features* are train-split-only `StandardScaler`-scaled
(the established convention `IntensityGRU`/`CliperTrack` both already use); the `(Δlat, Δlon)`
**targets** are left in raw degrees, deliberately not independently re-standardized — lat and lon
displacements are the same physical unit, and once the longitude term is cos-weighted they are
already directly comparable (unlike, say, combining knots and kilometres, where standardization would
be load-bearing). This is a documented design choice, not an omission.

Verified, not just claimed, to have a real, non-vacuous effect:
`test_track_gru.py::TestCosLatWeightedGeospatialLoss` confirms an identical 2°-longitude error scores
a strictly *lower* loss at 60°N (`cos(60°)=0.5`) than at the equator, that a pure latitude error is
never weighted at all (only longitude is, per spec), and that a hypothetical unweighted loss would
(wrongly) score both latitudes identically — proving the weighting test is exercising a real effect.

## 9. Training configuration

`TrackGRUConfig`: `hidden_size=64, num_layers=1, dropout=0.2, huber_delta=1.0, learning_rate=1e-3,
weight_decay=1e-5, batch_size=256, max_epochs=200, early_stopping_patience=15, seed=42,
device="auto"` (resolved to `"cuda"` on this workstation) — deliberately kept at the same
conservative point in `ML_ARCHITECTURE.md` §7.4/§9's authorised "1–2 layers, hidden 64–128" range
that Phase 7's `IntensityGRU` used, rather than opening a hyperparameter search this phase does not
authorise. Optimizer: `AdamW`. `set_deterministic(seed)` (imported from Phase 7's `intensity_gru`
module, not duplicated) is called first, before any model construction. Early-stopping model
selection uses **mean great-circle error in km, averaged across all 4 horizons** — reconstructed via
the same `displace()`/`haversine_km()` functions the final evaluation uses — rather than raw training
loss, so the selection criterion is the same physically-meaningful metric `ML_ARCHITECTURE.md` §7.2
declares primary. Full per-epoch training history (train weighted-Huber loss, validation mean
track-error km) is recorded (`model.history`); the best-val-km epoch's weights are checkpointed and
restored before evaluation. **Validation-only model selection; the test set was not touched until
the single, final `evaluate_track_model()` call.**

## 10. Leakage controls

`ml/tests/test_phase8_leakage.py` (7 tests, all passed) covers every named vector, extending (not
duplicating) the existing adversarial philosophy already established in `ml/tests/test_leakage.py`
(Phase 2) and `ml/tests/test_phase7_leakage.py` (Phase 7):

| Vector | Test | Method |
|---|---|---|
| 1. Future observations cannot enter the input sequence | `TestVector1And2And6And7And9NoFutureInformation` | Mutate every `y_dlat`/`y_dlon`/`y_*_future_*` column to a sentinel; assert the reshaped input tensor is bit-identical before/after |
| 2. Future lat/lon cannot enter features | (same) | Same mutation test |
| 3. Target coordinates cannot enter features | `TestVector3And10NoTargetOrPostStormFeatureLeakage` | Asserts `FEATURE_COLS` contains no `y_`-prefixed or `"future"`-containing column |
| 4. Test storms cannot appear in training | `TestVector4And5And8NoSplitOverlapOrContamination` | Real materialised train/val/test parquet storm sets checked pairwise-disjoint and against the frozen manifest |
| 5. Validation storms cannot appear in training | (same) | Same real-data check |
| 6. Sequence windows cannot cross storm boundaries | (same as #1; row-wise reshape already proven in Phase 7) | Reused `reshape_to_sequence`, previously proven purely row-wise in `test_intensity_gru.py`/`test_phase7_leakage.py` |
| 7. Feature construction remains causal | (same as #1) | Same mutation test |
| 8. The frozen split manifest is respected | (same as #4) | Same real-data check |
| 9. Target generation uses only future observations after the cutoff | (same as #1) | Same mutation test, from the input side |
| 10. No post-storm summary information enters the features | (same as #3) | Same feature-column check |

**Adversarial tests included, per the explicit instruction:**
`test_adversarial_leaky_feature_list_is_detected` (a future-position column smuggled into a feature
list IS flagged), `test_a_deliberately_leaky_reshape_would_be_caught` (a hypothetical reshape
function that reads a future column DOES change output under mutation), and
`test_adversarial_overlap_is_caught_by_the_reused_validator` (a hand-built overlapping manifest DOES
raise `ValueError("Split integrity violated...")` from the reused, unmodified
`validate_split_integrity`). An additional Phase-8-specific check,
`TestRefPositionIsCausalNotFuture`, confirms the `ref_lat`/`ref_lon` values the cos(latitude) loss
weight depends on are unaffected by perturbing any future column — i.e. the loss weight itself cannot
leak future information.

## 11. Test methodology

`evaluate_track_model()` (Phase 2's own function, `ml/geostrom_ml/evaluation/benchmark.py`, **zero
changes made to this file**) is called exactly once on the frozen test split (2010–2015, 88 storms,
1,732 windows), computing `track_point_metrics()` (also unmodified) for each horizon: mean/median/p90
/max/RMSE great-circle error (km), along-/cross-track decomposition, and mean lat/lon error in
degrees. No new evaluation code was written for Phase 8 — the model was designed specifically so it
plugs into the existing harness unmodified (§5, §7).

## 12. GRU results

`track_gru_v1` — best_epoch=39 (of 55 run, early-stopped, patience 15), best validation mean
track error=128.322 km:

| Horizon | Mean track error (km) | Median (km) | RMSE (km) |
|---|---:|---:|---:|
| +6h | 37.007 | 28.293 | 50.005 |
| +12h | 85.002 | 65.589 | 112.141 |
| +18h | 142.782 | 109.522 | 184.881 |
| +24h | 209.505 | 164.253 | 267.612 |

Full per-horizon metrics (including along-/cross-track decomposition and lat/lon degree error) plus
complete training history: `ml/reports/phase8_track_gru_results.json`.

## 13. Persistence comparison

| Horizon | Persistence (km) | GRU (km) | GRU vs. Persistence |
|---|---:|---:|---:|
| +6h | 30.542 | 37.007 | GRU **worse** (−21.2%) |
| +12h | 80.610 | 85.002 | GRU **worse** (−5.4%) |
| +18h | 146.672 | 142.782 | GRU **beats** persistence (+2.7%) |
| +24h | 226.243 | 209.505 | GRU **beats** persistence (+7.4%) |

**Honest, mixed finding**: the GRU is worse than plain constant-velocity persistence at the two
shorter horizons, but genuinely beats it at 18h and 24h. Per §7.3's own stated bar ("a deep model
that does not beat [persistence] has learned nothing"), the GRU clears that bar only at the two
longer horizons — a real, partial, honestly-reported signal, not a uniform pass or fail.

## 14. CLIPER comparison

| Horizon | CLIPER-style Ridge (km) | GRU (km) | GRU vs. CLIPER |
|---|---:|---:|---:|
| +6h | 29.975 | 37.007 | GRU worse (−23.5%) |
| +12h | 75.463 | 85.002 | GRU worse (−12.6%) |
| +18h | 133.290 | 142.782 | GRU worse (−7.1%) |
| +24h | 200.445 | 209.505 | GRU worse (−4.5%) |

**The GRU does not beat CLIPER-style Ridge at any horizon.** The gap narrows steadily and
substantially as the horizon lengthens (23.5% worse at +6h → 4.5% worse at +24h) — the same
qualitative pattern Phase 7's intensity GRU showed against LightGBM (`docs/
PHASE_7_INTENSITY_PREDICTION.md` §13/§16), not exaggerated here as a win, just reported as the
closest the GRU comes to parity.

## 15. LightGBM comparison

| Horizon | LightGBM (km) | GRU (km) | GRU vs. LightGBM |
|---|---:|---:|---:|
| +6h | 30.727 | 37.007 | GRU worse (−20.4%) |
| +12h | 76.510 | 85.002 | GRU worse (−11.1%) |
| +18h | 134.942 | 142.782 | GRU worse (−5.8%) |
| +24h | 203.341 | 209.505 | GRU worse (−3.0%) |

**The GRU does not beat LightGBM at any horizon either.** Re-confirming the exact numbers from
`ml/reports/phase2_benchmark_results.json` (read fresh, not assumed from memory, per the task's
explicit instruction): CLIPER remains the single best Phase 2 track baseline at every horizon,
narrowly ahead of LightGBM, exactly as `docs/PHASE_2_FORECASTING_BASELINES.md`'s own "honest finding"
already reported (CLIPER beats LightGBM on track at every tested horizon, unlike intensity, where
LightGBM won) — unchanged, re-verified, not reopened.

## 16. Per-horizon results (headline summary)

24h headline, read directly from `ml/reports/phase8_track_gru_results.json::headline_24h`:

```json
{
  "persistence_mean_track_error_km": 226.243,
  "phase2_cliper_mean_track_error_km": 200.445,
  "phase2_lightgbm_mean_track_error_km": 203.341,
  "phase8_gru_mean_track_error_km": 209.505,
  "best_phase2_baseline": "track_cliper_v1",
  "pct_change_vs_best_phase2_baseline": -4.52,
  "beats_best_phase2_baseline": false
}
```

**GRU DOES NOT BEAT the best Phase 2 baseline (CLIPER-style Ridge, 200.445 km) — a −4.5% change in
mean track error at 24h**, reported exactly as computed, not clipped to zero or reframed as a win.
This is the required honest disclosure.

## 17. Reproducibility

Two independent full training runs of `ml/scripts/train_track_gru.py` (same seed, same config, same
machine/GPU) produced **byte-identical** `ml/reports/phase8_track_gru_results.json` output, confirmed
via `diff` — identical sample selection, identical per-epoch training history, identical
`best_epoch=39`, identical final test metrics for `track_gru_v1`, to full floating-point precision.
Same result class as Phase 7's `IntensityGRU` (no CUDA-nondeterminism observed on this workstation's
specific driver/cuDNN version); the same general limitation applies — exact bit-reproducibility on
different hardware is not guaranteed, though the qualitative conclusion (GRU beats persistence at
long horizons, never beats CLIPER/LightGBM) is not expected to change.

## 18. Test results

| Suite | Result |
|---|---|
| Phase 8 tests (`test_track_gru.py` [12] + `test_phase8_leakage.py` [7]) | **19 passed / 0 failed** |
| Full `ml/tests/` fast suite (Phases 1/2/4/5/6/7/8) | **341 passed / 0 failed** |
| `ml/tests/test_satellite_pipeline_integration.py` (real-data, Phase 4, unchanged) | see final Phase 8 status report for the exact count |
| Backend (Phase 3) | **61 passed / 0 failed** |
| Frontend — typecheck | **0 errors** |
| Frontend — lint | **0 errors** |
| Frontend — tests | **13 passed / 0 failed** |

No historical phase's tests were modified to make Phase 8 pass; no existing test was weakened or
deleted.

## 19. Limitations

- The GRU does not beat CLIPER-style Ridge (or LightGBM) at any of the four evaluated horizons — it
  is not the model to ship for track prediction at the current dataset scale.
- The GRU is also worse than plain Persistence at the two shortest horizons (+6h, +12h); it only
  clears the "beats persistence" bar at +18h/+24h.
- A single fixed configuration was used throughout (hidden_size=64, num_layers=1 — within the
  roadmap's stated "1–2 layers, hidden 64–128" range) — no hyperparameter search was performed, per
  the explicit instruction against tuning against the test set; a validation-only search remains a
  legitimate, un-taken next step.
- The cos(latitude) weighting uses the reference (last-observed) latitude as a fixed per-window
  constant; it does not account for the (much smaller) latitude drift that occurs over the course of
  a 24h forecast, which would require weighting inside the loss by the *predicted* latitude at each
  horizon — a plausible refinement, not attempted here to keep the loss simple and interpretable.
- GPU-run bit-reproducibility (§17) was verified on this one workstation's specific driver/cuDNN
  combination only.
- No IR-scalar or CNN-embedding fusion input was added — this phase used IBTrACS-only features
  exclusively, matching Phase 2/Phase 7's scope and the explicit instruction against downloading new
  satellite data.

## 20. Recommendation for Phase 9

**Ship Phase 2's CLIPER-style Ridge as the track model** (unchanged recommendation from Phase 2,
re-confirmed rather than reopened). Keep `TrackGRU` and this report as documented, evaluated
exploration — it establishes a real, reproducible sequence-model bar (209.505 km at +24h) any future
revisit must clear, and identifies the same short-horizon-underperformance pattern Phase 7's
intensity GRU showed, now confirmed across a second, independent task — a genuinely informative,
cross-task finding, not a one-off. The most promising levers for a future revisit, in rough order of
expected leverage: (1) a validation-only hyperparameter search within the already-authorised
1–2-layer/64–128-hidden range; (2) latitude-drift-aware loss weighting (§19); (3) the same
IR-scalar/CNN-embedding fusion inputs `ML_ARCHITECTURE.md` §7.1 already designs for, once more
satellite-fused storms exist. Per this phase's explicit instruction, Phase 9 work is not started
here.
