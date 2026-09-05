# Phase 7 — Intensity Prediction: GRU (Sequence Model)

**Status: GRU trained and evaluated honestly. It does NOT beat the Phase 2 LightGBM baseline at
any of the four horizons (6/12/18/24h).** LightGBM remains the model to ship for intensity.
Both GRU variants (absolute-wind and Δwind) are documented here as a real, evaluated result, not
suppressed or reframed — consistent with `docs/ML_ARCHITECTURE.md` §6.5's own stated expectation
that "LightGBM is expected to be a strong contender, possibly beating the deep models on this
tabular problem."

---

## 1. Objective

Add the next intensity-prediction capability the existing roadmap calls for on top of Phase 2's
validated tabular baselines: a sequence model (GRU) that consumes the same causal 48h input window
and predicts wind speed at +6/+12/+18/+24h, trained as both an absolute-wind model and a Δwind
(intensity-change) diagnostic model, per `docs/ML_ARCHITECTURE.md` §6.4. This phase does not
recreate Phase 2 — it extends it with exactly one new model family, reusing every other piece of
Phase 2's frozen infrastructure (split, features, evaluation harness) unmodified.

## 2. Existing Phase 2 baseline (unchanged, re-verified, not reproduced artificially)

Confirmed unchanged in `ml/reports/phase2_benchmark_results.json`:

| Model | 24h MAE (kt) | Skill vs. persistence |
|---|---:|---:|
| `intensity_persistence_v1` | 10.641 | — (reference) |
| `intensity_ridge_v1` | 9.591 | +9.9% |
| `intensity_lightgbm_v1` | **8.535** | **+19.8%** |

This is the existing benchmark this phase is measured against — not a result to reproduce, tune
towards, or alter. No Phase 2 file was rewritten to produce these numbers; they were read directly
from the pre-existing, committed Phase 2 report.

## 3. Phase 7 methodology

Per explicit instruction not to assume scope from memory, the exact Phase 7 capability was derived
by reading the current repository roadmap fresh rather than the conversation's own informal phase
numbering (which does not match the roadmap's P0–P12 sequence). `docs/DEVELOPMENT_ROADMAP.md` §5.2
("Recommended initial models") pre-registers, for **Intensity**, a Tier-2 "MVP" model: **GRU
(2×128) → 4-horizon head, Huber loss; absolute + Δwind variants** — and `docs/ML_ARCHITECTURE.md`
§6 gives its full specification (flow, input shape, loss, metrics, L/H justification, tiering).
This is the pre-existing, Phase-0-locked authorization that satisfies the constraint against using
a recurrent architecture unless the roadmap explicitly requires it, and it names the exact next
capability to add. Everything in this document implements that specification — nothing more,
nothing invented.

Reused, unmodified: `ml/manifests/splits_v1.json` (frozen split), the materialised
`$DATA_ROOT/datasets/v1/{train,val,test}.parquet` windows (Phase 2's own build, not rebuilt),
`ml/geostrom_ml/features/engineering.py` (feature engineering, unchanged), `ml/geostrom_ml/
evaluation/benchmark.py::evaluate_intensity_model` (Phase 2's evaluation harness, called exactly
once per model/horizon), `ml/geostrom_ml/splits/split.py` (split loading/validation).

Added, new: `ml/geostrom_ml/models/intensity_gru.py` (the `IntensityGRU` model, conforming to the
existing `BaselineModel` contract), one pure-addition function `ri_recall()` in `ml/geostrom_ml/
evaluation/metrics.py`, `ml/scripts/train_intensity_gru.py` (training/evaluation driver),
`ml/tests/test_intensity_gru.py` and `ml/tests/test_phase7_leakage.py` (new tests).

## 4. Dataset

Identical to Phase 2 — no new split, no new download, no re-materialization:

| Split | Windows | Storms |
|---|---:|---:|
| Train | 6,109 | 288 |
| Val | 1,557 | 69 |
| Test | 1,732 | 88 |

Source: `$DATA_ROOT/datasets/v1/{train,val,test}.parquet`, built by Phase 2's `ml/scripts/
build_dataset.py` from the real IBTrACS North Atlantic archive (1980–2015). `train_intensity_gru.py`
re-verifies (does not re-derive) that every storm in each parquet file maps to that same split in
the frozen manifest before training begins — defense-in-depth against silent drift between the two
artifacts, not a re-split.

## 5. Frozen split

Reused exactly as Phase 2 froze it (`ml/manifests/splits_v1.json`, `SPLIT_VERSION="v1"`): storm/
season-block split, basin=NA, train seasons 1980–2004, val 2005–2009, test 2010–2015, seed 42. The
file was **not modified, not regenerated, not reopened** during Phase 7. `validate_split_integrity()`
(Phase 2's own function, reused not reimplemented) is called at the start of the training script and
in Phase 7's own leakage tests, and confirms zero storm overlap across splits on the real
materialised data.

## 6. Target definition

Unchanged from Phase 2: wind speed (kt) at forecast horizons +6/+12/+18/+24h (`y_wind_abs_{h}h`,
absolute scale). **New in Phase 7** (not a redefinition, an added variant already specified in
`ML_ARCHITECTURE.md` §6.4): a parallel Δwind target (`y_wind_delta_{h}h`, intensity **change** from
the reference timestep) is also trained as a diagnostic model. `predict()` always returns
predictions on the absolute scale for both variants — the Δwind model's raw output is reconstructed
via `ref_wind + predicted_delta` before being handed to the same evaluation harness Phase 2 uses,
so both models are compared on identical, absolute-kt terms. The identity
`ref_wind + y_wind_delta_{h}h == y_wind_abs_{h}h` is verified against the real test parquet in
`test_phase7_leakage.py::TestVector8CorrectHorizonAlignment`.

## 7. Input features

Unchanged from Phase 2 — no new feature was engineered, computed, or downloaded. The same 20
per-timestep features (`ml/geostrom_ml/features/engineering.py::PER_TIMESTEP_FEATURES`: lat,
abs_lat, lon_sin/cos, USA_WIND, USA_PRES, storm_speed_kt, storm_dir_sin/cos, Δwind_6/12/24h,
Δpres_6/12/24h, max_wind_so_far, hours_since_genesis, doy_sin/cos, dist2land) across `L=8` lagged
timesteps, already flattened by Phase 2 into 160 `x__<feature>__lag{k}` columns
(`flattened_feature_columns()`, reused unmodified for both the tabular baselines and the GRU's own
feature list). The only new code is `reshape_to_sequence()`, a pure reindexing transform that
un-flattens these same 160 already-computed columns into a chronological `(N, 8, 20)` tensor for the
GRU — it computes no new feature and reads no column outside the existing `FEATURE_COLS` list.
Missing values: train-only median imputation (medians computed on the training split, applied
unchanged to val/test), mirroring `RidgeIntensity`'s existing pattern in `ml/geostrom_ml/models/
intensity_baselines.py`. Normalization: `StandardScaler` fit on the training split only, applied
unchanged to val/test (verified structurally in `test_intensity_gru.py::TestScalingIsTrainOnly`).

## 8. Forecast horizons

Unchanged from Phase 2: `H ∈ {6, 12, 18, 24}` hours (4 steps of 6h each), predicted **simultaneously**
by one shared multi-output head — a genuine architectural difference from Phase 2's tabular models,
which train one independent model per horizon. This follows `ML_ARCHITECTURE.md` §6.1's flow
diagram ("DIRECT MULTI-OUTPUT HEAD → wind at +6, +12, +18, +24h") and §6.3's pre-existing
justification for `L=8`/`H=4` (24h is the deliverable horizon; 48h input spans two diurnal cycles
without discarding too many shorter-lived storms) — neither number was re-derived or changed.

## 9. Model architecture

`ml/geostrom_ml/models/intensity_gru.py::IntensityGRU`, conforming to the existing `BaselineModel`
ABC contract (`fit`/`predict`/`name`/`task`), extended (not modified) with an optional `val_df`
kwarg on `fit()` for early stopping:

- `nn.GRU(input_size=20, hidden_size=64, num_layers=1, batch_first=True, dropout=0.2)` → final
  layer's last hidden state `h_n[-1]` → `nn.Dropout(0.2)` → `nn.Linear(64, 4)` (one shared dense
  head producing all 4 horizons at once), per `ML_ARCHITECTURE.md` §6.5's Tier-2 MVP specification
  ("GRU (1–2 layers, hidden 64–128) → dropout → dense multi-output head").
- Two independently trained instances: `intensity_gru_v1` (`target_mode="absolute"`) and
  `intensity_gru_delta_v1` (`target_mode="delta"`) — `predict()` on the delta variant always
  reconstructs absolute wind (`ref_wind + predicted_delta`) so it plugs into the same evaluation
  harness Phase 2 uses, unmodified; a separate `predict_delta()` exposes raw deltas for the
  RI-recall diagnostic only (§16).
- Loss: `nn.HuberLoss()`, per `ML_ARCHITECTURE.md` §6.2's explicit rejection of MSE (outlier
  sensitivity from best-track revisions/jumps) and MAPE (inverts real-world importance for wind).
- No CNN/ResNet/LSTM/Transformer was used anywhere in this phase — the GRU is the only new
  architecture introduced, exactly matching the roadmap's Tier-2 authorization and nothing beyond
  it (Tier-3 "Advanced" architectures in §6.5 remain explicitly out of scope).

## 10. Training configuration

`GRUIntensityConfig`: `hidden_size=64, num_layers=1, dropout=0.2, learning_rate=1e-3,
weight_decay=1e-5, batch_size=256, max_epochs=200, early_stopping_patience=15, seed=42,
device="auto"` (resolved to `"cuda"` on this workstation — confirmed
`torch.cuda.is_available() == True`). Optimizer: `AdamW`. `set_deterministic(seed)` is called
**first**, before any model construction — a lesson learned the hard way in Phase 6 (weight
initialization consumed unseeded RNG state there) and applied correctly from the start here. Full
per-epoch training history (train Huber loss, val MAE in knots) is recorded for both models
(`model.history`); the best-val-MAE epoch's weights are checkpointed and restored before
evaluation.

## 11. Validation strategy

Early stopping on **validation MAE in knots** (reconstructed to absolute scale for the delta
variant, so both variants are selected on the same, interpretable unit), patience 15 epochs, no
minimum-delta threshold. The validation split (2005–2009, 69 storms, 1,557 windows) is used only
for checkpoint selection — never for gradient updates, never for hyperparameter tuning against the
test set (the config above was fixed before any test-set number was inspected).

## 12. Test results

Evaluated exactly once each, via the unmodified `evaluate_intensity_model()` harness, on the frozen
test split (2010–2015, 88 storms, 1,732 windows):

**`intensity_gru_v1` (absolute-wind, the deliverable)** — best_epoch=144 (of 160 run,
early-stopped), best validation MAE=8.030 kt:

| Horizon | MAE (kt) | RMSE (kt) | Bias (kt) |
|---|---:|---:|---:|
| +6h | 4.208 | 5.784 | −0.584 |
| +12h | 5.521 | 7.717 | −0.432 |
| +18h | 7.199 | 10.049 | −0.316 |
| +24h | **8.826** | 12.343 | −0.329 |

**`intensity_gru_delta_v1` (Δwind, diagnostic)** — best_epoch=22 (of 38 run, early-stopped), best
validation MAE=7.513 kt; predictions reconstructed to absolute scale before scoring:

| Horizon | MAE (kt) | RMSE (kt) | Bias (kt) |
|---|---:|---:|---:|
| +6h | 2.967 | 4.188 | +0.357 |
| +12h | 5.279 | 7.401 | +0.735 |
| +18h | 7.303 | 10.242 | +1.068 |
| +24h | 9.099 | 12.741 | +1.414 |

Full per-model, per-horizon metrics plus complete training history: `ml/reports/
phase7_intensity_gru_results.json`.

## 13. Comparison with Phase 2

Full 5-model × 4-horizon MAE (kt) comparison table (`comparison_vs_phase2` in the results JSON):

| Horizon | Persistence | Ridge | **LightGBM** | GRU (abs) | GRU (Δ) |
|---|---:|---:|---:|---:|---:|
| +6h | 3.210 | 3.080 | **2.881** | 4.208 | 2.967 |
| +12h | 6.094 | 5.492 | **4.973** | 5.521 | 5.279 |
| +18h | 8.531 | 7.707 | **6.823** | 7.199 | 7.303 |
| +24h | 10.641 | 9.591 | **8.535** | 8.826 | 9.099 |

**Honest finding: the Phase 2 LightGBM baseline has the lowest MAE at every single horizon
(6/12/18/24h), not only the 24h headline.** Neither GRU variant beats it anywhere.

Headline (24h): `beats_lightgbm_baseline: false`, GRU (abs) MAE = 8.826 kt vs. LightGBM's 8.535 kt
— a **−3.4% change** (i.e., 3.4% *worse*, computed as `100*(lgbm_mae - gru_mae)/lgbm_mae`, reported
exactly as it is, not clipped to zero or reframed as a win). This is the required honest disclosure:
Phase 7's model performs worse than the Phase 2 LightGBM baseline, reported as-is, per the explicit
instruction not to fabricate an improvement.

A genuine, interesting pattern emerges rather than a flat "GRU is worse everywhere": the
**absolute-wind GRU's relative gap to LightGBM narrows sharply as the horizon lengthens** — it is
~46% worse at +6h (4.21 vs. 2.88 kt) but only ~3% worse at +24h (8.83 vs. 8.54 kt). See §16 for the
root-cause discussion.

## 14. Leakage validation

`ml/tests/test_phase7_leakage.py` (14 tests, all passed) covers every named vector, extending
(not duplicating) the existing adversarial philosophy already established in Phase 2's `ml/tests/
test_leakage.py` and `ml/tests/test_splits.py`:

| Vector | Test | Method |
|---|---|---|
| 1. No future target leakage | `TestVector1And2And6NoFutureInformation` | Mutate every `y_*` column to a sentinel value; assert `reshape_to_sequence()`'s output is bit-identical before/after |
| 2. No future track information | (same) | Same mutation test — the reshape reads only `FEATURE_COLS`, which contain no track-future columns |
| 3. No cross-storm contamination | `TestVector3NoCrossStormContamination` | Two interleaved synthetic storms, each row's own index threaded through every feature; every reshaped row equals only its own index, never a neighbour's |
| 4. No train/test storm overlap | `TestVector4And5NoSplitOverlapOrContamination` | Real materialised train/val/test parquet storm sets checked pairwise-disjoint and against the frozen manifest |
| 5. No validation/test contamination | (same) | Same real-data check, extended to val |
| 6. No feature construction using future observations | (same as #1) | Same mutation test |
| 7. No target-derived feature leakage | `TestVector7NoTargetDerivedFeatureLeakage` | Asserts `FEATURE_COLS` contains no `y_`-prefixed or `"future"`-containing column |
| 8. Correct horizon alignment | `TestVector8CorrectHorizonAlignment` | Verifies `ref_wind + y_wind_delta_{h}h == y_wind_abs_{h}h` on the real test parquet, for all 4 horizons |

**At least one deliberately-leaky construction is proven to fail/be-caught in every category**, per
the explicit instruction, so these tests are not vacuously passing:
`test_adversarial_leaky_feature_list_is_detected` (a target column smuggled into a feature list IS
flagged), `test_a_deliberately_leaky_reshape_would_be_caught` (a hypothetical reshape function that
reads a target column DOES change output under mutation, proving the equality check is sensitive),
`test_adversarial_overlap_is_caught_by_the_reused_validator` (a hand-built overlapping manifest DOES
raise `ValueError("Split integrity violated...")` from the reused, unmodified
`validate_split_integrity`).

## 15. Reproducibility

Two independent full training runs of `ml/scripts/train_intensity_gru.py` (same seed, same
config, same machine/GPU) produced **byte-identical** `ml/reports/phase7_intensity_gru_results.json`
output, confirmed via `diff` — identical sample selection, identical per-epoch training history,
identical `best_epoch`, identical final test metrics for both `intensity_gru_v1` and
`intensity_gru_delta_v1`, to full floating-point precision. This is a *stronger* determinism result
than Phase 6's CNN case, which carried a documented CUDA-nondeterminism caveat; no such caveat was
observed here (`torch.backends.cudnn.deterministic=True`/`benchmark=False` were set, as in Phase 6,
and this workstation's specific driver/cuDNN version held constant across both runs — exact
bit-reproducibility on different hardware is not guaranteed, per the same general PyTorch/cuDNN
property noted in Phase 6, though it is not expected to change the qualitative conclusion given the
size of the gap in §13). Additionally verified at the unit level:
`test_intensity_gru.py::TestDeterminism` (identical seed → identical predictions on synthetic data;
different seed → different predictions, proving the check is not vacuous).

## 16. Error analysis

**Why the absolute-wind GRU underperforms LightGBM most at short horizons.** At +6h, the correct
answer is dominated almost entirely by the current wind value (persistence alone gets 3.21 kt MAE);
LightGBM, with direct per-tree access to the raw `lag0` wind value as one of 160 flat features, can
essentially learn "copy this column, adjust slightly" with very little modelling overhead. The
GRU must instead compress the entire 8-step, 20-feature window through a shared recurrent
bottleneck before its dense head can express even that same near-identity mapping — a real
architectural disadvantage for the "easy," persistence-dominated short-horizon case, not a bug. As
the horizon lengthens, the *raw* task gets harder for every model (all MAEs rise), the value of the
current single wind value degrades, and the sequence trend information the GRU is explicitly built
to exploit becomes relatively more valuable — hence the narrowing gap (46% worse at +6h → 3% worse
at +24h, §13).

**The Δwind (delta) variant behaves differently and is worth reading separately.** It wins on bias
direction less cleanly (positive bias, i.e. systematic slight over-forecasting of intensification,
at every horizon, growing with lead time: +0.36 kt at +6h → +1.41 kt at +24h) whereas the absolute
model is consistently mildly conservative (negative bias, roughly −0.3 to −0.6 kt at every horizon).
Both are honestly reported, not cherry-picked.

**RI (rapid intensification) recall — a genuine, unflattering diagnostic finding.** Using the
delta model's raw predictions and NHC's standard 30 kt/24h RI threshold (`ri_recall()` in
`ml/geostrom_ml/evaluation/metrics.py`, a pure-addition function):

| Horizon | True RI cases (test) | Recall |
|---|---:|---:|
| +6h | 0 | undefined (no true RI cases at this horizon) |
| +12h | 9 | **0.0** |
| +18h | 35 | **0.0** |
| +24h | 58 | **0.0** |

**The GRU delta model never correctly flags a single true RI case at any horizon where RI cases
exist in the test set.** This is reported plainly, not hidden — and matches `ML_ARCHITECTURE.md`
§6.5's own prior expectation ("It is rare and badly predicted by everything, including operational
models. Do not build a separate RI model for MVP."). No dedicated RI model was built, per that same
guidance; RI recall is used here exactly as prescribed — as a diagnostic, never as a headline metric
or as the basis for a claimed capability.

## 17. Known limitations

- The GRU does not beat the Phase 2 LightGBM baseline at any of the four evaluated horizons — it is
  not the model to ship for intensity prediction at the current dataset scale.
- RI recall is 0.0 at every horizon with true RI cases in the test set — the model has no
  demonstrated skill at detecting rapid intensification, an intrinsically rare and hard event
  (§16), consistent with the roadmap's own prior expectation rather than a new finding this phase
  discovered independently.
- Both GRU variants were trained with the single fixed configuration specified in
  `ML_ARCHITECTURE.md` §6.5 (hidden_size=64, num_layers=1) — no hyperparameter search (e.g. 2 layers
  or hidden_size=128, both within the roadmap's stated "1–2 layers, hidden 64–128" range) was
  performed, per the explicit instruction against tuning against the test set; a search using only
  the validation split remains a legitimate, un-taken next step.
- GPU-run bit-reproducibility (§15) was verified on this one workstation's specific driver/cuDNN
  combination; exact bit-reproducibility on different hardware is not guaranteed, though the
  qualitative conclusion is not expected to change.
- No IR-scalar or CNN-embedding fusion (`ML_ARCHITECTURE.md` §6.1's "L1 fusion"/"L2 fusion" boxes)
  was added — this phase used IBTrACS-only features exclusively, matching Phase 2's scope and the
  explicit instruction against downloading unrelated new datasets.

## 18. Exit criteria

| # | Criterion | Met? |
|---|---|---|
| 1 | Phase 7 scope derived from the actual current roadmap, not assumed from memory | ✅ (§3) |
| 2 | Builds on, does not recreate, Phase 2 | ✅ — reuses split/features/harness unmodified (§3–9) |
| 3 | Frozen split never modified | ✅ — `splits_v1.json` untouched, re-verified (§5) |
| 4 | No leakage; storm-level separation preserved | ✅ — 14/14 dedicated tests + reused Phase 2 tests pass (§14) |
| 5 | Test set isolated until final evaluation, never tuned against | ✅ — config fixed before any test-set inspection (§11) |
| 6 | Deterministic, reproducible training | ✅ — byte-identical across 2 independent runs (§15) |
| 7 | Model versioned with explicit config/metadata | ✅ — `intensity_gru_v1`/`intensity_gru_delta_v1`, full config recorded (§10, §12) |
| 8 | Honest comparison against Phase 2 LightGBM benchmark | ✅ — reported as NOT beating it, at every horizon (§13) |
| 9 | No fabricated improvement | ✅ — negative result stated plainly (§13, §16) |
| 10 | Regression suites run and reported with exact counts, not just Phase 7's own tests | ✅ (§19 of the final status report) |
| 11 | Only the roadmap-authorized architecture (GRU) used; no CNN/ResNet/LSTM/Transformer | ✅ (§9) |
| 12 | Documentation added without overwriting historical Phase 2 docs | ✅ — this document is new; Phase 2's doc untouched |

**All 12 exit criteria are met.** This does not mean the model is a success by MAE — it means the
phase was executed correctly and evaluated honestly, which is the actual exit bar.

## 19. Recommendation for Phase 8

**Ship Phase 2's LightGBM as the intensity model.** Keep both `IntensityGRU` variants and this
report as documented, evaluated exploration — it establishes a real, reproducible sequence-model
bar (8.826 kt at +24h) any future revisit must clear, and identifies a concrete, informative
failure mode (short-horizon underperformance vs. a tabular model with direct access to the current
value; zero RI recall) rather than a wasted step. The most promising levers for a future revisit,
in rough order of expected leverage, are: (1) a validation-only hyperparameter search within the
roadmap's already-authorized 1–2-layer/64–128-hidden range (§17); (2) the IR-scalar/CNN-embedding
fusion inputs `ML_ARCHITECTURE.md` §6.1 already designs for, once more satellite-fused storms exist
(Phase 4's known limitation); (3) more training data generally, following the same "sample-size
ceiling" pattern already observed in Phase 6. Per this phase's explicit instruction, Phase 8 work
is not started here.
