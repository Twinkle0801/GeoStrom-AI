# ML ARCHITECTURE — GeoStrom AI

**Phase:** 0 (Architecture) · **Status:** Complete · No model has been trained.

---

## 1. Design Principles

| # | Principle | Enforcement |
|---|---|---|
| 1 | **Four independent models, not one.** Detection, classification, intensity, and track are separate problems with separate labels and metrics. | Separate packages, separate configs, separate checkpoints |
| 2 | **Every model is replaceable without touching anything downstream.** | A single `ModelRegistry` interface (§2) |
| 3 | **A baseline exists before a deep model, always.** | Benchmark harness refuses to report a deep model without its baseline on the same split |
| 4 | **Simplest model that clears the bar wins.** Complexity must be *earned* by a measured improvement over the baseline on the held-out split. | Comparison table is the deliverable, not a single model |
| 5 | **Train in PyTorch, serve in ONNX.** | Export step in every training pipeline |
| 6 | **6 GB VRAM is a hard constraint**, not a suggestion. | §9 compute budget |
| 7 | **Uncertainty is part of the output**, not an afterthought. | Every prediction record carries an uncertainty field |

### 1.1 Why "compare models" is the deliverable

For a research-extensible system, the valuable artefact is not one tuned network — it is a
**benchmark table**: identical split, identical metrics, N models per task. Ninety percent of the
architectural effort here goes into making that table cheap to produce. Anyone can then drop in a new
model and immediately know whether it helped.

---

## 2. The Modularity Contract

Every model in the system, from a persistence baseline to a Transformer, implements one interface:

```
BaseModel
  ├─ name            : str          unique identifier, e.g. "track_gru_v1"
  ├─ task            : Task         DETECTION | CLASSIFICATION | INTENSITY | TRACK
  ├─ input_spec      : Spec         declared shapes/dtypes/feature names
  ├─ output_spec     : Spec         declared shapes/dtypes/units
  ├─ fit(train, val) -> Metrics
  ├─ predict(batch)  -> Prediction  values + uncertainty + model_version
  ├─ save(path) / load(path)
  └─ export_onnx(path)              optional; falls back to a pickled estimator
```

**Consequences:**

- The training CLI takes a task and a model name; it does not know what a GRU is.
- The benchmark harness loops over registered models for a task and emits one comparison table.
- The batch-inference job writes `model_version` on **every** prediction row, so the API and UI can
  display which model produced a number, and two model versions can coexist in the database.
- The API depends on the *output spec*, never on the model. **Swapping a model is a config change and
  a re-run of batch inference — no API change, no frontend change.**

This contract is the reason the architecture satisfies "individual ML models can be replaced later".

---

## 3. Model Selection Strategy

For every task, three tiers are specified. **Tier 1 is mandatory. Tier 2 is the MVP target. Tier 3
is only attempted if Tier 2 is complete and time remains.**

| Tier | Purpose |
|---|---|
| **Tier 1 — Baseline** | Establishes the score any real model must beat. Cheap, fast, often surprisingly strong. |
| **Tier 2 — Hackathon model** | The model we actually ship. Chosen for trainability on 6 GB VRAM within hours. |
| **Tier 3 — Advanced** | Research direction. Documented, not built during the hackathon. |

---

## 4. Detection Architecture

### 4.1 Flow

```
 Satellite IR frame (from Zarr; canonical grid is 301×301 as of Phase 4 -- see
 docs/PHASE_4_SATELLITE_PIPELINE.md §9. Resize/crop to 224×224 happens HERE,
 in preprocessing, not in the canonical store -- the store keeps native-resolution
 physical values so this transform stays reproducible and swappable.)
        │
        ▼  PREPROCESSING
   dequantise to brightness temperature (K) [or read irwin_k directly -- see Phase 4 doc]
   resize/crop 301×301 -> 224×224 (deterministic, documented; not yet implemented)
   clip to physical range, normalise to zero mean / unit variance (train stats only)
   fill masked pixels with the per-image median
   augment (train only): rotation 0–360°, horizontal/vertical flip, small scale jitter
        │
        ▼  VISION MODEL
   CNN backbone → global pooling → dropout → single logit
        │
        ▼  σ(logit) → P(tropical cyclone present)
        │
        ▼  CALIBRATION (temperature scaling, fitted on validation)
        │
        ▼  { probability, calibrated_confidence, model_version }
```

### 4.2 Specification

| Aspect | Choice | Why |
|---|---|---|
| **Input** | 1 × 224 × 224 float32 IR brightness temperature | 224 matches ImageNet-pretrained backbones; single channel keeps VRAM low |
| **Output** | One probability in `[0,1]` | Binary task |
| **Loss** | Binary cross-entropy with `pos_weight` | `pos_weight` corrects the constructed positive/negative ratio without discarding data |
| **Metrics** | **PR-AUC (primary)**, ROC-AUC, precision/recall/F1 at the operating threshold, confusion matrix, calibration curve + Brier score | PR-AUC is the honest primary metric under imbalance; ROC-AUC looks flattering when negatives dominate. Calibration matters because the UI displays a confidence number |
| **Threshold** | Chosen on validation to maximise F1, then frozen | Never tuned on test |

### 4.3 Models

| Tier | Model | Rationale |
|---|---|---|
| **1 — Baseline** | Majority class, plus logistic regression on ~8 IR scalars (min BT, mean BT, std, cold-cloud fraction at 3 thresholds, radial BT gradient) | If a handful of scalars already separate the classes, a CNN is not needed — and this baseline trains in seconds. It also tells us whether the constructed negatives are *trivially* separable, which would indicate a flawed negative-sampling design |
| **2 — MVP** | **ResNet-18 or EfficientNet-B0, ImageNet-pretrained, first conv adapted to 1 channel** | Both fit easily in 6 GB at batch 64. Pretrained weights help even though IR imagery is far from ImageNet — early edge/texture filters transfer. EfficientNet-B0 has fewer parameters and generally slightly better accuracy-per-FLOP; ResNet-18 is faster to train and more predictable. **Recommendation: start with ResNet-18** for iteration speed, try EfficientNet-B0 as the second entry in the comparison table |
| **3 — Advanced** | Vision Transformer / Swin | **Explicitly discouraged for MVP.** ViTs need far more data or heavy augmentation to beat CNNs, and are memory-hungry. With a few tens of thousands of single-channel images, a small CNN is the better-founded choice. Revisit only with multi-basin, multi-channel data |

**Note on rotation augmentation:** rotating a storm-centric image by an arbitrary angle is physically
reasonable for detection (presence is rotation-invariant). It is **less safe for classification**,
where hemisphere-dependent rotation direction and shear orientation carry information — see §5.4.

### 4.4 Inference flow

Batch, offline: read frames from Zarr → preprocess → ONNX Runtime → write
`{sid, timestamp, probability, model_version}` to Postgres. The API reads rows. No model is loaded in
the request path.

---

## 5. Classification Architecture

### 5.1 The label problem comes first

```
   Dataset inspection (Phase 3)
        │
        ▼
   LABEL ANALYSIS  ── notebook, blocking gate ──────────────────┐
   • Which label fields actually exist?                         │
   • Class counts and imbalance ratio                           │
   • Merge classes with too few samples?                        │
   • Is the target ordinal or nominal?                          │
   • Are classes separable at all by a scalar baseline?         │
        │                                                       │
        ▼                                                       │
   VALID CLASSIFICATION PROBLEM  ← the class list is defined HERE, not earlier
        │                                                       │
        ▼                                                       │
   Model training ───────────────────────────────────────────────┘
```

**No class list is written into code before this gate passes.** The candidate tiers are defined in
PROJECT_REQUIREMENTS.md §2.B. The classes live in a config file, not in source, so changing them does
not require a code change.

**Merge rule, decided in advance:** any class with fewer than ~200 held-out-storm samples is merged
with its ordinal neighbour or dropped, and the merge is recorded in the dataset manifest. Deciding
this rule *before* seeing the results prevents post-hoc class engineering to flatter the metrics.

### 5.2 Specification

| Aspect | Choice | Why |
|---|---|---|
| **Input** | Same 1×224×224 IR tensor as detection; optionally concatenate the scalar state vector at the head | Shared preprocessing reduces code and bugs |
| **Output** | Probability vector over K classes, plus expected-category value for ordinal targets | Expected value is a more useful UI number than argmax alone |
| **Loss** | Class-weighted cross-entropy with label smoothing (0.05) | Weights address imbalance; smoothing prevents overconfidence on a noisy, human-analysed label |
| **Loss (ordinal upgrade)** | Ordinal regression (CORAL/CORN-style cumulative-logit head) | If the target is Saffir–Simpson stage, it is **ordinal**: confusing Cat 4 with Cat 5 is a far smaller error than confusing Cat 5 with a tropical depression. Plain cross-entropy treats both as equally wrong. Tier 3 |
| **Metrics** | **Macro-F1 (primary)**, per-class precision/recall, confusion matrix, **quadratic-weighted Cohen's κ**, mean absolute error in category units | Macro-F1 refuses to be fooled by majority-class accuracy. Quadratic-weighted κ and category-MAE are the right ordinal metrics and are frequently omitted in comparable work |

### 5.3 Handling class imbalance — layered, in this order

1. **Class weights in the loss** — inverse-frequency, damped by a square root to avoid wildly
   over-weighting the rarest class. First and cheapest lever.
2. **Balanced sampling** — a `WeightedRandomSampler` for minority classes. Use *either* weights *or*
   sampling as the primary lever; applying both aggressively over-corrects and destabilises training.
3. **Augmentation targeted at minority classes** — more aggressive transforms for rare categories.
4. **Class merging** — per the pre-declared rule in §5.1.
5. **Focal loss** — only if 1–4 leave the rare classes at near-zero recall. It adds a tunable that
   costs time; it is not a default.
6. **Never oversample before splitting.** Duplicating rows and then splitting puts identical samples
   in train and test. Sampling happens strictly inside the training loop.

**Report per-class recall, always.** A macro-F1 that looks acceptable can still hide a class with
zero recall, and the rare classes here are the intense storms — the ones that matter most.

### 5.4 Augmentation policy

| Transform | Detection | Classification | Reason |
|---|---|---|---|
| Random rotation 0–360° | ✅ | ⚠️ limited (±30°) | Storm rotation direction is hemisphere-dependent, and shear orientation is meaningful structure. Free rotation may destroy the signal the model should learn |
| Horizontal/vertical flip | ✅ | ❌ | A flip mirrors the rotational sense — physically it converts a Northern-Hemisphere storm into a Southern-Hemisphere one. Unsafe when structure is the target |
| Small translation (±10 px) | ✅ | ✅ | Simulates centre-fix uncertainty — physically realistic |
| Scale jitter (±10%) | ✅ | ✅ | Simulates parallax and storm-size variation |
| Brightness/contrast jitter | ⚠️ small | ❌ | Brightness temperature is a **physical quantity**. Shifting it changes the implied cloud-top height. Effectively fabricating physics |
| Cutout / random erasing | ✅ | ✅ small | Robustness to missing scan lines, which genuinely occur |

**This table is a deliberate correction of the default "throw the standard augmentation pipeline at
it" habit.** Standard vision augmentations encode assumptions that are false for calibrated
geophysical imagery.

---

## 6. Intensity Prediction Architecture

### 6.1 Flow

```
  Best-track sequence for one storm (6-hourly synoptic)
        │
        ▼  SLIDING WINDOW   L = 8 steps (48 h)   H = 4 steps (24 h)
        │
        ▼  FEATURE ENGINEERING  (strictly causal)
     per timestep: wind, pressure, lat, |lat|, lon(sin/cos), storm_speed, storm_dir(sin/cos),
                   Δwind_6/12/24h, Δpres_6/12/24h, max_wind_so_far, hours_since_genesis,
                   doy_sin/cos, dist2land, [+ IR scalars — L1 fusion]
                   [+ CNN embedding — L2 fusion, config flag]
        │
        ▼  SCALING (fitted on train split only)
        │
        ▼  SEQUENCE MODEL
        │
        ▼  DIRECT MULTI-OUTPUT HEAD → wind at +6, +12, +18, +24 h
        │
        ▼  UNCERTAINTY → empirical error quantiles per horizon
```

### 6.2 Specification

| Aspect | Choice |
|---|---|
| **Input** | `(batch, 8, F)` where F ≈ 20–30 engineered features |
| **Target** | Wind at +6/+12/+18/+24 h. **Primary: absolute wind.** Also train a variant predicting *Δwind* (intensity change) — see §6.4 |
| **Loss** | **Huber (smooth L1)** |
| **Metrics** | **MAE in knots (primary)**, RMSE, bias (mean error, to expose systematic under/over-forecasting), **skill score vs persistence**, per-horizon breakdown, plus recall on rapid-intensification cases |

**Why Huber and not MSE:** best-track intensity contains occasional large jumps and analysis
revisions. MSE lets a handful of outliers dominate the gradient; Huber is quadratic near zero and
linear in the tail, keeping sensitivity to normal errors while resisting outlier domination.

**Why MAE is the primary metric:** it is in knots, it is directly interpretable, it is the
metric operational verification uses, and it is not dominated by rare large errors.

**On MAPE — a deliberate rejection.** The brief lists MAPE as a candidate. **Do not use MAPE for
wind.** It divides by the true value, so an error of 5 kt on a 25 kt depression is penalised five
times harder than the same 5 kt error on a 125 kt major hurricane — exactly inverting the real-world
importance. It is also asymmetric, punishing over-forecasts more than under-forecasts. For pressure
it is nearly meaningless, since values span only ~880–1010 hPa and the percentage barely varies.
**MAE, RMSE, bias, and skill-vs-persistence are the correct set.**

### 6.3 Justifying L = 8 and H = 4

Neither number is arbitrary.

**Horizon H = 4 (24 h)** — chosen first, because it is the deliverable:
- 24 h is the shortest horizon at which a forecast is genuinely useful and is a standard verification
  lead time, making the result comparable to published numbers.
- Error grows rapidly with lead time while sample count falls, so 24 h is the best
  skill-per-available-sample point.
- Longer horizons (48/72 h) require storms lasting `L + H` steps, and each extension discards the
  shorter storms — which are also the more numerous ones.

**Input length L = 8 (48 h)** — chosen second, as the shortest window that captures what matters:
- 2 steps (12 h) gives only an instantaneous tendency — no curvature, no acceleration.
- 8 steps (48 h) spans the timescale over which intensification trends and recurvature develop, and
  covers two diurnal cycles, which modulate convection.
- Longer windows (16+) increase the `L + H` duration requirement to 5+ days, which **discards a large
  fraction of storms** and shrinks the training set, while adding parameters.

**Both are configuration values, and Phase 3 must test them empirically.** `L ∈ {4, 8, 12}` is a
short, cheap sweep. **TO VERIFY:** if the storm-duration distribution shows too few storms surviving
`L + H = 12` steps (72 h), reduce `L` to 4 — never reduce `H`, because `H` is the product.

### 6.4 Absolute wind vs Δwind — a real modelling choice

Predicting absolute wind lets the model lean on persistence (the current value is an excellent
predictor), producing good-looking MAE with little actual skill. Predicting **Δwind** forces it to
model the *change*, which is the hard, useful part, and makes the skill score honest.

**DECISION: train both, report both.** The absolute-wind model is the deliverable; the Δwind model is
the diagnostic that reveals whether real skill exists beyond persistence. Serving reconstructs
absolute wind from `current_wind + predicted_Δwind`.

### 6.5 Models

| Tier | Model | Rationale |
|---|---|---|
| **1 — Baseline** | **(a) Persistence:** wind is unchanged at all horizons. **(b) Linear/Ridge** on the flattened window. **(c) LightGBM** on flattened window features, one model per horizon | Persistence is the mandatory reference. **LightGBM is expected to be a strong contender**, possibly beating the deep models on this tabular problem — it handles missing values natively, needs no scaling, trains in seconds on CPU, and is hard to beat on ~10⁴–10⁵ tabular rows. Treating it as a "mere baseline" would be a mistake |
| **2 — MVP** | **GRU (1–2 layers, hidden 64–128) → dropout → dense multi-output head** | GRU over LSTM: ~25% fewer parameters, trains faster, and performs equivalently on short sequences — with 8 timesteps there is no long-range dependency for LSTM's extra gating to exploit. Trivially fits in VRAM |
| **3 — Advanced** | Temporal CNN with dilated causal convolutions; small Transformer encoder; seq2seq with attention; probabilistic head (quantile regression) | A Transformer over 8 timesteps is over-parameterised for the data — attention needs more sequence and more samples to pay for itself. Documented as a research direction, not an MVP path |

**Rapid intensification (RI)** — the scientifically interesting hard case (large intensification
within 24 h). It is rare and badly predicted by everything, including operational models. **Do not
build a separate RI model for MVP.** Instead, **report recall on RI cases as a diagnostic metric** of
the main model. This is honest, costs nothing, and surfaces the limitation rather than hiding it.
A dedicated RI classifier is Advanced scope.

---

## 7. Track Prediction Architecture

### 7.1 Flow

```
  P₁ → P₂ → P₃ → P₄ … P₈       observed positions (L = 8, 48 h)
        │
        ▼  convert to displacement space
     Δlat, Δlon per step  +  storm speed / direction  +  intensity state
        │
        ▼  SEQUENCE MODEL
        │
        ▼  8 outputs: (Δlat, Δlon) at +6, +12, +18, +24 h   [relative to the last observed point]
        │
        ▼  reconstruct absolute positions  P₉ P₁₀ P₁₁ P₁₂
        │
        ▼  UNCERTAINTY: empirical error radius per horizon → forecast cone
        │
        ▼  EVALUATE: Haversine great-circle error, decomposed along/cross-track
```

### 7.2 Specification

| Aspect | Choice | Why |
|---|---|---|
| **Input** | Same window as intensity; the two models share the feature pipeline | Shared code, shared cache, one bug surface |
| **Output** | **Cumulative displacement (Δlat, Δlon) from the last observed position**, per horizon | Displacements are near-stationary and roughly zero-mean; the model does not memorise basin geography; no ±180° discontinuity; and the output composes trivially into absolute coordinates |
| **Loss** | Huber on scaled displacements, **with longitude displacement weighted by cos(latitude)** | **This detail matters.** One degree of longitude is ~111 km at the equator but ~55 km at 60°N. Unweighted, the loss over-penalises high-latitude longitude error and under-penalises tropical error — the opposite of what is wanted. The cos-latitude weight makes the loss approximate true distance |
| **Metrics** | **Mean and median great-circle error (km) per horizon (primary)**, along-track / cross-track decomposition, **skill vs persistence and CLIPER-style baseline**, error distribution (not just the mean — track error is heavily right-skewed) | The decomposition separates a speed error from a direction error, which a single distance number conflates |

**Haversine, with R = 6371 km**, is the error metric. Never Euclidean degrees.

### 7.3 Should a baseline come before deep learning? — **Yes, emphatically**

Three reasons specific to track forecasting:

1. **Persistence is a genuinely strong short-range track forecast.** Storms have substantial
   momentum; extrapolating the last motion vector is hard to beat at 6–12 h. A deep model that does
   not beat it has learned nothing.
2. **CLIPER-style regression is the historical standard reference.** Skill in operational track
   forecasting is *defined* relative to climatology-and-persistence. Reporting a raw error in km
   without it is uninterpretable — a reader cannot tell whether 150 km at 24 h is good or terrible.
3. **The baselines de-risk the schedule.** Persistence and a CLIPER-style linear regression can be
   implemented and evaluated in a day, using only IBTrACS. **That gives the project a working,
   demonstrable, honestly-evaluated track forecast very early**, before any satellite data is
   downloaded. Everything after that is improvement rather than existential risk.

**DECISION: Phase 4 delivers persistence + CLIPER-style + LightGBM track and intensity forecasts from
IBTrACS alone.** This is the project's insurance policy.

### 7.4 Models

| Tier | Model | Rationale |
|---|---|---|
| **1 — Baseline** | **(a) Persistence** (constant velocity from the last two positions). **(b) CLIPER-style**: linear/ridge regression of displacement on current position, motion, intensity, day-of-year, and their interactions. **(c) LightGBM** per output | Mandatory references, all cheap |
| **2 — MVP** | **GRU encoder → dense multi-output head** producing 8 displacement values | Same rationale as intensity. Shares the training harness |
| **3 — Advanced** | Seq2seq GRU with attention; Temporal CNN; probabilistic output (bivariate Gaussian / quantile); physics-informed constraints (limiting implied translation speed to realistic values) | Real research directions, correctly out of hackathon scope |

### 7.5 Uncertainty — the forecast cone

**DECISION for MVP: empirical error radii.**

Compute the error distribution on the **validation** split at each horizon; take a high quantile
(e.g. the 67th percentile, matching the convention of a forecast cone containing roughly two-thirds
of historical errors); store these radii as model metadata; render circles of those radii around each
predicted point and hull them into a cone.

**Why this and not a fancier method:**
- It is honest — the radius is literally the model's observed historical error.
- It costs one pass over the validation set — no architectural change, no extra training.
- It is directly renderable and immediately interpretable to a viewer.
- It degrades gracefully: a bad model produces a visibly huge cone, which is the correct behaviour.

**Advanced:** deep ensembles (5 seeds, spread as uncertainty) or quantile regression with pinball
loss, both giving per-storm rather than per-horizon-average uncertainty. MC-dropout is noted but is
generally poorly calibrated and is not recommended as the primary path.

### 7.6 Honest expectations

Modern operational agency track forecasts achieve errors on the order of tens of km at 24 h,
supported by ensembles of global numerical weather prediction models running on supercomputers with
full atmospheric state. **A GRU trained on best-track positions on a laptop will not approach this,
and the project must not claim otherwise.** The realistic and defensible goal is:

> Beat persistence and a CLIPER-style baseline on held-out storms, using only historical track data
> and satellite imagery, and quantify the result honestly.

That is a legitimate, publishable-shaped result. Claiming operational competitiveness would not be.

---

## 8. Benchmark Harness

One component makes the whole "comparable models" principle real:

```
benchmark(task) →
   for each registered model:
       load frozen split manifest
       fit on train  ·  select on val  ·  evaluate once on test
       write metrics + predictions + config hash to results/<task>/<model>.json
   emit comparison table (Markdown + JSON)
```

Rules:
- **Test set is touched once per model, at the end.** All selection happens on validation.
- Every result records the git commit, dataset build version, config hash, and random seed.
- The comparison table is exported to the frontend's methodology page — the evaluation is a
  user-facing feature, not an internal artefact.

---

## 9. Compute Budget — RTX 4050, 6 GB VRAM

| Model | Batch | Est. VRAM | Est. train time | Verdict |
|---|---|---|---|---|
| Logistic / Ridge / LightGBM | — | CPU only | seconds–minutes | ✅ |
| ResNet-18 @ 224², 1ch | 64 | ~2.5–3.5 GB | 1–3 h | ✅ comfortable |
| EfficientNet-B0 @ 224², 1ch | 48 | ~3–4 GB | 2–4 h | ✅ fits |
| GRU (2×128) on `(8, 30)` | 256 | < 0.5 GB | minutes | ✅ trivial |
| ViT-B/16 @ 224² | 16 | ~5.5 GB+ | 8 h+ | ❌ marginal, and data-starved |
| End-to-end CNN+RNN on image sequences | 4 | > 6 GB | — | ❌ does not fit |

**Practices required by the 6 GB limit:** mixed precision (AMP) everywhere; gradient accumulation
instead of large batches; `num_workers` tuned for Windows (spawn-based multiprocessing is expensive —
start at 4); checkpoint only the best epoch; keep the Zarr store on a fast local (non-OneDrive) path.

**This budget is why the MVP recommends ResNet-18 and a GRU.** They are not compromises made from
ignorance of larger models — they are the correct choices for this data volume on this hardware.

---

## 10. Summary Recommendations

| Task | Baseline (mandatory) | **MVP model** | Advanced |
|---|---|---|---|
| **Detection** | Majority class + logistic regression on IR scalars | **ResNet-18 (pretrained, 1-channel)** + temperature scaling | EfficientNet-B0, ViT/Swin, self-supervised pretraining on unlabelled IR |
| **Classification** | Class prior + LightGBM on IR scalars | **ResNet-18 / EfficientNet-B0**, class-weighted CE + label smoothing | Ordinal (CORAL/CORN) head, ADT scene-type target, image+scalar multimodal head |
| **Intensity** | **Persistence** + Ridge + **LightGBM** | **GRU (2×128) → 4-horizon head**, Huber loss; absolute + Δwind variants | Temporal CNN, Transformer, quantile head, RI specialist, reanalysis predictors |
| **Track** | **Persistence** + **CLIPER-style regression** + LightGBM | **GRU (2×128) → 8-output displacement head**, cos-lat-weighted Huber | Seq2seq + attention, probabilistic output, deep ensembles, physics constraints |

**Cross-cutting:** every model exports to ONNX; every prediction carries `model_version` and an
uncertainty field; every result is produced by the same benchmark harness on the same frozen split.
