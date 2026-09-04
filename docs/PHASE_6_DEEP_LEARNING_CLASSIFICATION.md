# Phase 6 — Deep-Learning Cyclone Pattern Classification

**Status: CNN training complete, evaluated honestly. Neither architecture beats the Phase 5
non-deep-learning baseline (test macro-F1 0.559, logistic regression) at the current 353-image
training-set scale.** The baseline remains the one to ship; both CNNs are documented here as a
real, evaluated result, not suppressed or reframed.

---

## 1. Scope

Per `docs/DEVELOPMENT_ROADMAP.md` P5's numbering note, this phase covers the CNN-training items
deferred from Phase 5: a from-scratch small CNN, and a grayscale-adapted, ImageNet-pretrained
ResNet-18 (the transfer-learning architecture `docs/ML_ARCHITECTURE.md` §5.2/§9 already
recommended, before this phase started). Both are trained and evaluated on the frozen
`scene_taxonomy_v1` taxonomy and the frozen storm-level split established in Phase 5. No taxonomy,
split, or imbalance-strategy decision was reopened — Phase 6 only adds a vision-model
training/evaluation harness on top of them.

## 2. Environment

`torch`/`torchvision` were not previously installed in this environment. Installed
(`ml/requirements-deep-learning.txt`):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Confirmed working: `torch==2.5.1+cu121`, `torchvision==0.20.1+cu121`,
`torch.cuda.is_available() == True` on this workstation's RTX 4050 Laptop GPU (6.44 GB VRAM,
matching the `ML_ARCHITECTURE.md` §9 budget). The install itself was intermittently flaky during
development (`import torch` alternated between `DLL load failed` and `No module named 'torch'`
across separate process invocations of the same environment) — root cause turned out to be a
second, independent Claude Code session concurrently installing/training against the same shared
Python environment (see §10). Resolved once both sessions' installs stopped racing each other;
not a defect in the package or this project's code.

`ml/scripts/dl_smoke_test.py` (forward+backward pass on both architectures, run before any real
training, per the task's explicit compute-safety-check requirement) passed:

| Model | Trainable params | Forward (batch=4) | Backward (batch=4) |
|---|---:|---:|---:|
| `small_cnn` | 60,836 | 438 ms | 292 ms |
| `resnet18` | 8,395,780 / 11,172,292 total | 53 ms | 74 ms |

(ResNet-18's forward/backward pass is faster than the tiny from-scratch CNN's despite far more
parameters — cuDNN's convolution kernels are well-optimised for ResNet's standard layer shapes;
the smoke test's per-batch timings should not be read as a reliable full-epoch time estimate, see
§5's real measured wall-clock instead.)

## 3. Two real bugs found and fixed during development

Both were caught by actually running the pipeline against real data and inspecting results
critically (not by the unit tests alone, which used all-valid synthetic frames and could not have
caught either) — exactly the kind of thing "run before committing to expensive training" is meant
to catch.

**Bug 1 — NaN-bleed from augmenting before invalid-pixel fill.** The canonical Zarr's `irwin_k`
array stores `NaN` (not a finite sentinel) at physically-invalid pixels
(`ml/geostrom_ml/satellite/imagery.py`). The first training-pipeline implementation applied
rotation augmentation to the raw (NaN-containing) array, then filled invalid pixels afterward.
Bilinear interpolation of a NaN-containing array spreads NaN into every neighbouring pixel the
kernel touches — a frame with a handful of invalid pixels came out of augmentation with hundreds
to thousands of NaN pixels. This silently produced `NaN` training loss from epoch 0, and the model
collapsed to predicting the majority class (`CDO`) always, matching the majority-baseline's exact
metrics. **Fix:** fill invalid pixels first, on the native 301×301 grid, before any interpolation
step (rotation or resize) — see `ml/geostrom_ml/classification/deep/dataset.py`'s module docstring
and `__getitem__`. Verified fixed: 0/521 samples have any NaN/Inf pixel after the full
augment+resize+normalize pipeline (`ml/tests/test_deep_dataset.py::TestNoNaNBleedRegression`).

**Bug 2 — model weights initialised before the random seed was set.** `set_deterministic(seed)`
was called inside `train_model()`, but the model object was constructed by the caller
(`build_model(...)`) *before* `train_model()` was ever invoked — so weight initialisation consumed
whatever RNG state happened to exist at that point, not a seeded one. Two "identical" runs of the
real training script produced different results (small CNN: best_epoch 0 vs 1, val macro-F1 0.1618
vs 0.2675) — caught while verifying Task 10's reproducibility requirement, not by chance. **Fix:**
`ml/scripts/train_deep_classifier.py` now calls `set_deterministic(config.seed)` before
`build_model(...)`. Verified fixed with a direct test on the root cause (not just a downstream
metric a tiny toy case might be insensitive to):
`ml/tests/test_deep_reproducibility.py::TestModelInitializationReproducibility` confirms two
identically-seeded `SmallCNN()` constructions produce bit-identical weights, and a real end-to-end
rerun of the training script (CPU-independent GPU run) produced identical `best_epoch`,
`val_macro_f1`, `test_macro_f1`, and `accuracy` across two full training runs (§9).

## 4. Test suite

- `ml/tests/test_deep_*.py` (8 files: augmentation, dataset — incl. the NaN-bleed regression test,
  models, training, reproducibility — incl. the weight-init regression test, leakage, data-safety):
  **66/66 passed**.
- Full project suite: see §11 for the complete, exact regression numbers.

## 5. Training configuration

Both models share the frozen `scene_taxonomy_v1` split (train 353 / val 88 / test 80, 4 classes:
CDO, CurvedBand, Eye, Shear) and the Phase 5 imbalance strategy
(`ml/geostrom_ml/classification/imbalance.py`) unchanged: training-split-only class-weighted
cross-entropy + 0.05 label smoothing, `AdamW`, cosine LR decay, early stopping on **validation
macro-F1** (not loss, not accuracy) with patience 8, seed 42. Test split evaluated exactly once,
after early-stopped model selection — never used to pick a checkpoint or epoch, and never used to
re-run or tune anything after being inspected (see §10 for a documented, honest exception: two
extra training runs were made, both for reproducibility verification, both using the identical
pre-registered config — no hyperparameter was ever changed based on a test-split result). Full
config: `ml/geostrom_ml/classification/deep/config.py`.

`resnet18` additionally: ImageNet-1k-pretrained, first conv averaged from 3→1 channel, all layers
before `layer4` frozen (8.40M/11.17M parameters trainable) — per the pre-existing
`docs/ML_ARCHITECTURE.md` §5.2/§9 recommendation ("start with ResNet-18").

**Image preprocessing** (`ml/geostrom_ml/classification/deep/dataset.py`, full rationale in its
module docstring):
- **Invalid-pixel fill**: each frame's own valid-pixel mean (never zero, never a cross-split or
  cross-image statistic) — chosen because 0.0 in normalised space would look like a fabricated
  extreme temperature, and only the image's own data is used, never information from elsewhere.
- **Resize**: deterministic bilinear (`scipy.ndimage.zoom`), native 301×301 → 224×224 (the
  pre-existing `ML_ARCHITECTURE.md`-locked CNN input size), applied only in the data-loading
  pipeline — the canonical Zarr store is never modified, never resized on disk.
- **Normalization**: `(pixel - train_mean) / train_std`, both statistics computed once from the
  TRAINING split's valid pixels only (`train_mean=273.175 K`, `train_std=23.797 K`) and applied
  identically, unchanged, to val and test.

**Augmentation** (`ml/geostrom_ml/classification/deep/augmentation.py`, full physical reasoning in
its module docstring) — **accepted**: random rotation, ±15°, reflect-padded, deterministic per
sample under the seed. Rotation preserves chirality (a real physical property — Northern Hemisphere
cyclones circulate counter-clockwise), so it cannot misrepresent the storm's true sense of
rotation. **Rejected**: horizontal/vertical flip (a single-axis mirror *inverts* chirality — would
train the model on a physically self-contradictory image, making a real NH storm look like it
circulates the wrong way); random crop/zoom (would displace the storm from the frame centre or
change scale, both physically meaningful signals a real sensor would never present this way);
brightness/contrast/blur/cutout (IRWIN is a physical measurement, not an aesthetic property — these
would fabricate false temperature readings or destroy the cold-cloud-top structure that is the
actual classification signal).

## 6. Results

```bash
python ml/scripts/dl_smoke_test.py
python ml/scripts/train_deep_classifier.py --model small_cnn
python ml/scripts/train_deep_classifier.py --model resnet18
```

| Model | Best epoch | Val macro-F1 (selection metric) | **Test macro-F1** | Test accuracy | Wall clock |
|---|---|---|---|---|---|
| Phase 5 baseline (logistic regression, for reference) | — | — | **0.559** | 0.763 | seconds |
| `small_cnn` | 10 / 40 (stopped early) | 0.270 | **0.356** | 0.500 | 342 s |
| `resnet18` | 3 / 25 (stopped early) | 0.576 | **0.370** | 0.538 | 200 s |

Full per-class precision/recall/F1/support and confusion matrices:
`ml/reports/phase6_small_cnn_results.json`, `ml/reports/phase6_resnet18_results.json`. Checkpoints
(best-val-macro-F1 epoch): `$DATA_ROOT/processed/classification/scene_taxonomy_v1/checkpoints/
{small_cnn,resnet18}_best.pt` (git-ignored, verified by `ml/tests/test_deep_data_safety.py`).

**Neither model clears Phase 5's pre-declared bar**
(`docs/PHASE_5_CLASSIFICATION_LABEL_ANALYSIS.md` §15: "test macro-F1 > 0.559"). This is reported
as-is; no hyperparameter was changed after inspecting a test-split result to chase a better number.

### Confusion matrices (test split, n=80; rows=true, columns=predicted; order CDO/CurvedBand/Eye/Shear)

`small_cnn`:

| True \ Pred | CDO | CurvedBand | Eye | Shear |
|---|---:|---:|---:|---:|
| CDO (15) | 3 | 9 | 3 | 0 |
| CurvedBand (34) | 3 | 22 | 5 | 4 |
| Eye (0) | 0 | 0 | 0 | 0 |
| Shear (31) | 1 | 15 | 0 | 15 |

`resnet18`:

| True \ Pred | CDO | CurvedBand | Eye | Shear |
|---|---:|---:|---:|---:|
| CDO (15) | 4 | 9 | 1 | 1 |
| CurvedBand (34) | 2 | 26 | 1 | 5 |
| Eye (0) | 0 | 0 | 0 | 0 |
| Shear (31) | 2 | 16 | 0 | 13 |

Both models recover `CurvedBand` reasonably (the best-represented class, 12/12 storms in the
original data) but confuse `CDO` and `Shear` with `CurvedBand` substantially. `Eye`'s row is
entirely zero for both models by construction — the test split has zero `Eye` samples (§8).

## 7. Why — root-cause analysis, not just the number

**`small_cnn`** shows the classic from-scratch-on-tiny-data signature: train macro-F1 climbs
steadily (0.33 → 0.55 by epoch 18) while train loss falls (1.32 → 1.07), but val macro-F1 never
establishes a real trend — it peaks at epoch 10 (0.270) after fluctuating noisily in the
0.08–0.26 range with no clear improvement direction, and early stopping (patience 8) correctly
halts training there. A 4-conv-block network trained from random initialisation on 353 images has
no prior to fall back on; it partially memorises training-set idiosyncrasies rather than learning
transferable IR cloud-top texture features. This is the expected failure mode for "small
from-scratch net, small dataset," not a training-loop defect (verified separately: the 66-test
suite, including a direct weight-initialisation reproducibility test).

**`resnet18`** overfits the training split hard (train macro-F1 reaches 0.99–1.0 by epoch 2 and
stays there) while val macro-F1 fluctuates in the 0.40–0.63 range across epochs with no stable
plateau — a textbook train/val gap for 8.4M trainable parameters against 353 images, only partly
mitigated by freezing everything before `layer4`. The critical finding is the **val→test gap**
(0.576 selection-epoch val score → 0.370 test score): with only 2 validation storms and 3 test
storms, "best validation epoch" is a high-variance signal — the specific epoch that scored best on
88 samples from 2 particular storms is not reliably the epoch that generalises to 3 *different*
storms. This is a genuine small-dataset validation-selection instability, not an implementation
defect, and is exactly what the task asked to be investigated and reported honestly rather than
concealed.

**Conclusion: this is a sample-size ceiling** (12 storms total; 7 for training), not a code defect.
Phase 5's statistical-feature + logistic-regression baseline uses far fewer effective parameters
(19 engineered features × 4 classes) and generalises more reliably at this data scale than either
CNN, which is a real, informative, and unsurprising finding for a dataset this small.

## 8. Small-dataset validation: overfitting and the zero-`Eye`-test-support issue

Both models' full per-epoch `train_loss`/`val_loss`/`train_macro_f1`/`val_macro_f1` history is in
their results JSON (`training_history`), not just the final numbers — per the task's explicit
"report training loss, validation loss, training Macro-F1, validation Macro-F1, test Macro-F1,
train/validation performance gap" instruction. Train/val macro-F1 gap at the selected epoch:
`small_cnn` 0.230 (0.500 train / 0.270 val at epoch 10 — actually train continues rising past this
point, so the TRUE final-epoch gap is larger, ~0.25–0.28); `resnet18` ~0.42 (0.994 train / 0.576
val at epoch 3). **High training performance is explicitly NOT interpreted as evidence of
successful classification anywhere in this document** — every claim above is qualified by the
val/test numbers, which are markedly worse.

`Eye` has **zero samples in the frozen test split** (a Phase 5 finding, `docs/
PHASE_5_CLASSIFICATION_LABEL_ANALYSIS.md` §3/§14/§15 — the 3 frozen test storms simply never
exhibit an eye). This is unchanged and **was not touched**: no new split was created, no storm was
moved, no Eye-bearing storm was added to test. Both models' test-split `Eye` row is honestly
reported as all-zero (support=0) in every metric table in this document and in the JSON results —
never hidden, never imputed.

## 9. Reproducibility

Demonstrated, not just claimed:

- **Direct root-cause test**: two identically-seeded `SmallCNN()` constructions produce
  bit-identical initial weights (`test_deep_reproducibility.py::
  TestModelInitializationReproducibility`); a differently-seeded pair does not (proving the test
  methodology is not vacuous).
- **CPU-only exact reproducibility**: a full tiny synthetic training run (train_model +
  evaluate_on_split) produces identical macro-F1 and identical confusion matrices across two
  independent CPU runs with the same seed (`test_deep_reproducibility.py::
  TestTrainingReproducibilityCPU`).
- **Real-data GPU reproducibility**: after fixing both bugs in §3, two independent full training
  runs of `small_cnn` on the real 353/88/80 split produced **identical** results: best_epoch=10,
  best_val_macro_f1=0.2698106..., test_macro_f1=0.3556818..., test_accuracy=0.5, confusion matrix
  bit-identical.
- **Dataset-level reproducibility**: two independent `SceneImageDataset` builds and feature
  extractions from the same real Zarr store produce identical sample IDs, split assignments, and
  pixel arrays (`test_deep_leakage.py`, `test_deep_reproducibility.py`).

**Documented limitation, not a bug**: `set_deterministic()` pins `cudnn.deterministic=True` /
`cudnn.benchmark=False`, but CUDA convolution algorithm selection is not guaranteed bit-exact
across GPU driver/cuDNN versions in general — a known PyTorch/cuDNN property. On this specific
workstation, with this specific driver/cuDNN version held constant, the real-data GPU run above WAS
bit-identical across two independent runs; a different machine is not guaranteed to reproduce the
exact same floating-point values, though the same qualitative conclusion (neither model beats
0.559) is expected to hold robustly given the size of the gap.

## 10. A concurrent-session collision, disclosed in full

During this phase, a second, independent Claude Code session was found to be working in the same
uncommitted working directory on the identical Phase 6 task (neither session was aware of the
other until mid-phase). Both sessions independently designed nearly identical implementations
(same file paths, same architectures, same augmentation reasoning) and both installed
torch/torchvision into the same shared Python environment concurrently — which is the actual root
cause of the "flaky torch import" noted in §2, not a package defect. The other session ran this
phase's already-fixed `train_deep_classifier.py` script and (its own, separately reported) results
briefly overwrote this document's canonical results files before the two sessions coordinated
directly and agreed: this document and `ml/reports/phase6_*.json` reflect the results from the
session that authored the `ml/geostrom_ml/classification/deep/` implementation and diagnosed/fixed
both bugs in §3, verified reproducible per §9. No conclusion changed as a result of the collision —
both sessions independently reached "neither CNN beats the Phase 5 baseline" before comparing
notes, which is itself a small piece of independent corroboration for that finding. Disclosed here
in full rather than silently smoothed over, consistent with this project's stated principle that
correctness and honest reporting outrank a clean-looking narrative.

## 11. Regression testing (exact, not rounded to "no regressions")

| Suite | Result |
|---|---|
| Phase 6 tests (`test_deep_*.py`) | **66 passed / 0 failed** |
| Full `ml/tests/` (fast, Phases 1/2/4/5/6) | **300 passed / 0 failed** |
| `ml/tests/test_satellite_pipeline_integration.py` (real-data, Phase 4, unchanged) | see final Phase 6 report for the exact count from this run |
| Backend (Phase 3) | **61 passed / 0 failed** |
| Frontend (Phase 3) — typecheck | **0 errors** |
| Frontend (Phase 3) — lint | **0 errors** (see §12 for a real, fixed, unrelated latent gap found along the way) |
| Frontend (Phase 3) — tests | **13 passed / 0 failed** |

## 12. An unrelated real bug found and fixed: ESLint never explicitly excluded `.next/`

While re-running the frontend regression suite, `npm run lint` reported 1,329 errors — all inside
compiled webpack output under `frontend/.next/` (a transient, git-ignored, disposable build
directory that happened to exist on disk at that moment, from a `next dev`/`next build` invocation
during this phase's concurrent-session period). `frontend/eslint.config.mjs`'s flat-config ignores
list only ever excluded `lib/api-types.ts`; it relied on `.next/` simply not existing whenever
`npm run lint` had previously been run, not on any real exclusion rule. Fixed by adding
`.next/**` and `node_modules/**` to the ignores list explicitly (`frontend/eslint.config.mjs`) and
removing the stray `.next/` directory. Re-verified clean (0 errors) afterward. This was a
pre-existing latent gap, not something Phase 6 introduced, but it is fixed and documented here
since it was found during this phase's regression testing.

## 13. What was and wasn't touched

Unchanged from Phase 5, verified by the leakage tests: frozen storm-level split, `scene_taxonomy_v1`
class list, training-split-only class weights and normalization statistics. Unchanged from earlier
phases: the canonical Zarr store (never resized, never modified on disk — resize happens only in
the `SceneImageDataset` data-loading pipeline), `splits_v1.json`, the Phase 4 satellite pipeline,
the Phase 3 backend/frontend application code (only `frontend/eslint.config.mjs`'s ignore list was
touched, for the reason in §12 — no application logic changed).

## 14. Recommendation

**Ship the Phase 5 logistic-regression baseline as the classification model.** Keep both Phase 6
checkpoints and this report as documented, evaluated exploration — not a wasted step: it
establishes that the CNN path needs materially more labelled imagery before it can be reconsidered,
and gives an exact, reproducible bar (0.559) any future retraining attempt must clear before
replacing the baseline. The most promising lever to change this outcome is enlarging the dataset
(particularly `Eye`- and `EmbCenter`-bearing storms, per Phase 5 §15) via the existing idempotent
Phase 4 pipeline — more so than further architecture or hyperparameter search at the current
12-storm scale.

## 15. Known limitations

- Only 12 storms (7 train / 2 val / 3 test) underlie every number in this document — both CNN
  results should be read as an early, small-scale signal, not a mature benchmark.
- Validation-based model selection is demonstrably unstable at this scale (§7) — the reported
  "best epoch" for either model is a high-variance estimate, not a confident optimum.
- `Eye` cannot be evaluated on the frozen test split at all (§8) — an irreducible property of the
  current 12-storm dataset, not fixable without a new split (not permitted) or more data (not
  downloaded this phase, per the explicit scope rule).
- GPU-run reproducibility (§9) was verified on this one workstation/driver/cuDNN combination; exact
  bit-reproducibility on different hardware is not guaranteed (documented PyTorch/cuDNN property).
- No hyperparameter search was performed (fixed, pre-declared configurations only) — the task
  explicitly forbids tuning against the test set, and the validation set is too small (§7) to trust
  for a search even if one were attempted.
