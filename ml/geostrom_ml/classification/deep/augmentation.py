"""Phase 6: physically-defensible, deterministic-under-seed augmentation.

Every candidate the Phase 6 task names is evaluated explicitly below.
Nothing is applied "because it's standard computer-vision practice" --
each has a stated physical justification or a stated rejection reason.

============================================================================
ACCEPTED
============================================================================

**Random rotation (+/- 15 degrees)**: ACCEPTED. Storm-centric HURSAT-B1
imagery has no meaningful "true north" for the purpose of Scene-pattern
classification -- a CDO, an eye, or a curved band is defined by its shape
relative to the storm centre, not by compass bearing. Critically, rotation
(unlike reflection) PRESERVES CHIRALITY: a counter-clockwise-spiralling band
in the Northern Hemisphere still spirals counter-clockwise after any
rotation angle. Restricted to a modest +/-15 degrees (not the full 360)
purely to limit the corner/edge fill artefacts introduced when a square
frame is rotated (see `_rotate_reflect` below) and to keep the augmented
image close to the real HURSAT-B1 viewing geometry distribution, not
because larger angles would be physically wrong.

**Intensity normalization**: ACCEPTED, but treated as required
PREPROCESSING, not a stochastic augmentation -- it is applied identically,
deterministically, to every image (train/val/test alike), computed from
TRAIN-split statistics only (see `dataset.py`). Physically meaningful: the
underlying quantity (Kelvin brightness temperature) is preserved; only its
scale is changed for numerical stability.

============================================================================
REJECTED
============================================================================

**Horizontal flip**: REJECTED. A horizontal (left-right) mirror INVERTS
apparent chirality -- a real Northern Hemisphere tropical cyclone's
counter-clockwise circulation (a genuine Coriolis-driven physical
asymmetry, not an imaging artefact) would appear to spiral clockwise after
a horizontal flip, i.e. it would look like a storm from the opposite
hemisphere. This is not a cosmetic difference: curved-band orientation is
part of what a Dvorak-family classifier is meant to read from the image.
Applying this augmentation would train the model on physically
self-contradictory examples.

**Vertical flip**: REJECTED, same reasoning as horizontal flip -- any
single-axis mirror inverts chirality. (A COMBINED horizontal+vertical flip
is equivalent to a 180-degree rotation, which -- unlike a single flip --
DOES preserve chirality; it is not implemented here only because it adds no
value beyond what continuous random rotation already covers.)

**Random crop / zoom**: REJECTED. HURSAT-B1 frames are already storm-centred
at a fixed physical scale (docs/PHASE_4_SATELLITE_PIPELINE.md); an
arbitrary crop/zoom would shift the storm centre out of the frame centre or
change the effective spatial scale, both of which are physically
meaningful signals (centre position and extent) that a real satellite
sensor would never present this way.

**Colour/contrast jitter, Gaussian blur, cutout**: REJECTED. IRWIN is a
single physical channel (Kelvin brightness temperature); the value at every
pixel is a measurement, not an aesthetic property, so brightness/contrast
jitter would fabricate physically false temperature readings. Blur/cutout
would destroy the fine cold-cloud-top structure (eye/band boundaries) that
IS the classification signal.

============================================================================
DETERMINISM
============================================================================

All randomness here is driven by a `numpy.random.Generator` seeded
explicitly per-sample (`RANDOM_SEED + sample_index`) rather than a shared
global generator, so:
  (a) reruns with the same seed reproduce the exact same augmented images
      (Phase 6 Task 10's reproducibility requirement), and
  (b) DataLoader worker processes (if `num_workers > 0`) cannot each
      silently re-derive a different, worker-order-dependent random stream.
"""

from __future__ import annotations

import numpy as np

MAX_ROTATION_DEG = 15.0


def rotate_reflect(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a 2D array by `angle_deg`, filling revealed corners by
    reflecting the image at its own border (never a fabricated constant
    temperature) -- deterministic given `angle_deg`.
    """
    from scipy.ndimage import rotate as ndi_rotate

    return ndi_rotate(image, angle_deg, reshape=False, order=1, mode="reflect").astype(image.dtype)


def sample_rotation_angle(sample_index: int, seed: int) -> float:
    """Deterministic per-sample rotation angle in [-MAX_ROTATION_DEG, +MAX_ROTATION_DEG]."""
    rng = np.random.default_rng(seed + sample_index)
    return float(rng.uniform(-MAX_ROTATION_DEG, MAX_ROTATION_DEG))


def augment_train_image(kelvin: np.ndarray, valid_mask: np.ndarray,
                        sample_index: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Apply the one accepted stochastic augmentation (rotation) to a
    training image + its mask together, so they stay pixel-aligned."""
    angle = sample_rotation_angle(sample_index, seed)
    if angle == 0.0:
        return kelvin, valid_mask
    rotated_k = rotate_reflect(kelvin, angle)
    rotated_mask = rotate_reflect(valid_mask.astype("float32"), angle) > 0.5
    return rotated_k, rotated_mask
