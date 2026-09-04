"""Phase 6: deep-learning cyclone scene-pattern classification.

Extends `ml/geostrom_ml/classification/` (Phase 5's frozen taxonomy,
classification index, and evaluation metrics -- all reused unchanged) with
PyTorch models trained on the same frozen storm-level split.

Nothing in this subpackage creates a new split, a new taxonomy, or touches
the canonical Zarr store. It reads `classification_index.parquet` and the
existing `irwin_k`/`valid_mask` Zarr arrays exactly as Phase 5's baselines
did.
"""
