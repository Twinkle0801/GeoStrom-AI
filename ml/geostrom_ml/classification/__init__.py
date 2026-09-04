"""Phase 5: Scene-label audit and satellite classification dataset/baselines.

Operates entirely on Phase 4's already-fused sample index
(`$DATA_ROOT/processed/satellite/<version>/sample_index.parquet`) and
canonical Zarr store. This package does not ingest, download, or fuse
anything new -- it audits, taxonomizes, and evaluates what Phase 4 already
produced.
"""
