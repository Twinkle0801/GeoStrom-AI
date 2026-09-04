"""Phase 4: satellite data preparation pipeline.

Converts verified HURSAT-B1 + ADT-HURSAT + IBTrACS sources (Phase 1) into a
clean, ML-ready satellite sample index + canonical Zarr imagery store. Does
not train, classify, or detect anything -- this package only prepares data.
"""
