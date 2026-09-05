/**
 * Presentational tier metadata (which badge to show), mirroring
 * `backend/app/services/analytics.py`'s `_TIERS`/`_DISPLAY_NAMES` maps.
 * This is UI labelling, not a scientific value -- keep in sync with the
 * backend if a model is ever added or reclassified.
 *
 * Only `intensity_persistence/ridge/lightgbm` and `track_persistence/
 * cliper/lightgbm` ever appear in real PER-STORM prediction data (Phase 3
 * ingestion never wrote GRU rows) -- GRU is deliberately absent here on
 * purpose, not by oversight; it is discussed only on the aggregate Model
 * Performance page (`/models`), sourced from the real analytics endpoint.
 */
export type ModelTier = "baseline" | "exploratory";

const TIERS: Record<string, ModelTier> = {
  intensity_persistence: "baseline",
  intensity_ridge: "baseline",
  intensity_lightgbm: "baseline",
  track_persistence: "baseline",
  track_cliper: "baseline",
  track_lightgbm: "baseline",
};

const RECOMMENDED = new Set(["intensity_lightgbm", "track_cliper"]);

export function modelTier(modelName: string): ModelTier {
  return TIERS[modelName] ?? "exploratory";
}

export function isRecommendedModel(modelName: string): boolean {
  return RECOMMENDED.has(modelName);
}
