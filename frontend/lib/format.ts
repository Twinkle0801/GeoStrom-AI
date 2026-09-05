/** Category -> label/colour mapping. Matches the Saffir-Simpson thresholds
 * backend/scripts/ingest_phase2_predictions.py::category_from_wind uses --
 * the label set is presentation of an already-computed value, never a
 * re-derivation of it. */
export function categoryLabel(category: number | null | undefined): string {
  if (category === null || category === undefined) return "Unknown";
  const labels: Record<number, string> = {
    [-1]: "Tropical Depression", 0: "Tropical Storm", 1: "Category 1",
    2: "Category 2", 3: "Category 3", 4: "Category 4", 5: "Category 5",
  };
  return labels[category] ?? "Unknown";
}

export function categoryColorClass(category: number | null | undefined): string {
  if (category === null || category === undefined) return "text-text-muted";
  const classes: Record<number, string> = {
    [-1]: "text-intensity-td", 0: "text-intensity-ts", 1: "text-intensity-c1",
    2: "text-intensity-c2", 3: "text-intensity-c3", 4: "text-intensity-c4",
    5: "text-intensity-c5",
  };
  return classes[category] ?? "text-text-muted";
}

export function formatTimestamp(iso: string): string {
  return new Date(iso).toLocaleString("en-US", {
    year: "numeric", month: "short", day: "2-digit",
    hour: "2-digit", minute: "2-digit", timeZone: "UTC", timeZoneName: "short",
  });
}

// Mirrors backend/app/services/analytics.py's `_DISPLAY_NAMES` -- kept in
// sync there and here since both independently render a model's name to a
// human, both from the same underlying `model_name` strings.
const KNOWN_DISPLAY_NAMES: Record<string, string> = {
  intensity_persistence: "Persistence", intensity_ridge: "Ridge", intensity_lightgbm: "LightGBM",
  intensity_gru: "GRU (absolute)", intensity_gru_delta: "GRU (Δwind)",
  track_persistence: "Persistence", track_cliper: "CLIPER-style Ridge", track_lightgbm: "LightGBM",
  track_gru: "GRU",
};

/**
 * Human-readable model name. Found via a real test failure (Phase 10,
 * ModelSelector.test.tsx) that a naive strip-prefix-and-capitalise approach
 * renders brand/acronym names wrong ("Lightgbm", "Cliper" instead of
 * "LightGBM", "CLIPER-style Ridge") -- a real, user-visible bug in a
 * "premium" UI. Falls back to the original capitalisation behaviour only
 * for a name not in the known table, so this never throws for an
 * unanticipated model name.
 */
export function modelDisplayName(name: string): string {
  if (name in KNOWN_DISPLAY_NAMES) return KNOWN_DISPLAY_NAMES[name];
  return name
    .replace(/^intensity_/, "")
    .replace(/^track_/, "")
    .replace(/^./, (c) => c.toUpperCase());
}
