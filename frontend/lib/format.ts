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

export function modelDisplayName(name: string): string {
  return name
    .replace(/^intensity_/, "")
    .replace(/^track_/, "")
    .replace(/^./, (c) => c.toUpperCase());
}
