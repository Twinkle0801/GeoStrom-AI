import ModelComparisonChart from "@/components/charts/ModelComparisonChart";
import type { ModelPerformanceResponse } from "@/lib/api";

/** Great-circle track error (km) per horizon per model -- a named,
 * task-specific wrapper around the shared `ModelComparisonChart`. */
export default function TrackErrorChart({ comparison }: { comparison: ModelPerformanceResponse["track"] }) {
  return <ModelComparisonChart comparison={comparison} metricKey="mean_track_error_km" unit="km" />;
}
