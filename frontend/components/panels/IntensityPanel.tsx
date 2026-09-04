"use client";

/**
 * Minimal intensity panel: observed vs. predicted wind per horizon per
 * model. Deliberately a plain table, not a chart -- docs/PROJECT_
 * REQUIREMENTS.md Phase 3 scope note says "a basic chart is acceptable...
 * do not build the final analytics dashboard yet." A table is simpler and
 * no less correct.
 *
 * Every row states model + model_version (UI_UX_ARCHITECTURE.md ModelBadge
 * rule) and uses "Historical baseline prediction" language, never
 * "forecast will happen" / "guaranteed" phrasing (Phase 3 task §19).
 */
import type { PredictionList } from "@/lib/api";
import { modelDisplayName } from "@/lib/format";

export default function IntensityPanel({ predictions }: { predictions: PredictionList }) {
  const intensity = predictions
    .filter((p) => p.task === "intensity")
    .sort((a, b) => a.lead_hours - b.lead_hours || a.model_name.localeCompare(b.model_name));

  if (intensity.length === 0) {
    return (
      <div className="rounded-lg border border-border-subtle bg-white/5 p-4 text-sm text-text-secondary">
        No intensity predictions available for this forecast origin.
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-border-subtle bg-white/5 p-4">
      <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-text-secondary">
        Intensity — Historical Baseline Predictions
      </h3>
      <table className="w-full text-left text-sm">
        <thead>
          <tr className="text-xs uppercase text-text-muted">
            <th className="pb-2 pr-3">Horizon</th>
            <th className="pb-2 pr-3">Model</th>
            <th className="pb-2 pr-3 text-right">Predicted (kt)</th>
            <th className="pb-2 pr-3 text-right">Observed (kt)</th>
            <th className="pb-2 text-right">Error (kt)</th>
          </tr>
        </thead>
        <tbody className="tabular-nums">
          {intensity.map((p, i) => (
            <tr key={i} className="border-t border-border-subtle/60">
              <td className="py-2 pr-3">+{p.lead_hours}h</td>
              <td className="py-2 pr-3">
                {modelDisplayName(p.model_name)}{" "}
                <span className="text-xs text-text-muted">{p.model_version}</span>
              </td>
              <td className="py-2 pr-3 text-right text-predicted">
                {p.pred_wind_kt != null ? p.pred_wind_kt.toFixed(1) : "—"}
              </td>
              <td className="py-2 pr-3 text-right text-truth">
                {p.true_wind_kt != null ? p.true_wind_kt.toFixed(1) : "—"}
              </td>
              <td className="py-2 text-right text-text-secondary">
                {p.wind_error_kt != null ? p.wind_error_kt.toFixed(1) : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-3 text-xs italic text-text-muted">
        Historical baseline model predictions, retrospectively evaluated against observed data.
        Not an operational forecast or safety guidance.
      </p>
    </div>
  );
}
