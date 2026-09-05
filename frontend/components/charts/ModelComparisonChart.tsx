"use client";

/**
 * Generic per-horizon model comparison bar chart, shared by the Model
 * Performance page's intensity and track sections (`TrackErrorChart` is a
 * thin, task-specific wrapper around this same component -- one chart
 * implementation, not two divergent copies). Every value plotted here is
 * read verbatim from `GET /api/v1/analytics/model-performance`, itself a
 * verbatim read of the committed Phase 2/7/8 benchmark JSON -- nothing is
 * computed in this component.
 */
import {
  Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { ModelPerformanceResponse } from "@/lib/api";

type TaskComparison = ModelPerformanceResponse["intensity"];

const SERIES_COLORS = ["#4C8DFF", "#22D3A7", "#FFB020", "#C77DFF", "#F72585"];

export default function ModelComparisonChart({
  comparison, metricKey, unit,
}: {
  comparison: TaskComparison;
  metricKey: string;
  unit: string;
}) {
  if (!comparison.horizons_h || comparison.models.length === 0) return null;

  const data = comparison.horizons_h.map((h) => {
    const row: Record<string, number | string> = { horizon: `+${h}h` };
    for (const model of comparison.models) {
      const v = model.metrics_by_horizon?.[String(h)]?.[metricKey];
      if (v != null) row[model.display_name] = Number(v.toFixed(2));
    }
    return row;
  });

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
          <XAxis dataKey="horizon" stroke="rgba(255,255,255,0.25)" tick={{ fill: "#9BA6B8", fontSize: 11 }} />
          <YAxis unit={` ${unit}`} stroke="rgba(255,255,255,0.25)" tick={{ fill: "#9BA6B8", fontSize: 11 }} width={56} />
          <Tooltip
            contentStyle={{ background: "#0B0F17", border: "1px solid rgba(255,255,255,0.09)", fontSize: 12 }}
            labelStyle={{ color: "#F2F5FA" }}
          />
          <Legend wrapperStyle={{ fontSize: 11, color: "#9BA6B8" }} />
          {comparison.models.map((model, i) => (
            <Bar
              key={model.model_name}
              dataKey={model.display_name}
              fill={SERIES_COLORS[i % SERIES_COLORS.length]}
              isAnimationActive={false}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
