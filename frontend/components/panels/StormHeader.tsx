import type { CycloneDetail } from "@/lib/api";
import { categoryColorClass, categoryLabel, formatTimestamp } from "@/lib/format";

export default function StormHeader({ storm }: { storm: CycloneDetail }) {
  return (
    <div className="border-b border-border-subtle pb-4">
      <div className="flex items-baseline gap-3">
        <h1 className="text-3xl font-semibold tracking-tight text-text-primary">
          {storm.name ?? storm.sid}
        </h1>
        <span className={`text-lg font-medium ${categoryColorClass(storm.max_category)}`}>
          {categoryLabel(storm.max_category)}
        </span>
      </div>
      <div className="mt-1 text-sm text-text-secondary">
        {storm.season} · {storm.basin} basin · {storm.sid}
      </div>
      <div className="mt-2 flex flex-wrap gap-x-6 gap-y-1 text-xs text-text-muted tabular-nums">
        <span>Start: {formatTimestamp(storm.start_time)}</span>
        <span>End: {formatTimestamp(storm.end_time)}</span>
        <span>{storm.n_observations} observations</span>
        {storm.max_wind_kt != null && <span>Max wind: {storm.max_wind_kt.toFixed(0)} kt</span>}
        {storm.min_pressure_hpa != null && (
          <span>Min pressure: {storm.min_pressure_hpa.toFixed(0)} hPa</span>
        )}
        <span className="rounded bg-white/5 px-2 py-0.5">{storm.split ?? "unassigned"} split</span>
      </div>
      <div className="mt-3 rounded-md border border-amber-500/20 bg-amber-500/5 px-3 py-2 text-xs text-amber-200/80">
        Retrospective research prototype. All predictions shown are historical baseline model
        output, evaluated against known outcomes. Not an operational forecast.
      </div>
    </div>
  );
}
