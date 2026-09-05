import Badge from "@/components/ui/Badge";
import type { CycloneDetail } from "@/lib/api";
import { categoryColorClass, categoryLabel, formatTimestamp } from "@/lib/format";

/**
 * Combines the flagship page's "HEADER" and "HERO / STORM SUMMARY" sections
 * (task §7) into one component -- both showed the same underlying storm
 * fields, so a second near-duplicate component would only add
 * indirection. Status is always "Historical Analysis"; this literal string
 * is the only status this component can ever render -- never "Live",
 * "Active", or "Warning" (task §19's scientific-honesty rule, enforced by
 * there being no other code path here that could produce those words).
 */
export default function StormHeader({
  storm, maxForecastHorizonH,
}: {
  storm: CycloneDetail;
  maxForecastHorizonH?: number | null;
}) {
  return (
    <div className="border-b border-border-subtle pb-6">
      <div className="flex flex-wrap items-center gap-3">
        <Badge tone="neutral">Historical Analysis</Badge>
        <span className="text-xs text-text-muted">{storm.sid}</span>
      </div>
      <div className="mt-3 flex flex-wrap items-baseline gap-3">
        <h1 className="text-3xl font-semibold tracking-tight text-text-primary sm:text-4xl">
          {storm.name ?? storm.sid}
        </h1>
        <span className={`text-lg font-medium ${categoryColorClass(storm.max_category)}`}>
          {categoryLabel(storm.max_category)}
        </span>
      </div>
      <div className="mt-1 text-sm text-text-secondary">
        North Atlantic · Season {storm.season}
      </div>

      <div className="mt-5 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Stat label="Observed duration">
          {formatTimestamp(storm.start_time)} → {formatTimestamp(storm.end_time)}
        </Stat>
        <Stat label="Maximum observed wind">
          {storm.max_wind_kt != null ? `${storm.max_wind_kt.toFixed(0)} kt` : "—"}
        </Stat>
        <Stat label="Minimum observed pressure">
          {storm.min_pressure_hpa != null ? `${storm.min_pressure_hpa.toFixed(0)} hPa` : "—"}
        </Stat>
        <Stat label="Available forecast horizon">
          {maxForecastHorizonH ? `Up to +${maxForecastHorizonH}h` : "Not available"}
        </Stat>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-1 text-xs text-text-muted tabular-nums">
        <span>{storm.n_observations} observations</span>
        <span className="rounded bg-white/5 px-2 py-0.5">{storm.split ?? "unassigned"} split</span>
      </div>

      <div className="mt-4 rounded-md border border-amber-500/20 bg-amber-500/5 px-3 py-2 text-xs text-amber-200/80">
        Retrospective research prototype. All predictions shown are historical baseline model
        output, evaluated against known outcomes. Not an operational forecast.
      </div>
    </div>
  );
}

function Stat({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="text-[11px] font-medium uppercase tracking-wide text-text-muted">{label}</div>
      <div className="mt-0.5 text-sm font-medium tabular-nums text-text-primary">{children}</div>
    </div>
  );
}
