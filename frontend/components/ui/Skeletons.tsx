import { clsx } from "clsx";

/**
 * Purpose-built loading skeletons that match the real layout dimensions of
 * what they stand in for (map/chart/card/metric), so a load-in never shifts
 * layout (CLS) and always signals what kind of content is coming -- a
 * plain generic pulse bar (still available as `LoadingSkeleton`, kept for
 * existing simple call sites) doesn't communicate "a map is loading" the
 * way a shaped skeleton does.
 */
function Shimmer({ className, style }: { className?: string; style?: React.CSSProperties }) {
  return (
    <div
      style={style}
      className={clsx(
        "animate-pulse rounded-md bg-white/[0.06]",
        className,
      )}
    />
  );
}

export function MapSkeleton({ className }: { className?: string }) {
  return (
    <div
      role="status"
      aria-label="Loading map"
      className={clsx(
        "relative overflow-hidden rounded-xl border border-border-subtle bg-bg-elevated",
        className,
      )}
    >
      <div
        aria-hidden
        className="absolute inset-0 bg-grid-fine bg-grid opacity-30"
      />
      <div aria-hidden className="absolute left-1/2 top-1/2 h-24 w-24 -translate-x-1/2 -translate-y-1/2 rounded-full border border-accent/20" />
      <div aria-hidden className="absolute left-1/2 top-1/2 h-40 w-40 -translate-x-1/2 -translate-y-1/2 rounded-full border border-accent/10" />
      <span className="sr-only">Loading map…</span>
    </div>
  );
}

export function ChartSkeleton({ className, height = "h-72" }: { className?: string; height?: string }) {
  return (
    <div role="status" aria-label="Loading chart" className={clsx(height, "w-full", className)}>
      <div className="flex h-full items-end gap-2 px-2 pb-4">
        {[38, 62, 45, 80, 52, 70, 40, 58].map((h, i) => (
          <Shimmer key={i} className="flex-1" style={{ height: `${h}%` }} />
        ))}
      </div>
      <span className="sr-only">Loading chart…</span>
    </div>
  );
}

export function CardSkeleton({ className }: { className?: string }) {
  return (
    <div
      role="status"
      aria-label="Loading"
      className={clsx("rounded-xl border border-border-subtle bg-white/[0.03] p-4", className)}
    >
      <div className="flex items-start justify-between gap-2">
        <Shimmer className="h-4 w-2/5" />
        <Shimmer className="h-4 w-10" />
      </div>
      <Shimmer className="mt-3 h-3 w-3/5" />
      <Shimmer className="mt-4 h-3 w-4/5" />
      <span className="sr-only">Loading…</span>
    </div>
  );
}

export function MetricSkeleton({ className }: { className?: string }) {
  return (
    <div
      role="status"
      aria-label="Loading metric"
      className={clsx("rounded-lg border border-border-subtle bg-white/[0.03] px-4 py-3", className)}
    >
      <Shimmer className="h-3 w-16" />
      <Shimmer className="mt-2 h-6 w-20" />
      <span className="sr-only">Loading…</span>
    </div>
  );
}

export function ModelCardSkeleton({ className }: { className?: string }) {
  return (
    <div
      role="status"
      aria-label="Loading model"
      className={clsx("rounded-xl border border-border-subtle bg-white/[0.03] p-4", className)}
    >
      <Shimmer className="h-3 w-1/3" />
      <Shimmer className="mt-3 h-5 w-2/3" />
      <div className="mt-4 space-y-2">
        <Shimmer className="h-2.5 w-full" />
        <Shimmer className="h-2.5 w-5/6" />
      </div>
      <span className="sr-only">Loading…</span>
    </div>
  );
}
