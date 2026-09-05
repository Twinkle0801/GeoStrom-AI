import { clsx } from "clsx";

export default function MetricCard({
  label, value, unit, tone = "default",
}: {
  label: string;
  value: string;
  unit?: string;
  tone?: "default" | "truth" | "predicted";
}) {
  return (
    <div className="rounded-lg border border-border-subtle bg-white/[0.03] px-4 py-3">
      <div className="text-[11px] font-medium uppercase tracking-wide text-text-muted">
        {label}
      </div>
      <div
        className={clsx(
          "mt-1 tabular-nums text-2xl font-semibold tracking-tight",
          tone === "truth" && "text-truth",
          tone === "predicted" && "text-predicted",
          tone === "default" && "text-text-primary",
        )}
      >
        {value}
        {unit && <span className="ml-1 text-sm font-normal text-text-muted">{unit}</span>}
      </div>
    </div>
  );
}
