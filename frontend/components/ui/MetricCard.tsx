import { clsx } from "clsx";

export default function MetricCard({
  label, value, unit, tone = "default", icon,
}: {
  label: string;
  value: string;
  unit?: string;
  tone?: "default" | "truth" | "predicted";
  icon?: React.ReactNode;
}) {
  return (
    <div className="group rounded-lg border border-border-subtle bg-white/[0.03] px-4 py-3 transition-colors duration-300 hover:border-border-strong hover:bg-white/[0.045]">
      <div className="flex items-center justify-between gap-2">
        <div className="text-[11px] font-medium uppercase tracking-wide text-text-muted">
          {label}
        </div>
        {icon && <span className="text-text-muted/70 transition-colors group-hover:text-accent-soft">{icon}</span>}
      </div>
      <div
        className={clsx(
          "mt-1 font-mono tabular-nums text-2xl font-semibold tracking-tight",
          tone === "truth" && "text-truth",
          tone === "predicted" && "text-predicted",
          tone === "default" && "text-text-primary",
        )}
      >
        {value}
        {unit && <span className="ml-1 font-sans text-sm font-normal text-text-muted">{unit}</span>}
      </div>
    </div>
  );
}
