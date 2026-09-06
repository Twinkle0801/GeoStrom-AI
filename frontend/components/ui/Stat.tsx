/**
 * A label/value pair used for compact metadata rows (storm header stats,
 * home-page live-system numbers). Extracted from the inline `Stat` helper
 * `StormHeader.tsx` originally defined for itself, so the home page's
 * "Live System Overview" section can reuse the exact same visual treatment
 * rather than a second, subtly-different one.
 */
export default function Stat({
  label, children, mono = true,
}: {
  label: string;
  children: React.ReactNode;
  mono?: boolean;
}) {
  return (
    <div>
      <div className="text-[11px] font-medium uppercase tracking-wide text-text-muted">{label}</div>
      <div className={`mt-0.5 text-sm font-medium text-text-primary ${mono ? "font-mono tabular-nums" : ""}`}>
        {children}
      </div>
    </div>
  );
}
