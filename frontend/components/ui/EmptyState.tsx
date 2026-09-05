/**
 * The one honest "we don't have this" surface, used identically for
 * satellite frames, classification results, missing predictions, etc.
 * Never a fabricated placeholder -- task §22/§8/§9's core rule.
 */
export default function EmptyState({
  title, hint,
}: {
  title: string;
  hint?: string;
}) {
  return (
    <div
      role="status"
      className="flex flex-col items-center justify-center gap-1 rounded-lg border border-dashed border-border-subtle bg-white/[0.02] px-4 py-8 text-center"
    >
      <p className="text-sm text-text-secondary">{title}</p>
      {hint && <p className="text-xs text-text-muted">{hint}</p>}
    </div>
  );
}
