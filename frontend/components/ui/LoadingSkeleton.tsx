import { clsx } from "clsx";

export default function LoadingSkeleton({
  className, lines = 1, label = "Loading",
}: {
  className?: string;
  lines?: number;
  label?: string;
}) {
  return (
    <div role="status" aria-label={label} className={clsx("space-y-2", className)}>
      {[...Array(lines)].map((_, i) => (
        <div key={i} className="h-4 animate-pulse rounded-md bg-white/[0.06]" />
      ))}
      <span className="sr-only">{label}</span>
    </div>
  );
}
