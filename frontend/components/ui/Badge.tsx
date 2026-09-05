import { clsx } from "clsx";

export type BadgeTone = "neutral" | "truth" | "predicted" | "recommended" | "exploratory" | "warning";

const toneClasses: Record<BadgeTone, string> = {
  neutral: "border-border-subtle bg-white/5 text-text-secondary",
  truth: "border-truth/30 bg-truth/10 text-truth",
  predicted: "border-predicted/30 bg-predicted/10 text-predicted",
  recommended: "border-emerald-400/30 bg-emerald-400/10 text-emerald-300",
  exploratory: "border-violet-400/30 bg-violet-400/10 text-violet-300",
  warning: "border-amber-500/30 bg-amber-500/10 text-amber-300",
};

export default function Badge({
  children, tone = "neutral", className,
}: {
  children: React.ReactNode;
  tone?: BadgeTone;
  className?: string;
}) {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide",
        toneClasses[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}
