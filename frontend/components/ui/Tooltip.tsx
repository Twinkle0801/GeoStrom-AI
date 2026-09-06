"use client";

/**
 * Lightweight hover/focus tooltip -- no dependency (no Radix/Floating UI),
 * used for map controls, legend swatches, and metric labels that need one
 * short line of extra context. Content shown here must never be the ONLY
 * source of a fact (task §31: "tooltips do not become the only source of
 * information") -- callers should treat this as a supplementary hint, not
 * the primary label.
 */
import { useId, useState } from "react";
import { clsx } from "clsx";

export default function Tooltip({
  content, children, side = "top",
}: {
  content: string;
  children: React.ReactNode;
  side?: "top" | "bottom";
}) {
  const [open, setOpen] = useState(false);
  const id = useId();

  return (
    <span
      className="relative inline-flex"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
    >
      <span aria-describedby={open ? id : undefined} className="inline-flex">
        {children}
      </span>
      <span
        role="tooltip"
        id={id}
        className={clsx(
          "pointer-events-none absolute left-1/2 z-50 w-max max-w-[220px] -translate-x-1/2 rounded-md border border-border-subtle bg-bg-overlay px-2.5 py-1.5 text-[11px] leading-snug text-text-secondary shadow-elevated transition-opacity duration-150",
          side === "top" ? "bottom-full mb-2" : "top-full mt-2",
          open ? "opacity-100" : "opacity-0",
        )}
      >
        {content}
      </span>
    </span>
  );
}
