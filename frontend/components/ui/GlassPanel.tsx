import type { HTMLAttributes } from "react";
import { clsx } from "clsx";

/**
 * The one reusable "panel" surface treatment used everywhere: translucent
 * glass, thin border, restrained. Deliberately not a shadcn/ui Card import
 * (task: "avoid unnecessary dependencies" -- this project's existing design
 * tokens in tailwind.config.ts already cover everything a Card component
 * would need; adding Radix/CVA for one visual treatment would be the kind
 * of dependency this phase explicitly warns against).
 *
 * `hover`: opt-in subtle elevation + border illumination on hover, for
 * panels that are themselves an interactive target (a storm card, a model
 * card) -- omitted by default so purely-informational panels (evidence
 * rows, legends) never look clickable when they are not.
 */
export default function GlassPanel({
  className, children, hover, ...rest
}: HTMLAttributes<HTMLDivElement> & { hover?: boolean }) {
  return (
    <div
      className={clsx(
        "rounded-xl border border-border-subtle bg-white/[0.03] backdrop-blur-sm",
        "shadow-[0_1px_0_0_rgba(255,255,255,0.04)_inset]",
        hover &&
          "transition-all duration-300 ease-premium hover:-translate-y-0.5 hover:border-accent-soft/40 hover:bg-white/[0.05] hover:shadow-panel",
        className,
      )}
      {...rest}
    >
      {children}
    </div>
  );
}
