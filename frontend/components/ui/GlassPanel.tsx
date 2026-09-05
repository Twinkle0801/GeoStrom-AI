import type { HTMLAttributes } from "react";
import { clsx } from "clsx";

/**
 * The one reusable "panel" surface treatment used everywhere: translucent
 * glass, thin border, restrained. Deliberately not a shadcn/ui Card import
 * (task: "avoid unnecessary dependencies" -- this project's existing design
 * tokens in tailwind.config.ts already cover everything a Card component
 * would need; adding Radix/CVA for one visual treatment would be the kind
 * of dependency this phase explicitly warns against).
 */
export default function GlassPanel({
  className, children, ...rest
}: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={clsx(
        "rounded-xl border border-border-subtle bg-white/[0.03] backdrop-blur-sm",
        "shadow-[0_1px_0_0_rgba(255,255,255,0.04)_inset]",
        className,
      )}
      {...rest}
    >
      {children}
    </div>
  );
}
