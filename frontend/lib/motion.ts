/**
 * Shared Framer Motion variants + a `prefers-reduced-motion` hook. Every
 * entrance/reveal animation in the app should import from here rather than
 * hand-rolling transition timings, so the whole product moves at one
 * consistent, restrained pace (task: "150-400ms micro-interactions").
 */
"use client";

import { useEffect, useState } from "react";
import type { Variants } from "framer-motion";

export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduced(mq.matches);
    const handler = (e: MediaQueryListEvent) => setReduced(e.matches);
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []);
  return reduced;
}

/** Fade + slight rise. The default entrance for headers, cards, panels. */
export const fadeUp: Variants = {
  hidden: { opacity: 0, y: 16 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.4, ease: "easeOut" } },
};

/** Parent wrapper for staggered children -- use with `fadeUp` on children. */
export const staggerContainer: Variants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.08, delayChildren: 0.05 } },
};

export const fadeIn: Variants = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { duration: 0.3, ease: "easeOut" } },
};

/** Micro-interaction timing for hover/tap/focus -- 150-400ms per the task's
 * animation principles. */
export const microTransition = { duration: 0.2, ease: "easeOut" } as const;

/** Variants with all motion collapsed to an instant opacity change -- pass
 * to `<motion.*>` when `usePrefersReducedMotion()` is true, so the same
 * component tree works for both cases without branching every call site. */
export const reducedMotionVariants: Variants = {
  hidden: { opacity: 1, y: 0 },
  visible: { opacity: 1, y: 0, transition: { duration: 0 } },
};
