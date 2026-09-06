/**
 * A pure-CSS entrance animation (no framer-motion instance, no
 * IntersectionObserver) for repeated card grids below the fold.
 *
 * Found via the Phase 13 performance audit: the home page's first version
 * mounted a `framer-motion` `whileInView` instance PER CARD across ~7
 * sections (~20+ elements) -- a real, Lighthouse-measured regression
 * (Total Blocking Time roughly doubled, home page performance score
 * 0.80 -> 0.68). Framer-motion's per-instance viewport tracking and style
 * computation was the dominant cost, not any single expensive component.
 *
 * This component plays a single CSS keyframe (`fade-in-up`, already in
 * tailwind.config.ts) once on mount -- GPU-accelerated (transform+opacity
 * only), reduced-motion aware via the `motion-safe:` variant, and costs
 * zero JS at runtime. The Hero (above the fold, genuinely benefits from
 * true scroll/orchestrated entrance) keeps framer-motion; everything
 * further down the page does not need it to look identical to a viewer.
 */
export default function FadeIn({
  children, className, delayMs = 0,
}: {
  children: React.ReactNode;
  className?: string;
  delayMs?: number;
}) {
  return (
    <div
      className={`motion-safe:animate-fade-in-up ${className ?? ""}`}
      style={delayMs ? { animationDelay: `${delayMs}ms` } : undefined}
    >
      {children}
    </div>
  );
}
