/**
 * Extracted from the Phase 3 `/predict/[sid]` page's inline `MapLegend`
 * (behaviour unchanged) into its own reusable component, per the suggested
 * `components/maps/` structure. Observed vs. predicted differ by colour
 * AND line style -- mandatory for grayscale/colour-blind accessibility
 * (task §7/§15), never colour alone.
 */
export default function TrackLegend() {
  return (
    <div className="rounded-lg border border-border-subtle bg-white/5 p-4 text-xs">
      <h3 className="mb-2 font-semibold uppercase tracking-wide text-text-secondary">Legend</h3>
      <div className="flex items-center gap-2">
        <span className="inline-block h-0.5 w-6 bg-truth" />
        <span className="text-text-secondary">Observed track (solid)</span>
      </div>
      <div className="mt-1 flex items-center gap-2">
        <span className="inline-block h-0.5 w-6 border-t-2 border-dashed border-predicted" />
        <span className="text-text-secondary">Predicted track (dashed)</span>
      </div>
      <div className="mt-1 flex items-center gap-2">
        <span className="inline-block h-2.5 w-2.5 rounded-full border border-accent-soft bg-accent-soft/30" />
        <span className="text-text-secondary">Current scrub position</span>
      </div>
      <p className="mt-2 text-text-muted">
        Each predicted line is one historical baseline model. Colour and line style both
        distinguish prediction from observation.
      </p>
    </div>
  );
}
