import GlassPanel from "@/components/ui/GlassPanel";

/**
 * Extracted from the Phase 3 `/predict/[sid]` page's inline `MapLegend`
 * (behaviour unchanged) into its own reusable component, per the suggested
 * `components/maps/` structure. Observed vs. predicted differ by colour
 * AND line style -- mandatory for grayscale/colour-blind accessibility
 * (task §7/§15), never colour alone.
 */
export default function TrackLegend() {
  return (
    <GlassPanel className="p-4 text-xs">
      <h3 className="mb-3 text-[11px] font-semibold uppercase tracking-wide text-text-secondary">
        Legend
      </h3>
      <div className="space-y-2.5">
        <div className="flex items-center gap-2.5">
          <span aria-hidden className="inline-block h-0.5 w-6 rounded-full bg-truth shadow-[0_0_6px_1px_rgba(34,211,167,0.35)]" />
          <span className="text-text-secondary">Observed track <span className="text-text-muted">(solid)</span></span>
        </div>
        <div className="flex items-center gap-2.5">
          <span aria-hidden className="inline-block h-0.5 w-6 border-t-2 border-dashed border-predicted" />
          <span className="text-text-secondary">Predicted track <span className="text-text-muted">(dashed)</span></span>
        </div>
        <div className="flex items-center gap-2.5">
          <span aria-hidden className="inline-block h-2.5 w-2.5 rounded-full border-2 border-accent-soft bg-accent-soft/30" />
          <span className="text-text-secondary">Current scrub position</span>
        </div>
      </div>
      <p className="mt-3 border-t border-border-subtle pt-2.5 text-text-muted">
        Each predicted line is one historical baseline model. Colour and line style both
        distinguish prediction from observation.
      </p>
    </GlassPanel>
  );
}
