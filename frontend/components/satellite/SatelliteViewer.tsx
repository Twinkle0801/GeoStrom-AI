import EmptyState from "@/components/ui/EmptyState";
import { formatTimestamp } from "@/lib/format";

/**
 * Phase 4 built a real HURSAT-B1 -> Zarr satellite pipeline
 * (docs/PHASE_4_SATELLITE_PIPELINE.md), but no backend endpoint serves
 * those frames to a browser, and no per-storm/per-timestamp image lookup
 * exists in the current API contract (`backend/app/api/v1/` has no
 * satellite route -- confirmed by inspection before writing this
 * component, per Phase 10's explicit instruction). Serving raw Zarr frames
 * as web images would require a new image-conversion endpoint and is a
 * genuinely new backend capability, not "the smallest additive endpoint" --
 * out of this frontend phase's scope.
 *
 * BLOCKED BY EXISTING DATA/API CONTRACT: this panel is a correct, honest
 * empty-state shell, never a fabricated placeholder image, documented in
 * docs/PHASE_10_FRONTEND_DASHBOARD.md.
 */
export default function SatelliteViewer({ timestamp }: { timestamp: string | null }) {
  return (
    <div>
      <div className="relative mb-3 flex aspect-square w-full items-center justify-center overflow-hidden rounded-lg border border-dashed border-border-subtle bg-white/[0.02]">
        <div aria-hidden className="absolute inset-0 bg-grid-fine bg-grid opacity-[0.12]" />
        <div className="relative">
          <EmptyState
            title="Satellite frame unavailable for this timestamp."
            hint="No satellite-serving endpoint is exposed by the current backend API."
          />
        </div>
      </div>
      {timestamp && (
        <p className="text-center font-mono text-[11px] text-text-muted">
          Requested timestamp: {formatTimestamp(timestamp)}
        </p>
      )}
    </div>
  );
}
