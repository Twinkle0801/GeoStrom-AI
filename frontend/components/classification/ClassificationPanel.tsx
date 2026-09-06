import Badge from "@/components/ui/Badge";
import EmptyState from "@/components/ui/EmptyState";

const TAXONOMY = ["CDO", "CurvedBand", "Eye", "Shear"] as const;

/**
 * No `classifications` table exists in the current database schema
 * (backend/app/db/models.py's own docstring: no detection/classification
 * table was ever built), and the Phase 9 evidence packet's own
 * `classification` field is always `None` today for the same reason
 * (docs/PHASE_9_GEMINI_INTEGRATION.md §4/§17). Adding one is a database
 * migration, outside this frontend phase's scope and not "absolutely
 * required" for Phase 10.
 *
 * BLOCKED BY EXISTING DATA/API CONTRACT: shown honestly as an empty state
 * plus the real, frozen `scene_taxonomy_v1` taxonomy for context, never a
 * fabricated label or confidence value.
 */
export default function ClassificationPanel() {
  return (
    <div>
      <EmptyState
        title="No classification result available."
        hint="No per-storm classification record is exposed by the current backend API."
      />
      <div className="mt-4">
        <div className="text-[11px] font-semibold uppercase tracking-wide text-text-muted">
          scene_taxonomy_v1 (Phase 5)
        </div>
        <div className="mt-2 flex flex-wrap gap-1.5">
          {TAXONOMY.map((label) => (
            <Badge key={label} tone="neutral">{label}</Badge>
          ))}
        </div>
        <p className="mt-2 text-xs text-text-muted">
          CDO merges IrrCDO; Eye merges LargeEye. Land and EmbCenter are excluded (not genuine
          storm-pattern classes at the current dataset scale). Production baseline: Logistic
          Regression. CNN/ResNet-18 are exploratory research results that did not beat it — see{" "}
          <a href="/models" className="underline decoration-text-muted/40 underline-offset-2 hover:text-text-secondary hover:decoration-text-secondary">
            Model Performance
          </a>.
        </p>
      </div>
    </div>
  );
}
