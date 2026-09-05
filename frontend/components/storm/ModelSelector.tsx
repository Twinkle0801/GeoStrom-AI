"use client";

import Badge from "@/components/ui/Badge";
import { modelDisplayName } from "@/lib/format";
import { isRecommendedModel, modelTier } from "@/lib/modelTiers";

export interface ModelOption {
  name: string;
  version: string;
}

/**
 * Only ever offers models actually present in this storm's real prediction
 * rows (`options` is derived by the caller from the fetched prediction
 * series, never a hardcoded list) -- task: "Only display models that are
 * actually available for that storm." GRU never appears here because no
 * per-storm GRU prediction exists in the database (Phase 7/8's GRU work
 * produced aggregate benchmark metrics only, shown on `/models` instead).
 */
export default function ModelSelector({
  label, options, value, onChange,
}: {
  label: string;
  options: ModelOption[];
  value: string | null;
  onChange: (name: string) => void;
}) {
  if (options.length === 0) return null;

  return (
    <div>
      <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-text-muted">
        {label}
      </div>
      <div className="flex flex-wrap gap-2" role="radiogroup" aria-label={label}>
        {options.map((opt) => {
          const tier = modelTier(opt.name);
          const recommended = isRecommendedModel(opt.name);
          const active = value === opt.name;
          return (
            <button
              key={opt.name}
              type="button"
              role="radio"
              aria-checked={active}
              onClick={() => onChange(opt.name)}
              className={`flex items-center gap-2 rounded-lg border px-3 py-1.5 text-sm font-medium transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent ${
                active
                  ? "border-accent-soft/50 bg-accent/10 text-text-primary"
                  : "border-border-subtle bg-white/[0.02] text-text-secondary hover:bg-white/5"
              }`}
            >
              {modelDisplayName(opt.name)}
              {recommended ? (
                <Badge tone="recommended">Best baseline</Badge>
              ) : tier === "exploratory" ? (
                <Badge tone="exploratory">Exploratory</Badge>
              ) : null}
            </button>
          );
        })}
      </div>
    </div>
  );
}
