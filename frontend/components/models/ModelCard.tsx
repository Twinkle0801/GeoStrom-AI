import Badge, { type BadgeTone } from "@/components/ui/Badge";
import GlassPanel from "@/components/ui/GlassPanel";
import { getModelInfo } from "@/lib/modelInfo";

const TIER_TONE: Record<string, BadgeTone> = {
  floor: "neutral",
  baseline: "recommended",
  exploratory: "exploratory",
};

const TIER_LABEL: Record<string, string> = {
  floor: "Reference floor",
  baseline: "Production baseline",
  exploratory: "Exploratory",
};

export default function ModelCard({
  modelName, displayName, version, tier, isRecommended,
}: {
  modelName: string;
  displayName: string;
  version: string;
  tier: "floor" | "baseline" | "exploratory";
  isRecommended: boolean;
}) {
  const info = getModelInfo(modelName);
  return (
    <GlassPanel hover className="flex h-full flex-col gap-3 p-4">
      <div className="flex items-start justify-between gap-2">
        <div>
          <h4 className="text-sm font-semibold text-text-primary">{displayName}</h4>
          <p className="font-mono text-[11px] text-text-muted">{modelName} · {version}</p>
        </div>
        {isRecommended ? (
          <Badge tone="recommended">Recommended</Badge>
        ) : (
          <Badge tone={TIER_TONE[tier]}>{TIER_LABEL[tier]}</Badge>
        )}
      </div>
      <p className="text-xs leading-relaxed text-text-secondary">{info.purpose}</p>
      <dl className="mt-auto space-y-1.5 border-t border-border-subtle pt-3 text-xs">
        <div className="flex gap-2">
          <dt className="w-14 shrink-0 font-medium uppercase tracking-wide text-text-muted">Input</dt>
          <dd className="text-text-secondary">{info.input}</dd>
        </div>
        <div className="flex gap-2">
          <dt className="w-14 shrink-0 font-medium uppercase tracking-wide text-text-muted">Output</dt>
          <dd className="text-text-secondary">{info.output}</dd>
        </div>
      </dl>
    </GlassPanel>
  );
}
