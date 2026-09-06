/**
 * Compact preview of the real `/models` benchmark data -- every metric here
 * is read verbatim from `GET /api/v1/analytics/model-performance`, the
 * exact same response the full Model Performance page renders (never
 * recomputed, never a second source of truth). Zero-JS server component --
 * see FadeIn.tsx.
 */
import Link from "next/link";
import GlassPanel from "@/components/ui/GlassPanel";
import SectionHeader from "@/components/ui/SectionHeader";
import Badge from "@/components/ui/Badge";
import FadeIn from "@/components/ui/FadeIn";
import { ArrowRightIcon } from "@/components/ui/Icons";
import type { ModelPerformanceResponse } from "@/lib/api";

function headlineMetric(
  task: ModelPerformanceResponse["intensity"] | ModelPerformanceResponse["track"],
  metricKey: string, unit: string,
): string | null {
  const best = task.models.find((m) => m.is_recommended);
  const v = best?.metrics_by_horizon?.["24"]?.[metricKey];
  return v != null ? `${Number(v).toFixed(2)} ${unit} @ 24h` : null;
}

export default function ModelIntelligence({ data }: { data: ModelPerformanceResponse | null }) {
  if (!data) return null;

  const cards = [
    {
      task: "Intensity", recommended: data.intensity.recommended_model,
      metric: headlineMetric(data.intensity, "mae_kt", "kt MAE"),
      count: data.intensity.models.length,
    },
    {
      task: "Track", recommended: data.track.recommended_model,
      metric: headlineMetric(data.track, "mean_track_error_km", "km error"),
      count: data.track.models.length,
    },
    {
      task: "Classification", recommended: data.classification.recommended_model,
      metric: (() => {
        const best = data.classification.models.find((m) => m.is_recommended);
        return best?.metrics?.macro_f1 != null ? `${best.metrics.macro_f1.toFixed(3)} macro-F1` : null;
      })(),
      count: data.classification.models.length,
    },
  ];

  return (
    <section className="mx-auto max-w-7xl px-6 py-16">
      <SectionHeader
        eyebrow="Model intelligence"
        title="Every model, benchmarked on the same frozen test split"
        description="Recommended baselines never change silently -- exploratory models (GRU, CNN/ResNet-18) are always labelled, never presented as the winner."
      />
      <div className="mt-8 grid grid-cols-1 gap-4 sm:grid-cols-3">
        {cards.map((c, i) => (
          <FadeIn key={c.task} delayMs={i * 60}>
            <GlassPanel hover className="flex h-full flex-col gap-3 p-5">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-text-primary">{c.task}</h3>
                <span className="font-mono text-[11px] text-text-muted">{c.count} models</span>
              </div>
              <Badge tone="recommended" className="w-fit">{c.recommended}</Badge>
              <div className="font-mono text-lg font-semibold tabular-nums text-text-primary">
                {c.metric ?? "—"}
              </div>
            </GlassPanel>
          </FadeIn>
        ))}
      </div>
      <FadeIn className="mt-6">
        <Link
          href="/models"
          className="inline-flex items-center gap-1.5 text-sm font-medium text-accent-soft hover:text-accent"
        >
          View full benchmark comparison <ArrowRightIcon width={14} height={14} />
        </Link>
      </FadeIn>
    </section>
  );
}
