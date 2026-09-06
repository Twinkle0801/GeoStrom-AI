import ClassificationComparisonChart from "@/components/charts/ClassificationComparisonChart";
import ModelComparisonChart from "@/components/charts/ModelComparisonChart";
import TrackErrorChart from "@/components/charts/TrackErrorChart";
import ModelCard from "@/components/models/ModelCard";
import Badge from "@/components/ui/Badge";
import GlassPanel from "@/components/ui/GlassPanel";
import MetricCard from "@/components/ui/MetricCard";
import SectionHeader from "@/components/ui/SectionHeader";
import { getModelPerformance, type ModelPerformanceResponse } from "@/lib/api";

export const metadata = {
  title: "Model Performance — GeoStrom AI",
  description: "Intensity, track, and classification model comparison on the frozen test split.",
};

type ModelEntry = ModelPerformanceResponse["intensity"]["models"][number];

export default async function ModelPerformancePage() {
  const data = await getModelPerformance();

  return (
    <main className="relative mx-auto max-w-7xl px-6 py-10">
      <div
        aria-hidden
        className="pointer-events-none absolute inset-x-0 top-0 -z-10 h-72 bg-[radial-gradient(60%_60%_at_50%_0%,rgba(76,141,255,0.08),transparent_70%)]"
      />
      <SectionHeader
        eyebrow="Model Performance"
        title="Benchmark comparison"
        description="Metrics are calculated on storm-disjoint held-out test data, per the project's frozen split. Retrospective evaluation only -- not a claim of operational forecasting skill."
      />

      <TaskSection
        title="Intensity"
        recommended={data.intensity.recommended_model}
        note={data.intensity.methodology_note}
      >
        <ModelCardGrid models={data.intensity.models} />
        <ModelMetricGrid models={data.intensity.models} horizon="24" metricKey="mae_kt" unit="kt" label="MAE @ 24h" />
        <GlassPanel className="mt-4 p-4">
          <ModelComparisonChart comparison={data.intensity} metricKey="mae_kt" unit="kt" />
        </GlassPanel>
      </TaskSection>

      <TaskSection
        title="Track"
        recommended={data.track.recommended_model}
        note={data.track.methodology_note}
      >
        <ModelCardGrid models={data.track.models} />
        <ModelMetricGrid models={data.track.models} horizon="24" metricKey="mean_track_error_km" unit="km" label="Mean error @ 24h" />
        <GlassPanel className="mt-4 p-4">
          <TrackErrorChart comparison={data.track} />
        </GlassPanel>
      </TaskSection>

      <TaskSection
        title="Classification"
        recommended={data.classification.recommended_model}
        note={data.classification.methodology_note}
      >
        <ModelCardGrid models={data.classification.models} />
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3 lg:grid-cols-5">
          {data.classification.models.map((m) => (
            <MetricCard
              key={m.model_name}
              label={m.display_name}
              value={m.metrics?.macro_f1 != null ? m.metrics.macro_f1.toFixed(3) : "—"}
              unit="macro-F1"
              tone={m.is_recommended ? "truth" : "default"}
            />
          ))}
        </div>
        <GlassPanel className="mt-4 p-4">
          <ClassificationComparisonChart models={data.classification.models} />
        </GlassPanel>
      </TaskSection>

      <p className="mt-10 text-xs italic text-text-muted">{data.disclaimer}</p>
    </main>
  );
}

function TaskSection({
  title, recommended, note, children,
}: {
  title: string;
  recommended: string;
  note: string;
  children: React.ReactNode;
}) {
  return (
    <section className="mt-12 first:mt-8">
      <div className="flex flex-wrap items-center gap-3 border-b border-border-subtle pb-3">
        <h2 className="text-xl font-semibold tracking-tight text-text-primary">{title}</h2>
        <Badge tone="recommended">Best baseline: {recommended}</Badge>
      </div>
      <p className="mt-3 max-w-3xl text-sm leading-relaxed text-text-secondary">{note}</p>
      <div className="mt-5 space-y-5">{children}</div>
    </section>
  );
}

function ModelCardGrid({ models }: { models: ModelEntry[] }) {
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
      {models.map((m) => (
        <ModelCard
          key={m.model_name}
          modelName={m.model_name}
          displayName={m.display_name}
          version={m.model_version}
          tier={m.tier}
          isRecommended={m.is_recommended}
        />
      ))}
    </div>
  );
}

function ModelMetricGrid({
  models, horizon, metricKey, unit, label,
}: {
  models: ModelEntry[];
  horizon: string;
  metricKey: string;
  unit: string;
  label: string;
}) {
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
      {models.map((m) => {
        const v = m.metrics_by_horizon?.[horizon]?.[metricKey];
        return (
          <div key={m.model_name} className="rounded-lg border border-border-subtle bg-white/[0.03] p-3">
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-medium text-text-secondary">{m.display_name}</span>
              <Badge tone={m.is_recommended ? "recommended" : m.tier === "exploratory" ? "exploratory" : "neutral"}>
                {m.is_recommended ? "Best" : m.tier}
              </Badge>
            </div>
            <div className="mt-1 font-mono tabular-nums text-lg font-semibold text-text-primary">
              {v != null ? v.toFixed(2) : "—"}
              <span className="ml-1 font-sans text-xs font-normal text-text-muted">{unit}</span>
            </div>
            <div className="text-[10px] text-text-muted">{label}</div>
          </div>
        );
      })}
    </div>
  );
}
