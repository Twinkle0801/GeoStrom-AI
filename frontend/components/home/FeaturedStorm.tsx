/**
 * Showcases ONE real storm from the live catalogue -- selected server-side
 * (app/page.tsx) as the highest recorded `max_wind_kt` in the fetched page,
 * a real, deterministic, non-arbitrary criterion, never a hand-picked or
 * invented "hero" storm. Zero-JS server component -- see FadeIn.tsx.
 */
import Link from "next/link";
import GlassPanel from "@/components/ui/GlassPanel";
import SectionHeader from "@/components/ui/SectionHeader";
import Badge from "@/components/ui/Badge";
import FadeIn from "@/components/ui/FadeIn";
import { ArrowRightIcon, ClockIcon, GaugeIcon, WindIcon } from "@/components/ui/Icons";
import { categoryColorClass, categoryLabel, formatTimestamp } from "@/lib/format";
import type { CycloneDetail } from "@/lib/api";

export default function FeaturedStorm({ storm }: { storm: CycloneDetail | null }) {
  if (!storm) return null;

  return (
    <section className="mx-auto max-w-7xl px-6 py-16">
      <SectionHeader
        eyebrow="Featured storm"
        title="The most intense storm in the catalogue"
        description="Selected by recorded peak wind speed -- a real storm, not a curated example."
      />
      <FadeIn className="mt-8">
        <GlassPanel hover className="grid grid-cols-1 gap-6 p-6 sm:p-8 lg:grid-cols-[1.4fr_1fr] lg:items-center">
          <div>
            <div className="flex flex-wrap items-center gap-3">
              <Badge tone="neutral">Historical Analysis</Badge>
              {storm.has_predictions && <Badge tone="accent">Forecast available</Badge>}
            </div>
            <div className="mt-3 flex flex-wrap items-baseline gap-3">
              <h3 className="text-2xl font-semibold tracking-tight text-text-primary sm:text-3xl">
                {storm.name ?? storm.sid}
              </h3>
              <span className={`text-base font-medium ${categoryColorClass(storm.max_category)}`}>
                {categoryLabel(storm.max_category)}
              </span>
            </div>
            <p className="mt-1 font-mono text-xs text-text-muted">{storm.sid} · Season {storm.season}</p>
            <Link
              href={`/predict/${storm.sid}`}
              className="mt-5 inline-flex items-center gap-1.5 rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-white shadow-glow transition-transform hover:scale-[1.02] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-soft"
            >
              Open full analysis <ArrowRightIcon width={14} height={14} />
            </Link>
          </div>
          <div className="grid grid-cols-3 gap-3">
            <MiniStat icon={WindIcon} label="Max wind" value={storm.max_wind_kt != null ? `${storm.max_wind_kt.toFixed(0)}` : "—"} unit="kt" />
            <MiniStat icon={GaugeIcon} label="Min pressure" value={storm.min_pressure_hpa != null ? `${storm.min_pressure_hpa.toFixed(0)}` : "—"} unit="hPa" />
            <MiniStat icon={ClockIcon} label="Observations" value={String(storm.n_observations)} />
          </div>
        </GlassPanel>
      </FadeIn>
      <p className="mt-3 text-center text-xs text-text-muted sm:text-left">
        {formatTimestamp(storm.start_time)} → {formatTimestamp(storm.end_time)} · every value observed, not predicted
      </p>
    </section>
  );
}

function MiniStat({
  icon: Icon, label, value, unit,
}: {
  icon: (props: React.SVGProps<SVGSVGElement>) => React.ReactElement;
  label: string;
  value: string;
  unit?: string;
}) {
  return (
    <div className="rounded-lg border border-border-subtle bg-white/[0.03] p-3 text-center">
      <Icon width={16} height={16} className="mx-auto text-text-muted" />
      <div className="mt-1.5 font-mono text-lg font-semibold tabular-nums text-text-primary">
        {value}
        {unit && <span className="ml-0.5 font-sans text-xs font-normal text-text-muted">{unit}</span>}
      </div>
      <div className="text-[10px] uppercase tracking-wide text-text-muted">{label}</div>
    </div>
  );
}
