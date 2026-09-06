/**
 * "Live System Overview" -- every number here is a real aggregate computed
 * from the actual storm catalogue (`GET /api/v1/cyclones`), passed down as
 * a prop from the server-rendered home page. Nothing here is invented or
 * hardcoded: if the backend is unreachable, `storms` is an empty array and
 * this section renders nothing rather than a fabricated placeholder number
 * (task §35/§6's "use ONLY existing API data"). Zero-JS server component --
 * see FadeIn.tsx.
 */
import MetricCard from "@/components/ui/MetricCard";
import SectionHeader from "@/components/ui/SectionHeader";
import FadeIn from "@/components/ui/FadeIn";
import { ClockIcon, DatabaseIcon, LayersIcon, WindIcon } from "@/components/ui/Icons";
import type { CyclonesList } from "@/lib/api";

export default function LiveOverview({ storms }: { storms: CyclonesList | null }) {
  const items = storms?.items ?? [];
  if (items.length === 0) return null;

  const totalStorms = storms!.total;
  const seasons = items.map((s) => s.season);
  const seasonMin = Math.min(...seasons);
  const seasonMax = Math.max(...seasons);
  const totalObservations = items.reduce((sum, s) => sum + s.n_observations, 0);
  const winds = items.map((s) => s.max_wind_kt).filter((w): w is number => w != null);
  const maxWind = winds.length > 0 ? Math.max(...winds) : null;
  const splitCounts = items.reduce<Record<string, number>>((acc, s) => {
    const key = s.split ?? "unassigned";
    acc[key] = (acc[key] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <section className="mx-auto max-w-7xl px-6 py-16">
      <SectionHeader
        eyebrow="Live system overview"
        title="What's in the catalogue right now"
        description="Computed directly from the live storm catalogue at request time -- not a static claim."
      />
      <div className="mt-8 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <FadeIn>
          <MetricCard label="Storms catalogued" value={String(totalStorms)} icon={<DatabaseIcon width={14} height={14} />} />
        </FadeIn>
        <FadeIn delayMs={60}>
          <MetricCard
            label="Seasons covered"
            value={seasonMin === seasonMax ? String(seasonMin) : `${seasonMin}–${seasonMax}`}
            icon={<ClockIcon width={14} height={14} />}
          />
        </FadeIn>
        <FadeIn delayMs={120}>
          <MetricCard
            label="Observed data points"
            value={totalObservations.toLocaleString("en-US")}
            icon={<LayersIcon width={14} height={14} />}
          />
        </FadeIn>
        <FadeIn delayMs={180}>
          <MetricCard
            label="Peak observed intensity"
            value={maxWind != null ? maxWind.toFixed(0) : "—"}
            unit={maxWind != null ? "kt" : undefined}
            icon={<WindIcon width={14} height={14} />}
          />
        </FadeIn>
      </div>
      {Object.keys(splitCounts).length > 0 && (
        <FadeIn className="mt-4 flex flex-wrap items-center gap-x-5 gap-y-1.5 text-xs text-text-muted">
          <span className="font-medium uppercase tracking-wide text-text-muted">Frozen evaluation split</span>
          {Object.entries(splitCounts).map(([split, count]) => (
            <span key={split} className="font-mono tabular-nums">
              {split}: {count}
            </span>
          ))}
        </FadeIn>
      )}
    </section>
  );
}
