/**
 * A visual restatement of the SAME pipeline docs/PHASE_*.md and
 * methodology/page.tsx already describe in prose -- no new claim, purely a
 * different presentation (a horizontal flow diagram) of already-documented
 * architecture facts. Zero-JS server component -- see FadeIn.tsx.
 */
import Link from "next/link";
import GlassPanel from "@/components/ui/GlassPanel";
import SectionHeader from "@/components/ui/SectionHeader";
import FadeIn from "@/components/ui/FadeIn";
import { ArrowRightIcon, BrainIcon, DatabaseIcon, MapPinIcon, SatelliteIcon } from "@/components/ui/Icons";

const STAGES = [
  { icon: SatelliteIcon, title: "Satellite & best-track", body: "IBTrACS position/intensity + HURSAT-B1 imagery" },
  { icon: DatabaseIcon, title: "Feature extraction", body: "Causal 48h sliding windows, storm-level split" },
  { icon: BrainIcon, title: "AI models", body: "LightGBM / CLIPER-style Ridge baselines, GRU exploratory" },
  { icon: MapPinIcon, title: "Forecast", body: "Stored per-storm predictions, read-only from the API" },
  { icon: ArrowRightIcon, title: "Geospatial visualization", body: "Map, charts, and Gemini explanation, this product" },
] as const;

export default function HowItWorks() {
  return (
    <section className="mx-auto max-w-7xl px-6 py-16">
      <SectionHeader
        eyebrow="How GeoStrom AI works"
        title="From raw archive to on-screen explanation"
        description="Every stage runs offline, once, ahead of time -- the live API you're using only ever reads what was already computed."
      />
      <div className="mt-10 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-5">
        {STAGES.map((s, i) => (
          <FadeIn key={s.title} delayMs={i * 60} className="relative">
            <GlassPanel className="flex h-full flex-col gap-2 p-4">
              <div className="flex items-center gap-2">
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-accent-soft/30 bg-accent/10 font-mono text-xs font-semibold text-accent-soft">
                  {i + 1}
                </span>
                <s.icon width={16} height={16} className="text-text-muted" />
              </div>
              <h3 className="text-sm font-semibold text-text-primary">{s.title}</h3>
              <p className="text-xs leading-relaxed text-text-secondary">{s.body}</p>
            </GlassPanel>
            {i < STAGES.length - 1 && (
              <span
                aria-hidden
                className="absolute -right-3 top-1/2 z-10 hidden -translate-y-1/2 text-border-strong lg:block"
              >
                <ArrowRightIcon width={14} height={14} />
              </span>
            )}
          </FadeIn>
        ))}
      </div>
      <FadeIn className="mt-6">
        <Link
          href="/methodology"
          className="inline-flex items-center gap-1.5 text-sm font-medium text-accent-soft hover:text-accent"
        >
          Read the full methodology <ArrowRightIcon width={14} height={14} />
        </Link>
      </FadeIn>
    </section>
  );
}
