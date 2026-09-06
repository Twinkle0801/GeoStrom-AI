/**
 * The four real, already-documented model capabilities this project has
 * (docs/DEVELOPMENT_ROADMAP.md's own phase list) -- described here as
 * plain-language cards, not re-derived data. "Pattern Classification"
 * links out to the honest empty-state on /predict/[sid] rather than
 * implying a live per-storm result exists (it does not -- see
 * ClassificationPanel.tsx).
 *
 * No "use client" needed -- `FadeIn` is a plain CSS-animated div, so this
 * whole section renders as a zero-JS server component (Phase 13
 * performance audit finding, see FadeIn.tsx's docstring).
 */
import GlassPanel from "@/components/ui/GlassPanel";
import SectionHeader from "@/components/ui/SectionHeader";
import Badge from "@/components/ui/Badge";
import FadeIn from "@/components/ui/FadeIn";
import { GaugeIcon, MapPinIcon, SatelliteIcon, WindIcon } from "@/components/ui/Icons";

const CAPABILITIES = [
  {
    icon: SatelliteIcon,
    title: "Pattern Identification",
    tier: null,
    body: "HURSAT-B1 satellite imagery, quality-controlled and aligned to each storm's observation timeline, describes visual structure where archive coverage exists.",
  },
  {
    icon: GaugeIcon,
    title: "Pattern Classification",
    tier: "exploratory" as const,
    body: "A frozen Dvorak-derived scene taxonomy (CDO, CurvedBand, Eye, Shear) with a Logistic Regression production baseline; CNN/ResNet-18 were evaluated as exploratory research and did not beat it.",
  },
  {
    icon: WindIcon,
    title: "Intensity Forecast",
    tier: "recommended" as const,
    body: "LightGBM is the recommended production baseline for +6/+12/+18/+24h wind-speed forecasts, evaluated on a storm-disjoint held-out test split.",
  },
  {
    icon: MapPinIcon,
    title: "Track Forecast",
    tier: "recommended" as const,
    body: "A CLIPER-style Ridge model is the recommended production baseline for future-position forecasts, benchmarked against Persistence and LightGBM.",
  },
] as const;

export default function Capabilities() {
  return (
    <section className="mx-auto max-w-7xl px-6 py-16">
      <SectionHeader
        eyebrow="Capabilities"
        title="Four stages of cyclone intelligence"
        description="Each capability below is backed by real, benchmarked models on the frozen test split -- see Model Performance for exact numbers."
      />
      <div className="mt-8 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {CAPABILITIES.map((c, i) => (
          <FadeIn key={c.title} delayMs={i * 60}>
            <GlassPanel hover className="flex h-full flex-col gap-3 p-5">
              <div className="flex items-center justify-between">
                <span className="flex h-9 w-9 items-center justify-center rounded-lg border border-border-subtle bg-white/[0.04] text-accent-soft">
                  <c.icon width={18} height={18} />
                </span>
                {c.tier === "recommended" && <Badge tone="recommended">Baseline</Badge>}
                {c.tier === "exploratory" && <Badge tone="exploratory">Exploratory</Badge>}
              </div>
              <h3 className="text-sm font-semibold text-text-primary">{c.title}</h3>
              <p className="text-sm leading-relaxed text-text-secondary">{c.body}</p>
            </GlassPanel>
          </FadeIn>
        ))}
      </div>
    </section>
  );
}
