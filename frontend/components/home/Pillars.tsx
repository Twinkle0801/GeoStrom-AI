/**
 * Zero-JS server component -- see FadeIn.tsx (Phase 13 performance audit:
 * replaced a per-card framer-motion instance with a pure-CSS entrance).
 */
import GlassPanel from "@/components/ui/GlassPanel";
import SectionHeader from "@/components/ui/SectionHeader";
import FadeIn from "@/components/ui/FadeIn";

const PILLARS = [
  {
    title: "Observed evidence",
    body: "Historical IBTrACS positions, wind, and pressure — real measurements, never model output.",
  },
  {
    title: "Model predictions",
    body: "Baseline and exploratory models forecast intensity and track from that history, retrospectively evaluated on held-out storms.",
  },
  {
    title: "Satellite structure",
    body: "HURSAT-B1 imagery and ADT-derived Dvorak scene labels describe the storm's visual structure where available.",
  },
  {
    title: "Evidence-grounded explanation",
    body: "Gemini narrates the stored evidence in plain language — it never computes a number, and every claim is checked against the record.",
  },
] as const;

export default function Pillars() {
  return (
    <section className="mx-auto max-w-7xl px-6 py-16">
      <SectionHeader
        eyebrow="What GeoStrom AI shows"
        title="Four kinds of information — never blended"
        description="Every panel in this product states plainly whether you're looking at an observation, a prediction, satellite-derived structure, or an AI-generated explanation."
      />
      <div className="mt-8 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {PILLARS.map((p, i) => (
          <FadeIn key={p.title} delayMs={i * 60}>
            <GlassPanel className="h-full p-5">
              <h3 className="text-sm font-semibold text-text-primary">{p.title}</h3>
              <p className="mt-2 text-sm text-text-secondary">{p.body}</p>
            </GlassPanel>
          </FadeIn>
        ))}
      </div>
    </section>
  );
}
