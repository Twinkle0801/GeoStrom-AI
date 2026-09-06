import MethodologyPipeline from "@/components/methodology/MethodologyPipeline";
import GlassPanel from "@/components/ui/GlassPanel";
import SectionHeader from "@/components/ui/SectionHeader";
import { AlertTriangleIcon, CheckCircleIcon, DatabaseIcon } from "@/components/ui/Icons";

export const metadata = {
  title: "Methodology — GeoStrom AI",
  description: "How GeoStrom AI's data, models, and evidence-grounded explanations are built.",
};

const CALLOUTS = [
  {
    icon: DatabaseIcon,
    title: "Why storm-level splitting matters",
    body: "A tropical cyclone's consecutive observations are highly correlated -- wind at hour 6 is a near-perfect predictor of wind at hour 12. Splitting by individual observation (rather than by whole storm) would let a model see part of a storm's life in training and be tested on the rest of the same storm, producing an optimistic score that would not hold on a genuinely new storm. GeoStrom AI freezes its split at the storm level, once, and every model in this product is evaluated against that same frozen split.",
  },
  {
    icon: AlertTriangleIcon,
    title: "A retrospective system, not an operational forecast",
    body: "Every storm shown in GeoStrom AI has already happened; every “prediction” is a historical baseline model output, generated from data available before the forecast horizon and evaluated against the outcome that was later observed. This product does not ingest live data, does not run in real time, and is not a substitute for an official forecast, warning, or advisory from a national meteorological agency.",
  },
  {
    icon: CheckCircleIcon,
    title: "Baselines vs. exploratory models",
    body: "LightGBM (intensity) and CLIPER-style Ridge (track) are the current production baselines -- the strongest models found on the frozen test split. GRU sequence models were also trained and evaluated as a research exploration; at the current dataset scale they did not beat the tabular baselines on either task, and are labelled exploratory everywhere they appear, never presented as the recommended model.",
    link: { href: "/models", label: "Model Performance" },
  },
] as const;

export default function MethodologyPage() {
  return (
    <main className="relative mx-auto max-w-5xl px-6 py-10">
      <div
        aria-hidden
        className="pointer-events-none absolute inset-x-0 top-0 -z-10 h-72 bg-[radial-gradient(60%_60%_at_50%_0%,rgba(76,141,255,0.08),transparent_70%)]"
      />
      <SectionHeader
        eyebrow="Methodology"
        title="How GeoStrom AI is built"
        description="An end-to-end view of the pipeline, from raw archives to the explanation on screen."
      />

      <div className="mt-10">
        <MethodologyPipeline />
      </div>

      <section className="mt-14 space-y-4">
        <SectionHeader title="Why this holds up to scrutiny" />
        <div className="mt-2 grid grid-cols-1 gap-4 sm:grid-cols-3">
          {CALLOUTS.map((c) => (
            <GlassPanel key={c.title} className="flex h-full flex-col gap-2 p-5">
              <c.icon width={18} height={18} className="text-accent-soft" />
              <h3 className="text-sm font-semibold text-text-primary">{c.title}</h3>
              <p className="text-xs leading-relaxed text-text-secondary">
                {c.body}
                {"link" in c && c.link && (
                  <>
                    {" "}See{" "}
                    <a href={c.link.href} className="underline decoration-text-muted/40 underline-offset-2 hover:text-text-secondary hover:decoration-text-secondary">
                      {c.link.label}
                    </a>{" "}
                    for the exact numbers.
                  </>
                )}
              </p>
            </GlassPanel>
          ))}
        </div>
      </section>
    </main>
  );
}
