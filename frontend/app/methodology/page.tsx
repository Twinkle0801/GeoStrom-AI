import SectionHeader from "@/components/ui/SectionHeader";
import GlassPanel from "@/components/ui/GlassPanel";

export const metadata = {
  title: "Methodology — GeoStrom AI",
  description: "How GeoStrom AI's data, models, and evidence-grounded explanations are built.",
};

const PIPELINE = [
  { title: "Data sources", body: "IBTrACS (best-track position/intensity), HURSAT-B1 (satellite imagery), and ADT-HURSAT (Dvorak scene labels) are combined -- no single source alone provides position, intensity, imagery, and pattern labels together." },
  { title: "Satellite preprocessing", body: "HURSAT-B1 frames are quality-controlled (viewing-angle deduplication, spatial/temporal gates) and written to a canonical Zarr store, verified end-to-end against real archive data." },
  { title: "Storm/observation alignment", body: "Satellite frames and ADT scene labels are joined to IBTrACS observations by storm ID and timestamp, producing one fused record per valid (storm, time) pair." },
  { title: "Feature engineering", body: "Causal, lag-based sliding windows (48h of history, 6-hourly) are built from each storm's own prior observations -- no feature is ever built from a timestep after the one it describes." },
  { title: "Storm-level splitting", body: "Train/validation/test are split by whole storm and season block, never by individual observation -- a storm's early observations can never leak into a split containing its later ones, and the frozen split is committed once and reused unmodified." },
  { title: "Model training", body: "Persistence, Ridge/CLIPER-style, and LightGBM baselines are trained per task; exploratory GRU sequence models are trained separately and evaluated against the same frozen split, never used to redefine the baseline." },
  { title: "Evaluation", body: "Every model is scored exactly once on the held-out test split -- MAE/RMSE/bias for intensity, great-circle error for track, macro-F1 for classification -- and compared against persistence and each other." },
  { title: "Prediction storage", body: "Baseline model predictions are written to PostgreSQL/PostGIS as long-form rows (storm, origin time, horizon, model), read-only from that point on -- the API never computes a prediction at request time." },
  { title: "Evidence packet", body: "A backend service assembles a versioned, typed JSON packet from these stored rows for one storm/forecast -- observed state, predictions, model metrics, known limitations. Nothing in it is invented." },
  { title: "Gemini explanation", body: "Gemini converts that packet into plain-language text. A deterministic validator checks every numeric and categorical claim against the packet before anything is shown; an ungrounded response is discarded in favour of a template built directly from the same evidence." },
] as const;

export default function MethodologyPage() {
  return (
    <main className="mx-auto max-w-4xl px-6 py-10">
      <SectionHeader
        eyebrow="Methodology"
        title="How GeoStrom AI is built"
        description="An end-to-end view of the pipeline, from raw archives to the explanation on screen."
      />

      <ol className="mt-8 space-y-3">
        {PIPELINE.map((step, i) => (
          <li key={step.title}>
            <GlassPanel className="flex gap-4 p-4">
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-border-subtle text-xs font-semibold text-text-secondary">
                {i + 1}
              </div>
              <div>
                <h3 className="text-sm font-semibold text-text-primary">{step.title}</h3>
                <p className="mt-1 text-sm text-text-secondary">{step.body}</p>
              </div>
            </GlassPanel>
          </li>
        ))}
      </ol>

      <section className="mt-12 space-y-6">
        <div>
          <h2 className="text-lg font-semibold text-text-primary">Why storm-level splitting matters</h2>
          <p className="mt-2 text-sm text-text-secondary">
            A tropical cyclone&apos;s consecutive observations are highly correlated -- wind at hour
            6 is a near-perfect predictor of wind at hour 12. Splitting by individual observation
            (rather than by whole storm) would let a model see part of a storm&apos;s life in
            training and be tested on the rest of the same storm, producing an optimistic score
            that would not hold on a genuinely new storm. GeoStrom AI freezes its split at the
            storm level, once, and every model in this product is evaluated against that same
            frozen split.
          </p>
        </div>

        <div>
          <h2 className="text-lg font-semibold text-text-primary">A retrospective system, not an operational forecast</h2>
          <p className="mt-2 text-sm text-text-secondary">
            Every storm shown in GeoStrom AI has already happened; every &ldquo;prediction&rdquo;
            is a historical baseline model output, generated from data available before the
            forecast horizon and evaluated against the outcome that was later observed. This
            product does not ingest live data, does not run in real time, and is not a substitute
            for an official forecast, warning, or advisory from a national meteorological agency.
          </p>
        </div>

        <div>
          <h2 className="text-lg font-semibold text-text-primary">Baselines vs. exploratory models</h2>
          <p className="mt-2 text-sm text-text-secondary">
            LightGBM (intensity) and CLIPER-style Ridge (track) are the current production
            baselines -- the strongest models found on the frozen test split. GRU sequence models
            were also trained and evaluated as a research exploration; at the current dataset
            scale they did not beat the tabular baselines on either task, and are labelled
            <em> exploratory</em> everywhere they appear, never presented as the recommended
            model. See <a href="/models" className="underline hover:text-text-secondary">Model
            Performance</a> for the exact numbers.
          </p>
        </div>
      </section>
    </main>
  );
}
