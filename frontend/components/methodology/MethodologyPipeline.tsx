"use client";

/**
 * Visual restatement of the SAME 10-step pipeline the Phase 3 methodology
 * page always described in prose -- every `body` string below is verbatim
 * unchanged (task: "do not change the actual methodology"), only the
 * presentation (a connected, expandable pipeline) differs. The first step
 * is expanded by default; the rest use progressive disclosure so the page
 * doesn't read as one long wall of text.
 */
import { useState } from "react";
import { ChevronDownIcon } from "@/components/ui/Icons";

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

export default function MethodologyPipeline() {
  const [openIndex, setOpenIndex] = useState<number | null>(0);

  return (
    <ol className="relative">
      <div aria-hidden className="absolute bottom-4 left-[15px] top-4 w-px bg-border-subtle" />
      {PIPELINE.map((step, i) => {
        const open = openIndex === i;
        return (
          <li key={step.title} className="relative pb-3 pl-10">
            <span
              aria-hidden
              className={`absolute left-0 top-0 flex h-8 w-8 items-center justify-center rounded-full border font-mono text-xs font-semibold transition-colors ${
                open ? "border-accent-soft/60 bg-accent/15 text-accent-soft" : "border-border-subtle bg-bg-elevated text-text-secondary"
              }`}
            >
              {i + 1}
            </span>
            <button
              type="button"
              onClick={() => setOpenIndex(open ? null : i)}
              aria-expanded={open}
              className="flex w-full items-center justify-between gap-3 rounded-lg border border-border-subtle bg-white/[0.03] px-4 py-3 text-left transition-colors hover:bg-white/[0.05] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
            >
              <h3 className="text-sm font-semibold text-text-primary">{step.title}</h3>
              <ChevronDownIcon
                width={16} height={16}
                className={`shrink-0 text-text-muted transition-transform duration-200 ${open ? "rotate-180" : ""}`}
              />
            </button>
            {open && (
              <p className="mt-2 rounded-lg border border-border-subtle bg-white/[0.015] px-4 py-3 text-sm leading-relaxed text-text-secondary">
                {step.body}
              </p>
            )}
          </li>
        );
      })}
    </ol>
  );
}
