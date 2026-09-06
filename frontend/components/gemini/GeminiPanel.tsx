"use client";

/**
 * Calls the backend's Gemini endpoint ONLY (`explainForecast` in lib/api.ts
 * -> `POST /api/v1/explain/forecast`) -- never Gemini directly, never with
 * an API key in this bundle (there is none: the frontend has no Gemini
 * SDK, no key, nothing to expose). No free-text input is offered, so no
 * arbitrary user text can ever become an unrestricted prompt (task §10).
 *
 * Visual refinement only: request/response handling, the fetch call, and
 * every tested string/label below are unchanged from before.
 */
import { useState } from "react";
import { motion } from "framer-motion";
import EvidenceDrawer from "@/components/gemini/EvidenceDrawer";
import GroundingBadge from "@/components/gemini/GroundingBadge";
import ErrorState from "@/components/ui/ErrorState";
import { CopyIcon, GaugeIcon, MapPinIcon, SatelliteIcon, SparklesIcon } from "@/components/ui/Icons";
import { fadeUp, reducedMotionVariants, usePrefersReducedMotion } from "@/lib/motion";
import { ApiError, explainForecast, type ExplainResponse } from "@/lib/api";

type Status = "idle" | "loading" | "success" | "error";

export default function GeminiPanel({
  sid, intensityModelVersion, trackModelVersion,
}: {
  sid: string;
  intensityModelVersion?: string;
  trackModelVersion?: string;
}) {
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<ExplainResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const reducedMotion = usePrefersReducedMotion();

  const generate = () => {
    setStatus("loading");
    setErrorMessage("");
    explainForecast(sid, { intensityModelVersion, trackModelVersion })
      .then((res) => {
        setResult(res);
        setStatus("success");
      })
      .catch((err: unknown) => {
        setErrorMessage(err instanceof ApiError ? err.message : "Request failed");
        setStatus("error");
      });
  };

  const copy = async () => {
    if (!result) return;
    try {
      await navigator.clipboard.writeText(result.explanation.summary);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard API can be unavailable (permissions, non-secure context) --
      // fail silently rather than showing a scary error for a non-critical action.
    }
  };

  return (
    <div className="relative overflow-hidden rounded-xl border border-border-subtle bg-gradient-to-b from-violet-500/[0.04] via-white/[0.03] to-white/[0.03] p-5 sm:p-6">
      <div aria-hidden className="pointer-events-none absolute -right-16 -top-16 h-48 w-48 rounded-full bg-violet-500/10 blur-3xl" />
      <div className="relative flex items-start justify-between gap-4">
        <div className="flex items-start gap-3">
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-violet-400/30 bg-violet-400/10 text-violet-300">
            <SparklesIcon width={16} height={16} />
          </span>
          <div>
            <h3 className="text-sm font-semibold text-text-primary">AI Forecast Interpretation</h3>
            <p className="text-xs text-text-muted">Evidence-grounded explanation, never a new claim</p>
          </div>
        </div>
        {result && <GroundingBadge result={result} />}
      </div>

      <div className="relative mt-5">
        {status === "idle" && (
          <button
            type="button"
            onClick={generate}
            className="group inline-flex items-center gap-2 rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-white shadow-glow transition-all duration-300 ease-premium hover:scale-[1.02] hover:shadow-glow-soft focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-soft"
          >
            <SparklesIcon width={14} height={14} />
            Generate explanation
          </button>
        )}

        {status === "loading" && (
          <div role="status" className="flex items-center gap-3 text-sm text-text-secondary">
            <span className="flex gap-1">
              <span className="h-1.5 w-1.5 animate-pulse-glow rounded-full bg-violet-400 [animation-delay:0ms]" />
              <span className="h-1.5 w-1.5 animate-pulse-glow rounded-full bg-violet-400 [animation-delay:200ms]" />
              <span className="h-1.5 w-1.5 animate-pulse-glow rounded-full bg-violet-400 [animation-delay:400ms]" />
            </span>
            Analyzing stored evidence…
          </div>
        )}

        {status === "error" && (
          <ErrorState
            title="Explanation unavailable. Showing deterministic evidence summary."
            detail={errorMessage}
            onRetry={generate}
          />
        )}

        {status === "success" && result && (
          <motion.div
            initial="hidden"
            animate="visible"
            variants={reducedMotion ? reducedMotionVariants : fadeUp}
            className="space-y-4"
          >
            <p className="text-sm leading-relaxed text-text-primary">{result.explanation.summary}</p>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <ExplanationBlock icon={GaugeIcon} label="Intensity" text={result.explanation.intensity_explanation} />
              <ExplanationBlock icon={MapPinIcon} label="Track" text={result.explanation.track_explanation} />
              <ExplanationBlock icon={SatelliteIcon} label="Classification" text={result.explanation.classification_explanation} />
              <ExplanationBlock icon={SparklesIcon} label="Limitations" text={result.explanation.limitations} />
            </div>

            <div className="flex flex-wrap items-center gap-2 pt-1">
              <button
                type="button"
                onClick={generate}
                className="rounded-md border border-border-subtle px-3 py-1.5 text-xs font-medium text-text-secondary transition-colors hover:bg-white/5 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
              >
                Regenerate
              </button>
              <button
                type="button"
                onClick={copy}
                className="inline-flex items-center gap-1.5 rounded-md border border-border-subtle px-3 py-1.5 text-xs font-medium text-text-secondary transition-colors hover:bg-white/5 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
              >
                <CopyIcon width={12} height={12} />
                {copied ? "Copied" : "Copy"}
              </button>
              <button
                type="button"
                onClick={() => setDrawerOpen(true)}
                className="rounded-md border border-border-subtle px-3 py-1.5 text-xs font-medium text-text-secondary transition-colors hover:bg-white/5 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
              >
                View evidence
              </button>
            </div>

            <p className="text-[11px] italic text-text-muted">{result.disclaimer}</p>
          </motion.div>
        )}
      </div>

      {drawerOpen && result && (
        <EvidenceDrawer evidence={result.evidence} onClose={() => setDrawerOpen(false)} />
      )}
    </div>
  );
}

function ExplanationBlock({
  icon: Icon, label, text,
}: {
  icon: (props: React.SVGProps<SVGSVGElement>) => React.ReactElement;
  label: string;
  text: string;
}) {
  return (
    <div className="rounded-lg border border-border-subtle bg-white/[0.02] p-3">
      <div className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-text-muted">
        <Icon width={12} height={12} />
        {label}
      </div>
      <p className="mt-1.5 text-xs leading-relaxed text-text-secondary">{text}</p>
    </div>
  );
}
