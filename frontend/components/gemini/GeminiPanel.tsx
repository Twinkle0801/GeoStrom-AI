"use client";

/**
 * Calls the backend's Gemini endpoint ONLY (`explainForecast` in lib/api.ts
 * -> `POST /api/v1/explain/forecast`) -- never Gemini directly, never with
 * an API key in this bundle (there is none: the frontend has no Gemini
 * SDK, no key, nothing to expose). No free-text input is offered, so no
 * arbitrary user text can ever become an unrestricted prompt (task §10).
 */
import { useState } from "react";
import EvidenceDrawer from "@/components/gemini/EvidenceDrawer";
import GroundingBadge from "@/components/gemini/GroundingBadge";
import ErrorState from "@/components/ui/ErrorState";
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
    <div className="rounded-xl border border-border-subtle bg-white/[0.03] p-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h3 className="text-sm font-semibold text-text-primary">AI Analysis</h3>
          <p className="text-xs text-text-muted">Evidence-grounded explanation</p>
        </div>
        {result && <GroundingBadge result={result} />}
      </div>

      <div className="mt-4">
        {status === "idle" && (
          <button
            type="button"
            onClick={generate}
            className="rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-white transition-transform hover:scale-[1.02] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-soft"
          >
            Generate explanation
          </button>
        )}

        {status === "loading" && (
          <p role="status" className="text-sm text-text-secondary">
            Analyzing stored evidence…
          </p>
        )}

        {status === "error" && (
          <ErrorState
            title="Explanation unavailable. Showing deterministic evidence summary."
            detail={errorMessage}
            onRetry={generate}
          />
        )}

        {status === "success" && result && (
          <div className="space-y-4">
            <p className="text-sm leading-relaxed text-text-primary">{result.explanation.summary}</p>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <ExplanationBlock label="Intensity" text={result.explanation.intensity_explanation} />
              <ExplanationBlock label="Track" text={result.explanation.track_explanation} />
              <ExplanationBlock label="Classification" text={result.explanation.classification_explanation} />
              <ExplanationBlock label="Limitations" text={result.explanation.limitations} />
            </div>

            <div className="flex flex-wrap items-center gap-3 pt-1">
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
                className="rounded-md border border-border-subtle px-3 py-1.5 text-xs font-medium text-text-secondary transition-colors hover:bg-white/5 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
              >
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
          </div>
        )}
      </div>

      {drawerOpen && result && (
        <EvidenceDrawer evidence={result.evidence} onClose={() => setDrawerOpen(false)} />
      )}
    </div>
  );
}

function ExplanationBlock({ label, text }: { label: string; text: string }) {
  return (
    <div className="rounded-lg border border-border-subtle bg-white/[0.02] p-3">
      <div className="text-[11px] font-semibold uppercase tracking-wide text-text-muted">{label}</div>
      <p className="mt-1 text-xs text-text-secondary">{text}</p>
    </div>
  );
}
