"use client";

/**
 * Shows exactly the `EvidencePacket` the backend sent Gemini (task §11) --
 * no database implementation detail, no SQL, no secret ever appears here,
 * because the packet itself never contains one
 * (backend/app/gemini/schemas.py::EvidencePacket has no credential field,
 * by construction, since Phase 9).
 */
import { useEffect, useRef } from "react";
import type { ExplainResponse } from "@/lib/api";
import { formatTimestamp } from "@/lib/format";

export default function EvidenceDrawer({
  evidence, onClose,
}: {
  evidence: ExplainResponse["evidence"];
  onClose: () => void;
}) {
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <button
        type="button"
        aria-label="Close evidence drawer"
        onClick={onClose}
        className="absolute inset-0 bg-black/60 backdrop-blur-[1px]"
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="evidence-drawer-title"
        className="relative flex h-full w-full max-w-md flex-col overflow-y-auto border-l border-border-subtle bg-bg-elevated p-6 sm:max-w-lg"
      >
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 id="evidence-drawer-title" className="text-lg font-semibold text-text-primary">
              Evidence
            </h2>
            <p className="mt-1 text-xs text-text-muted">
              This explanation is grounded in stored project evidence — the exact packet below.
            </p>
          </div>
          <button
            ref={closeRef}
            type="button"
            onClick={onClose}
            className="rounded-md border border-border-subtle px-2 py-1 text-xs text-text-secondary hover:bg-white/5 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          >
            Close
          </button>
        </div>

        <div className="mt-6 space-y-6 text-sm">
          <Section title="Storm">
            <Row k="Name" v={evidence.storm.name ?? evidence.storm.sid} />
            <Row k="Storm ID" v={evidence.storm.sid} />
            <Row k="Season" v={String(evidence.storm.season)} />
            <Row k="Basin" v={evidence.storm.basin} />
            <Row k="Observations" v={String(evidence.storm.n_observations)} />
          </Section>

          {evidence.current_state && (
            <Section title="Observation window">
              <Row k="Timestamp" v={formatTimestamp(evidence.current_state.timestamp)} />
              <Row k="Position" v={`${evidence.current_state.lat.toFixed(2)}, ${evidence.current_state.lon.toFixed(2)}`} />
              <Row k="Wind" v={evidence.current_state.wind_kt != null ? `${evidence.current_state.wind_kt} kt` : "—"} />
              <Row k="Pressure" v={evidence.current_state.pressure_hpa != null ? `${evidence.current_state.pressure_hpa} hPa` : "—"} />
            </Section>
          )}

          {evidence.intensity && (
            <Section title="Intensity model">
              <Row k="Model" v={`${evidence.intensity.context.display_name} ${evidence.intensity.context.model_version}`} />
              <Row k="Dataset version" v={evidence.intensity.context.dataset_version} />
              <Row k="Forecast origin" v={formatTimestamp(evidence.intensity.origin_ts)} />
              {evidence.intensity.forecasts.map((f) => (
                <Row
                  key={f.lead_hours}
                  k={`+${f.lead_hours}h`}
                  v={`predicted ${f.pred_wind_kt?.toFixed(1) ?? "—"} kt${f.true_wind_kt != null ? ` · observed ${f.true_wind_kt.toFixed(1)} kt` : ""}`}
                />
              ))}
            </Section>
          )}

          {evidence.track && (
            <Section title="Track model">
              <Row k="Model" v={`${evidence.track.context.display_name} ${evidence.track.context.model_version}`} />
              <Row k="Forecast origin" v={formatTimestamp(evidence.track.origin_ts)} />
              {evidence.track.forecasts.map((f) => (
                <Row
                  key={f.lead_hours}
                  k={`+${f.lead_hours}h`}
                  v={f.pred_lat != null && f.pred_lon != null ? `${f.pred_lat.toFixed(2)}, ${f.pred_lon.toFixed(2)}` : "—"}
                />
              ))}
            </Section>
          )}

          <Section title="Classification">
            {evidence.classification ? (
              <>
                <Row k="Label" v={evidence.classification.class_label} />
                <Row k="Model" v={`${evidence.classification.model_name} ${evidence.classification.model_version}`} />
                <Row k="Confidence" v={evidence.classification.confidence != null ? `${(evidence.classification.confidence * 100).toFixed(0)}%` : "Not available"} />
              </>
            ) : (
              <p className="text-xs text-text-muted">No classification result in this packet.</p>
            )}
          </Section>

          <Section title="Satellite source">
            <p className="text-xs text-text-muted">Not included in this packet (text/structured-data grounded only).</p>
          </Section>

          <Section title="Known limitations">
            <ul className="list-inside list-disc space-y-1 text-xs text-text-secondary">
              {evidence.known_limitations.map((l) => (
                <li key={l}>{l}</li>
              ))}
            </ul>
          </Section>

          <div className="border-t border-border-subtle pt-3 text-[11px] text-text-muted">
            Evidence schema {evidence.evidence_schema_version} · generated {formatTimestamp(evidence.generated_at)}
          </div>
        </div>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h3 className="text-[11px] font-semibold uppercase tracking-wide text-text-muted">{title}</h3>
      <div className="mt-2 space-y-1">{children}</div>
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4 text-xs">
      <span className="text-text-muted">{k}</span>
      <span className="text-right tabular-nums text-text-secondary">{v}</span>
    </div>
  );
}
