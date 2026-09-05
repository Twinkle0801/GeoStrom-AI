"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { listStorms, type CyclonesList } from "@/lib/api";
import { categoryColorClass, categoryLabel } from "@/lib/format";

type LoadState = "loading" | "empty" | "error" | "ready";

export interface StormSelectorFilters {
  season?: number;
  split?: string;
  q?: string;
}

/**
 * The storm list, reused by both the home page (Phase 3 legacy usage,
 * still exercised with no props by StormSelector.test.tsx) and the Phase
 * 10 Storm Explorer page (`app/storms/page.tsx`), which passes live filter
 * state down as `filters` -- an additive, backward-compatible prop, never
 * a rewrite of this component's default (no-filter) behaviour.
 */
export default function StormSelector({ filters }: { filters?: StormSelectorFilters }) {
  const [data, setData] = useState<CyclonesList | null>(null);
  const [state, setState] = useState<LoadState>("loading");
  const [errorMessage, setErrorMessage] = useState<string>("");

  useEffect(() => {
    let cancelled = false;
    setState("loading");
    listStorms({ limit: 100, season: filters?.season, split: filters?.split, q: filters?.q })
      .then((res) => {
        if (cancelled) return;
        setData(res);
        setState(res.items.length === 0 ? "empty" : "ready");
      })
      .catch((err: Error) => {
        if (cancelled) return;
        setErrorMessage(err.message || "Failed to load storms");
        setState("error");
      });
    return () => {
      cancelled = true;
    };
  }, [filters?.season, filters?.split, filters?.q]);

  if (state === "loading") {
    return (
      <div className="space-y-2 p-1" data-testid="storm-selector-loading">
        {[...Array(5)].map((_, i) => (
          <div key={i} className="h-16 animate-pulse rounded-lg bg-white/5" />
        ))}
      </div>
    );
  }

  if (state === "error") {
    return (
      <div
        className="rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-300"
        data-testid="storm-selector-error"
      >
        Could not load storms: {errorMessage}
        <div className="mt-1 text-xs text-red-400">
          Is the backend running at the configured API URL?
        </div>
      </div>
    );
  }

  if (state === "empty") {
    const hasFilters = Boolean(filters?.season || filters?.split || filters?.q);
    return (
      <div
        className="rounded-lg border border-border-subtle bg-white/5 p-6 text-center text-sm text-text-secondary"
        data-testid="storm-selector-empty"
      >
        {hasFilters
          ? "No storms match these filters."
          : "No storms available. Run the Phase 2 ingestion script to populate the database."}
      </div>
    );
  }

  return (
    <ul
      className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3"
      data-testid="storm-selector-list"
    >
      {data!.items.map((storm) => (
        <li key={storm.sid}>
          <Link
            href={`/predict/${storm.sid}`}
            className="flex h-full flex-col justify-between gap-3 rounded-xl border border-border-subtle bg-white/[0.03] p-4 transition-colors hover:border-accent-soft/40 hover:bg-white/[0.06] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          >
            <div className="flex items-start justify-between gap-2">
              <div>
                <div className="font-medium text-text-primary">{storm.name ?? storm.sid}</div>
                {storm.name && <div className="text-xs text-text-muted">{storm.sid}</div>}
              </div>
              <span className={`shrink-0 text-sm font-semibold tabular-nums ${categoryColorClass(storm.max_category)}`}>
                {categoryLabel(storm.max_category)}
              </span>
            </div>
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-text-secondary">
              <span>{storm.season}</span>
              <span aria-hidden>·</span>
              <span>{storm.n_observations} observations</span>
              {storm.max_wind_kt != null && (
                <>
                  <span aria-hidden>·</span>
                  <span>{storm.max_wind_kt.toFixed(0)} kt max</span>
                </>
              )}
            </div>
            <div className="rounded bg-white/5 px-2 py-0.5 text-[11px] text-text-muted w-fit">
              {storm.split ?? "unassigned"} split
            </div>
          </Link>
        </li>
      ))}
    </ul>
  );
}
