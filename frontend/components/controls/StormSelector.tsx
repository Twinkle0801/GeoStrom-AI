"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import Badge from "@/components/ui/Badge";
import { CardSkeleton } from "@/components/ui/Skeletons";
import { ClockIcon, GaugeIcon, WindIcon } from "@/components/ui/Icons";
import { listStorms, type CyclonesList } from "@/lib/api";
import { categoryColorClass, categoryLabel } from "@/lib/format";

type LoadState = "loading" | "empty" | "error" | "ready";
export type StormSort = "recent" | "season" | "wind" | "name";

export interface StormSelectorFilters {
  season?: number;
  split?: string;
  q?: string;
  /** Client-side reorder of the already-fetched page -- never a new API
   * call, since sorting doesn't change WHICH storms match the filters,
   * only their display order. Optional and defaulted so every existing
   * caller/test that never passes it keeps the original API-given order. */
  sort?: StormSort;
}

function sortItems(items: CyclonesList["items"], sort?: StormSort): CyclonesList["items"] {
  if (!sort || sort === "recent") return items;
  const copy = [...items];
  if (sort === "season") return copy.sort((a, b) => b.season - a.season);
  if (sort === "wind") return copy.sort((a, b) => (b.max_wind_kt ?? -1) - (a.max_wind_kt ?? -1));
  if (sort === "name") return copy.sort((a, b) => (a.name ?? a.sid).localeCompare(b.name ?? b.sid));
  return copy;
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
      <div className="grid grid-cols-1 gap-3 p-1 sm:grid-cols-2 lg:grid-cols-3" data-testid="storm-selector-loading">
        {[...Array(6)].map((_, i) => (
          <CardSkeleton key={i} />
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

  const items = sortItems(data!.items, filters?.sort);

  return (
    <>
      <div className="mb-3 text-xs text-text-muted">
        {data!.total} storm{data!.total === 1 ? "" : "s"} match{data!.total === 1 ? "es" : ""}
      </div>
      <ul
        className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3"
        data-testid="storm-selector-list"
      >
        {items.map((storm) => (
          <li key={storm.sid}>
            <Link
              href={`/predict/${storm.sid}`}
              className="group flex h-full flex-col justify-between gap-3 rounded-xl border border-border-subtle bg-white/[0.03] p-4 transition-all duration-300 ease-premium hover:-translate-y-0.5 hover:border-accent-soft/40 hover:bg-white/[0.06] hover:shadow-panel focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
            >
              <div className="flex items-start justify-between gap-2">
                <div>
                  <div className="font-medium text-text-primary">{storm.name ?? storm.sid}</div>
                  {storm.name && <div className="font-mono text-xs text-text-muted">{storm.sid}</div>}
                </div>
                <span className={`shrink-0 text-sm font-semibold tabular-nums ${categoryColorClass(storm.max_category)}`}>
                  {categoryLabel(storm.max_category)}
                </span>
              </div>
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 font-mono text-xs text-text-secondary">
                <span className="flex items-center gap-1">
                  <ClockIcon width={12} height={12} className="text-text-muted" />
                  {storm.season}
                </span>
                {storm.max_wind_kt != null && (
                  <span className="flex items-center gap-1">
                    <WindIcon width={12} height={12} className="text-text-muted" />
                    {storm.max_wind_kt.toFixed(0)} kt
                  </span>
                )}
                {storm.min_pressure_hpa != null && (
                  <span className="flex items-center gap-1">
                    <GaugeIcon width={12} height={12} className="text-text-muted" />
                    {storm.min_pressure_hpa.toFixed(0)} hPa
                  </span>
                )}
              </div>
              <div className="flex items-center justify-between gap-2">
                <Badge tone="neutral">{storm.split ?? "unassigned"} split</Badge>
                <span className="text-[11px] text-text-muted opacity-0 transition-opacity group-hover:opacity-100">
                  Open analysis →
                </span>
              </div>
            </Link>
          </li>
        ))}
      </ul>
    </>
  );
}
