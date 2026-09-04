"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { listStorms, type CyclonesList } from "@/lib/api";
import { categoryColorClass, categoryLabel } from "@/lib/format";

type LoadState = "loading" | "empty" | "error" | "ready";

export default function StormSelector() {
  const [data, setData] = useState<CyclonesList | null>(null);
  const [state, setState] = useState<LoadState>("loading");
  const [errorMessage, setErrorMessage] = useState<string>("");

  useEffect(() => {
    let cancelled = false;
    setState("loading");
    listStorms({ limit: 100 })
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
  }, []);

  if (state === "loading") {
    return (
      <div className="space-y-2" data-testid="storm-selector-loading">
        {[...Array(5)].map((_, i) => (
          <div key={i} className="h-12 animate-pulse rounded-lg bg-white/5" />
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
    return (
      <div
        className="rounded-lg border border-border-subtle bg-white/5 p-6 text-center text-sm text-text-secondary"
        data-testid="storm-selector-empty"
      >
        No storms available. Run the Phase 2 ingestion script to populate the database.
      </div>
    );
  }

  return (
    <ul className="divide-y divide-border-subtle" data-testid="storm-selector-list">
      {data!.items.map((storm) => (
        <li key={storm.sid}>
          <Link
            href={`/predict/${storm.sid}`}
            className="flex items-center justify-between gap-4 px-3 py-3 transition-colors hover:bg-white/5"
          >
            <div>
              <div className="font-medium text-text-primary">
                {storm.name ?? storm.sid}
                <span className="ml-2 text-xs text-text-muted">{storm.season}</span>
              </div>
              <div className="text-xs text-text-secondary">
                {storm.n_observations} observations · {storm.split ?? "unassigned"} split
              </div>
            </div>
            <div className={`text-sm font-semibold tabular-nums ${categoryColorClass(storm.max_category)}`}>
              {categoryLabel(storm.max_category)}
            </div>
          </Link>
        </li>
      ))}
    </ul>
  );
}
