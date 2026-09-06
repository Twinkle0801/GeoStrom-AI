"use client";

import { useMemo, useState } from "react";
import StormSelector, { type StormSort } from "@/components/controls/StormSelector";
import { SearchIcon } from "@/components/ui/Icons";

/**
 * Filter chrome around the existing `StormSelector` (reused, not
 * duplicated). Basin is fixed to North Atlantic -- the project's only
 * covered basin (docs/PROJECT_REQUIREMENTS.md MVP scope) -- shown as a
 * disabled control rather than a fake multi-basin picker, so nothing here
 * implies data that does not exist. Sorting is a client-side reorder of the
 * already-fetched page (StormSelector's own `sort` handling), never a new
 * query parameter the backend doesn't support.
 */
export default function StormExplorer() {
  const [q, setQ] = useState("");
  const [season, setSeason] = useState<string>("");
  const [split, setSplit] = useState<string>("");
  const [sort, setSort] = useState<StormSort>("recent");

  const filters = useMemo(
    () => ({
      q: q.trim() || undefined,
      season: season ? Number(season) : undefined,
      split: split || undefined,
      sort,
    }),
    [q, season, split, sort],
  );

  return (
    <div>
      <form
        className="flex flex-wrap items-end gap-4 rounded-xl border border-border-subtle bg-white/[0.03] p-4"
        onSubmit={(e) => e.preventDefault()}
        role="search"
        aria-label="Storm filters"
      >
        <div className="flex flex-col gap-1">
          <label htmlFor="storm-search" className="text-xs font-medium text-text-secondary">
            Search storm ID
          </label>
          <div className="relative">
            <SearchIcon width={14} height={14} className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-text-muted" />
            <input
              id="storm-search"
              type="search"
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="e.g. 2010176N16278"
              className="w-56 rounded-md border border-border-subtle bg-bg-base py-1.5 pl-8 pr-3 text-sm text-text-primary placeholder:text-text-muted focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
            />
          </div>
        </div>

        <div className="flex flex-col gap-1">
          <label htmlFor="storm-season" className="text-xs font-medium text-text-secondary">
            Season
          </label>
          <input
            id="storm-season"
            type="number"
            inputMode="numeric"
            value={season}
            onChange={(e) => setSeason(e.target.value)}
            placeholder="Any"
            min={1980}
            max={2015}
            className="w-28 rounded-md border border-border-subtle bg-bg-base px-3 py-1.5 text-sm text-text-primary placeholder:text-text-muted focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          />
        </div>

        <div className="flex flex-col gap-1">
          <label htmlFor="storm-split" className="text-xs font-medium text-text-secondary">
            Evaluation split
          </label>
          <select
            id="storm-split"
            value={split}
            onChange={(e) => setSplit(e.target.value)}
            className="w-36 rounded-md border border-border-subtle bg-bg-base px-3 py-1.5 text-sm text-text-primary focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          >
            <option value="">Any</option>
            <option value="train">Train</option>
            <option value="val">Validation</option>
            <option value="test">Test</option>
          </select>
        </div>

        <div className="flex flex-col gap-1">
          <label htmlFor="storm-sort" className="text-xs font-medium text-text-secondary">
            Sort by
          </label>
          <select
            id="storm-sort"
            value={sort}
            onChange={(e) => setSort(e.target.value as StormSort)}
            className="w-36 rounded-md border border-border-subtle bg-bg-base px-3 py-1.5 text-sm text-text-primary focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          >
            <option value="recent">Most recent</option>
            <option value="season">Season (newest)</option>
            <option value="wind">Max wind (highest)</option>
            <option value="name">Name / ID</option>
          </select>
        </div>

        <div className="flex flex-col gap-1">
          <span className="text-xs font-medium text-text-secondary">Basin</span>
          <div
            className="w-40 rounded-md border border-border-subtle bg-white/[0.02] px-3 py-1.5 text-sm text-text-muted"
            aria-disabled
            title="North Atlantic is the project's only covered basin (MVP scope)"
          >
            North Atlantic
          </div>
        </div>
      </form>

      <div className="mt-6">
        <StormSelector filters={filters} />
      </div>
    </div>
  );
}
