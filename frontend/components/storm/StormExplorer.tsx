"use client";

import { useMemo, useState } from "react";
import StormSelector from "@/components/controls/StormSelector";

/**
 * Filter chrome around the existing `StormSelector` (reused, not
 * duplicated). Basin is fixed to North Atlantic -- the project's only
 * covered basin (docs/PROJECT_REQUIREMENTS.md MVP scope) -- shown as a
 * disabled control rather than a fake multi-basin picker, so nothing here
 * implies data that does not exist.
 */
export default function StormExplorer() {
  const [q, setQ] = useState("");
  const [season, setSeason] = useState<string>("");
  const [split, setSplit] = useState<string>("");

  const filters = useMemo(
    () => ({
      q: q.trim() || undefined,
      season: season ? Number(season) : undefined,
      split: split || undefined,
    }),
    [q, season, split],
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
          <input
            id="storm-search"
            type="search"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="e.g. 2010176N16278"
            className="w-56 rounded-md border border-border-subtle bg-bg-base px-3 py-1.5 text-sm text-text-primary placeholder:text-text-muted focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          />
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
