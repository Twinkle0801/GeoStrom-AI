"use client";

/**
 * Scrubs through the storm's REAL observation timestamps (task: "Use
 * actual timestamps. Do not create artificial interpolation if the
 * underlying timestamp does not exist.") -- every index this component can
 * land on corresponds to one real IBTrACS observation row. A tick is drawn
 * brighter where a real model forecast was actually issued at that
 * timestamp (`originTimestamps`); scrubbing to any other timestamp still
 * shows the real observed position, just with the prediction-dependent
 * panels in their empty state (handled by the parent, not this component).
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { formatTimestamp } from "@/lib/format";

export default function TimeScrubber({
  timestamps, originTimestamps, index, onChange,
}: {
  timestamps: string[];
  originTimestamps: Set<string>;
  index: number;
  onChange: (index: number) => void;
}) {
  const [playing, setPlaying] = useState(false);

  // Setting `playing` false is enough: the effect below depends on
  // `playing` and its cleanup (`clearInterval`) runs automatically.
  const stop = useCallback(() => setPlaying(false), []);

  // Advance using the latest index via a ref, avoiding the stale-closure
  // problem a plain `[playing, index]` dependency array would introduce
  // (which would otherwise recreate the interval, and drift, every tick).
  const indexRef = useRef(index);
  indexRef.current = index;
  useEffect(() => {
    if (!playing) return undefined;
    const id = setInterval(() => {
      const next = indexRef.current + 1;
      if (next >= timestamps.length) {
        stop();
        return;
      }
      onChange(next);
    }, 900);
    return () => clearInterval(id);
  }, [playing, timestamps.length, onChange, stop]);

  if (timestamps.length === 0) {
    return null;
  }

  const current = timestamps[index];

  return (
    <div className="rounded-xl border border-border-subtle bg-white/[0.03] p-4">
      <div className="flex items-center justify-between gap-4">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-wide text-text-muted">
            Observed timestamp
          </div>
          <div className="tabular-nums text-sm font-medium text-text-primary">
            {formatTimestamp(current)}
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          <ScrubButton
            label="Previous timestamp"
            disabled={index === 0}
            onClick={() => {
              stop();
              onChange(Math.max(0, index - 1));
            }}
          >
            <PrevIcon />
          </ScrubButton>
          <ScrubButton
            label={playing ? "Pause" : "Play"}
            onClick={() => setPlaying((p) => !p)}
          >
            {playing ? <PauseIcon /> : <PlayIcon />}
          </ScrubButton>
          <ScrubButton
            label="Next timestamp"
            disabled={index === timestamps.length - 1}
            onClick={() => {
              stop();
              onChange(Math.min(timestamps.length - 1, index + 1));
            }}
          >
            <NextIcon />
          </ScrubButton>
        </div>
      </div>

      <div className="relative mt-4">
        <input
          type="range"
          min={0}
          max={timestamps.length - 1}
          step={1}
          value={index}
          onChange={(e) => {
            stop();
            onChange(Number(e.target.value));
          }}
          aria-label="Storm timeline"
          aria-valuetext={formatTimestamp(current)}
          className="w-full accent-accent"
        />
        {/* Real-forecast-origin ticks -- purely informational, never interactive on their own */}
        <div aria-hidden className="pointer-events-none absolute inset-x-0 top-1/2 flex justify-between px-[2px]">
          {timestamps.map((ts, i) => (
            <span
              key={ts}
              className={
                originTimestamps.has(ts)
                  ? "block h-1.5 w-0.5 -translate-y-1/2 rounded-full bg-predicted/70"
                  : "block h-0.5 w-0.5 -translate-y-1/2 rounded-full bg-white/10"
              }
              style={{ marginLeft: i === 0 ? 0 : undefined }}
            />
          ))}
        </div>
      </div>
      <p className="mt-2 text-[11px] text-text-muted">
        Amber ticks mark timestamps with an issued model forecast. Drag, use arrow keys, or press
        play to move through the storm&apos;s real observation history.
      </p>
    </div>
  );
}

function ScrubButton({
  children, label, onClick, disabled,
}: {
  children: React.ReactNode;
  label: string;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      disabled={disabled}
      onClick={onClick}
      className="flex h-8 w-8 items-center justify-center rounded-md border border-border-subtle text-text-secondary transition-colors hover:bg-white/5 hover:text-text-primary disabled:opacity-30 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
    >
      {children}
    </button>
  );
}

function PlayIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor" aria-hidden>
      <path d="M4 2.5v11l10-5.5-10-5.5z" />
    </svg>
  );
}
function PauseIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor" aria-hidden>
      <rect x="4" y="2.5" width="3" height="11" />
      <rect x="9" y="2.5" width="3" height="11" />
    </svg>
  );
}
function PrevIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor" aria-hidden>
      <path d="M12 2.5v11L4 8l8-5.5z" />
    </svg>
  );
}
function NextIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor" aria-hidden>
      <path d="M4 2.5v11l8-5.5-8-5.5z" />
    </svg>
  );
}
