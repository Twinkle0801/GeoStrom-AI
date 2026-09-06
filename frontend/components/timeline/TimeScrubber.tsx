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
 *
 * Visual refinement only (frontend visual/UX phase): the underlying native
 * `<input type="range">` and its `onChange`/keyboard/drag behaviour are
 * UNCHANGED -- every accessible name a test asserts on (`Storm timeline`,
 * `Previous timestamp`, `Next timestamp`, `Play`/`Pause`) is byte-identical
 * to before. Only presentation (a custom track visualization layered
 * behind the still-functional, still-focusable native input, plus a hover
 * preview) and interaction polish were added.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const trackRef = useRef<HTMLDivElement>(null);

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

  const percentFor = useCallback(
    (i: number) => (timestamps.length <= 1 ? 0 : (i / (timestamps.length - 1)) * 100),
    [timestamps.length],
  );

  const originPercents = useMemo(
    () => timestamps.map((ts, i) => ({ i, ts, pct: percentFor(i), isOrigin: originTimestamps.has(ts) })),
    [timestamps, originTimestamps, percentFor],
  );

  const handleTrackHover = (e: React.MouseEvent<HTMLDivElement>) => {
    const track = trackRef.current;
    if (!track || timestamps.length === 0) return;
    const rect = track.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    setHoverIndex(Math.round(ratio * (timestamps.length - 1)));
  };

  if (timestamps.length === 0) {
    return null;
  }

  const current = timestamps[index];
  const currentPct = percentFor(index);

  return (
    <div className="rounded-xl border border-border-subtle bg-white/[0.03] p-4 sm:p-5">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-wide text-text-muted">
            Observed timestamp
          </div>
          <div className="font-mono text-sm font-medium tabular-nums text-text-primary">
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
            active={playing}
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

      <div
        ref={trackRef}
        className="relative mt-6 h-8"
        onMouseMove={handleTrackHover}
        onMouseLeave={() => setHoverIndex(null)}
      >
        {/* Decorative track visualization -- the real, functional control is
            the native <input type="range"> below; this layer only draws it. */}
        <div aria-hidden className="absolute inset-x-0 top-1/2 h-1.5 -translate-y-1/2 overflow-hidden rounded-full bg-white/[0.06]">
          <div
            className="h-full bg-gradient-to-r from-accent-dim to-accent transition-[width] duration-150"
            style={{ width: `${currentPct}%` }}
          />
        </div>

        {/* Forecast-origin ticks -- purely informational, never interactive
            on their own; larger and amber-glowing so a real forecast origin
            reads immediately against the plain observed-only track. */}
        <div aria-hidden className="pointer-events-none absolute inset-x-0 top-1/2 -translate-y-1/2">
          {originPercents.filter((p) => p.isOrigin).map((p) => (
            <span
              key={p.ts}
              className="absolute top-1/2 h-2.5 w-1 -translate-x-1/2 -translate-y-1/2 rounded-full bg-predicted shadow-[0_0_6px_1px_rgba(255,176,32,0.55)]"
              style={{ left: `${p.pct}%` }}
            />
          ))}
        </div>

        {/* Current-position marker -- a third, distinct treatment (accent
            ring), matching the map's own current-scrub-position styling. */}
        <div
          aria-hidden
          className="pointer-events-none absolute top-1/2 h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-accent-soft bg-bg-elevated shadow-glow transition-[left] duration-150"
          style={{ left: `${currentPct}%` }}
        />

        {/* Hover preview -- purely decorative UI affordance, never changes
            `index`/`onChange` state on its own (only a real drag/click via
            the native input below does that). */}
        {hoverIndex != null && hoverIndex !== index && (
          <div
            aria-hidden
            className="pointer-events-none absolute -top-8 -translate-x-1/2 whitespace-nowrap rounded-md border border-border-subtle bg-bg-overlay px-2 py-1 font-mono text-[10px] text-text-secondary shadow-elevated"
            style={{ left: `${percentFor(hoverIndex)}%` }}
          >
            {formatTimestamp(timestamps[hoverIndex])}
          </div>
        )}

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
          className="absolute inset-x-0 top-1/2 h-8 w-full -translate-y-1/2 cursor-pointer appearance-none bg-transparent
            [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:appearance-none
            [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-transparent
            [&::-webkit-slider-thumb]:opacity-0
            [&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:rounded-full
            [&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:opacity-0
            focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-accent"
        />
      </div>

      <div className="mt-2 flex items-center justify-between font-mono text-[10px] text-text-muted">
        <span>{formatTimestamp(timestamps[0])}</span>
        <span>{formatTimestamp(timestamps[timestamps.length - 1])}</span>
      </div>
      <p className="mt-2 text-[11px] text-text-muted">
        Amber ticks mark timestamps with an issued model forecast. Drag, use arrow keys, or press
        play to move through the storm&apos;s real observation history.
      </p>
    </div>
  );
}

function ScrubButton({
  children, label, onClick, disabled, active,
}: {
  children: React.ReactNode;
  label: string;
  onClick: () => void;
  disabled?: boolean;
  active?: boolean;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      disabled={disabled}
      onClick={onClick}
      className={`flex h-8 w-8 items-center justify-center rounded-md border transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:opacity-30 ${
        active
          ? "border-accent-soft/50 bg-accent/15 text-accent-soft"
          : "border-border-subtle text-text-secondary hover:bg-white/5 hover:text-text-primary"
      }`}
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
