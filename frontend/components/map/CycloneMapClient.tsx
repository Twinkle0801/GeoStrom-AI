"use client";

/**
 * Leaflet touches `window` at import time, so it must never be part of the
 * server-rendered bundle -- this wrapper is the one place `next/dynamic`
 * with `ssr:false` is used, isolating that constraint to a single file.
 */
import dynamic from "next/dynamic";
import type { TrackFeatureCollection } from "@/lib/api";

const CycloneMap = dynamic(() => import("./CycloneMap"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full w-full items-center justify-center rounded-lg bg-bg-elevated text-sm text-text-muted">
      Loading map…
    </div>
  ),
});

export default function CycloneMapClient({
  track, currentPosition, selectedModelName,
}: {
  track: TrackFeatureCollection;
  currentPosition?: { lat: number; lon: number } | null;
  selectedModelName?: string | null;
}) {
  return (
    <CycloneMap track={track} currentPosition={currentPosition} selectedModelName={selectedModelName} />
  );
}
