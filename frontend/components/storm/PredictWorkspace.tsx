"use client";

/**
 * The flagship /predict/[sid] workspace: TimeScrubber -> map / intensity
 * chart / satellite / classification / model selector all read from the
 * SAME selected real observation timestamp. Initial data (storm,
 * observations, full prediction series, model list, and the track GeoJSON
 * for the latest origin) is fetched server-side by `app/predict/[sid]/
 * page.tsx` and passed in as props; only the per-origin track refetch (when
 * the user scrubs to a different real forecast origin) happens client-side,
 * via the same typed `getStormTrack` function the Phase 3 page always used.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import ClassificationPanel from "@/components/classification/ClassificationPanel";
import IntensityChart from "@/components/charts/IntensityChart";
import GeminiPanel from "@/components/gemini/GeminiPanel";
import TrackLegend from "@/components/maps/TrackLegend";
import SatelliteViewer from "@/components/satellite/SatelliteViewer";
import ModelSelector, { type ModelOption } from "@/components/storm/ModelSelector";
import StormHeader from "@/components/panels/StormHeader";
import TimeScrubber from "@/components/timeline/TimeScrubber";
import GlassPanel from "@/components/ui/GlassPanel";
import { ChartSkeleton, MapSkeleton } from "@/components/ui/Skeletons";
import SectionHeader from "@/components/ui/SectionHeader";
import { GaugeIcon, MapPinIcon, SatelliteIcon } from "@/components/ui/Icons";
import { formatTimestamp } from "@/lib/format";
import {
  getStormTrack, type CycloneDetail, type ObservationList, type PredictionList,
  type TrackFeatureCollection,
} from "@/lib/api";

const CycloneMapClient = dynamic(() => import("@/components/map/CycloneMapClient"), {
  ssr: false,
  loading: () => <MapSkeleton className="h-full w-full" />,
});

function uniqueSorted(values: string[]): string[] {
  return Array.from(new Set(values)).sort();
}

export default function PredictWorkspace({
  storm, observations, predictionSeries, initialTrack,
}: {
  storm: CycloneDetail;
  observations: ObservationList;
  predictionSeries: PredictionList;
  initialTrack: TrackFeatureCollection;
}) {
  const timestamps = useMemo(() => observations.map((o) => o.ts), [observations]);
  const originTimestamps = useMemo(
    () => new Set(predictionSeries.map((p) => p.origin_ts)),
    [predictionSeries],
  );

  const initialIndex = useMemo(() => {
    const lastOriginIdx = timestamps.reduce(
      (best, ts, i) => (originTimestamps.has(ts) ? i : best), -1,
    );
    return lastOriginIdx >= 0 ? lastOriginIdx : Math.max(0, timestamps.length - 1);
  }, [timestamps, originTimestamps]);

  const [index, setIndex] = useState(initialIndex);
  const currentTs = timestamps[index];
  const currentObservation = observations[index];
  const isOrigin = originTimestamps.has(currentTs);

  const [track, setTrack] = useState<TrackFeatureCollection>(initialTrack);
  const [trackLoading, setTrackLoading] = useState(false);

  // Phase 12 performance audit (measured via Lighthouse + a network-request
  // trace on /predict/[sid]): the server component above already fetches
  // the track for the initial origin (`initialTrack`) and hands it to us as
  // a prop. Without this guard, this effect's very first run -- which
  // always corresponds to that SAME initial origin, since `index` is never
  // changed before mount -- fired an identical, wholly redundant
  // `getStormTrack` call on every single page load, before the user ever
  // touched the TimeScrubber. Skipping exactly the first run removes that
  // duplicate request while leaving every real, user-driven scrub-to-a-
  // different-origin refetch untouched.
  const isFirstTrackEffectRun = useRef(true);
  useEffect(() => {
    if (!isOrigin) return;
    if (isFirstTrackEffectRun.current) {
      isFirstTrackEffectRun.current = false;
      return;
    }
    let cancelled = false;
    setTrackLoading(true);
    getStormTrack(storm.sid, currentTs)
      .then((res) => {
        if (!cancelled) setTrack(res);
      })
      .finally(() => {
        if (!cancelled) setTrackLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [currentTs, isOrigin, storm.sid]);

  const predictionsAtOrigin = useMemo(
    () => (isOrigin ? predictionSeries.filter((p) => p.origin_ts === currentTs) : []),
    [predictionSeries, currentTs, isOrigin],
  );

  const intensityModelOptions: ModelOption[] = useMemo(() => {
    const names = uniqueSorted(
      predictionSeries.filter((p) => p.task === "intensity").map((p) => p.model_name),
    );
    return names.map((name) => ({
      name, version: predictionSeries.find((p) => p.model_name === name)?.model_version ?? "v1",
    }));
  }, [predictionSeries]);

  const trackModelOptions: ModelOption[] = useMemo(() => {
    const names = uniqueSorted(
      predictionSeries.filter((p) => p.task === "track").map((p) => p.model_name),
    );
    return names.map((name) => ({
      name, version: predictionSeries.find((p) => p.model_name === name)?.model_version ?? "v1",
    }));
  }, [predictionSeries]);

  const [intensityModel, setIntensityModel] = useState<string | null>(
    intensityModelOptions[0]?.name ?? null,
  );
  const [trackModel, setTrackModel] = useState<string | null>(trackModelOptions[0]?.name ?? null);

  const maxForecastHorizonH = useMemo(
    () => predictionSeries.reduce((max, p) => Math.max(max, p.lead_hours), 0) || null,
    [predictionSeries],
  );

  const intensityForChart = intensityModel
    ? predictionSeries.filter((p) => p.task === "intensity" && p.model_name === intensityModel)
    : [];

  return (
    <div className="mx-auto max-w-7xl px-6 py-8">
      <StormHeader storm={storm} maxForecastHorizonH={maxForecastHorizonH} />

      <div className="mt-6">
        <TimeScrubber
          timestamps={timestamps}
          originTimestamps={originTimestamps}
          index={index}
          onChange={setIndex}
        />
      </div>

      <section className="mt-10">
        <SectionHeader
          eyebrow="Geospatial track"
          title="Observed & predicted trajectory"
          description="The map is the source of truth for position -- observed and predicted paths never share a colour or line style."
        />
        <div className="mt-4 grid grid-cols-1 gap-6 lg:grid-cols-3">
          <div className="relative h-[480px] overflow-hidden rounded-xl border border-border-subtle lg:col-span-2">
            <CycloneMapClient
              track={track}
              currentPosition={currentObservation ? { lat: currentObservation.lat, lon: currentObservation.lon } : null}
              selectedModelName={trackModel}
            />
            {currentObservation && (
              <div className="pointer-events-none absolute left-3 top-3 z-[400] rounded-lg border border-border-subtle bg-bg-base/85 px-3 py-2 font-mono text-[11px] text-text-secondary shadow-elevated backdrop-blur-sm">
                <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-text-muted">
                  <MapPinIcon width={11} height={11} /> Scrub position
                </div>
                <div className="mt-1 tabular-nums">{formatTimestamp(currentTs)}</div>
                {currentObservation.wind_kt != null && (
                  <div className="tabular-nums">{currentObservation.wind_kt.toFixed(0)} kt</div>
                )}
              </div>
            )}
            {trackLoading && (
              <div className="pointer-events-none absolute right-3 top-3 z-[400] flex items-center gap-1.5 rounded-md bg-black/60 px-2 py-1 text-[11px] text-text-secondary">
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent-soft" />
                Updating forecast origin…
              </div>
            )}
          </div>
          <div className="flex flex-col gap-4">
            <TrackLegend />
            <GlassPanel className="p-4">
              <ModelSelector
                label="Track model"
                options={trackModelOptions}
                value={trackModel}
                onChange={setTrackModel}
              />
            </GlassPanel>
          </div>
        </div>
      </section>

      <section className="mt-10">
        <SectionHeader
          eyebrow="Intensity"
          title="Observed vs. predicted wind"
          description={isOrigin
            ? "This forecast origin has an issued model prediction."
            : "No model prediction was issued at this exact timestamp -- showing observed history only."}
        />
        <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
          <ModelSelector
            label="Intensity model"
            options={intensityModelOptions}
            value={intensityModel}
            onChange={setIntensityModel}
          />
          <div className="flex items-center gap-3 text-[11px] text-text-muted">
            <span className="flex items-center gap-1.5">
              <span aria-hidden className="h-0.5 w-4 rounded-full bg-truth" /> Observed
            </span>
            <span className="flex items-center gap-1.5">
              <span aria-hidden className="h-0.5 w-4 rounded-full border-t-2 border-dashed border-predicted" /> Model prediction
            </span>
          </div>
        </div>
        <GlassPanel className="mt-4 p-4">
          {observations.length > 0 ? (
            <IntensityChart
              observations={observations}
              predictions={intensityForChart}
              originTs={isOrigin ? currentTs : null}
            />
          ) : (
            <ChartSkeleton />
          )}
        </GlassPanel>
        {isOrigin && intensityModel && (
          <IntensityHorizonTable predictions={predictionsAtOrigin} model={intensityModel} />
        )}
      </section>

      <section className="mt-10 grid grid-cols-1 gap-6 lg:grid-cols-2">
        <GlassPanel className="p-5">
          <div className="flex items-center gap-2">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg border border-border-subtle bg-white/[0.04] text-accent-soft">
              <SatelliteIcon width={15} height={15} />
            </span>
            <div>
              <h3 className="text-sm font-semibold text-text-primary">Satellite Intelligence</h3>
              <p className="text-xs text-text-muted">HURSAT-B1 imagery and Dvorak scene labels.</p>
            </div>
          </div>
          <div className="mt-4">
            <SatelliteViewer timestamp={currentTs} />
          </div>
        </GlassPanel>
        <GlassPanel className="p-5">
          <div className="flex items-center gap-2">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg border border-border-subtle bg-white/[0.04] text-accent-soft">
              <GaugeIcon width={15} height={15} />
            </span>
            <div>
              <h3 className="text-sm font-semibold text-text-primary">Classification</h3>
              <p className="text-xs text-text-muted">Cyclone pattern / scene classification.</p>
            </div>
          </div>
          <div className="mt-4">
            <ClassificationPanel />
          </div>
        </GlassPanel>
      </section>

      <section className="mt-10">
        <GeminiPanel sid={storm.sid} />
      </section>
    </div>
  );
}

function IntensityHorizonTable({
  predictions, model,
}: {
  predictions: PredictionList;
  model: string;
}) {
  const rows = predictions
    .filter((p) => p.task === "intensity" && p.model_name === model)
    .sort((a, b) => a.lead_hours - b.lead_hours);
  if (rows.length === 0) return null;
  return (
    <div className="mt-4 overflow-x-auto rounded-lg border border-border-subtle">
      <table className="w-full min-w-[420px] text-left text-sm">
        <thead>
          <tr className="text-xs uppercase text-text-muted">
            <th className="px-3 py-2 font-mono">Horizon</th>
            <th className="px-3 py-2 text-right text-predicted">Predicted (kt)</th>
            <th className="px-3 py-2 text-right text-truth">Observed (kt)</th>
            <th className="px-3 py-2 text-right">Error (kt)</th>
          </tr>
        </thead>
        <tbody className="font-mono tabular-nums">
          {rows.map((p, i) => (
            <tr key={i} className="border-t border-border-subtle/60 transition-colors hover:bg-white/[0.02]">
              <td className="px-3 py-2">+{p.lead_hours}h</td>
              <td className="px-3 py-2 text-right text-predicted">
                {p.pred_wind_kt != null ? p.pred_wind_kt.toFixed(1) : "—"}
              </td>
              <td className="px-3 py-2 text-right text-truth">
                {p.true_wind_kt != null ? p.true_wind_kt.toFixed(1) : "—"}
              </td>
              <td className="px-3 py-2 text-right text-text-secondary">
                {p.wind_error_kt != null ? p.wind_error_kt.toFixed(1) : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
