"use client";

/**
 * The storm track map. Renders a GeoJSON FeatureCollection already
 * assembled server-side (backend/app/services/geometry.py) -- this
 * component performs NO geodesic maths and NO reprojection, per
 * docs/SYSTEM_ARCHITECTURE.md §6.1 ("the frontend receives GeoJSON that is
 * already correct").
 *
 * MANDATORY Phase 3 rule (enforced here, not just by convention): observed
 * and predicted tracks are never given the same visual treatment. Observed
 * is a solid teal line; predicted is a dashed amber line -- distinguishable
 * by colour AND by line style, so the difference survives greyscale
 * (docs/UI_UX_ARCHITECTURE.md §8 accessibility rule).
 */
import { useMemo } from "react";
import {
  CircleMarker, MapContainer, Polyline, Popup, TileLayer, useMap,
} from "react-leaflet";
import type { TrackFeatureCollection } from "@/lib/api";
import { formatTimestamp } from "@/lib/format";

type LonLat = [number, number];

function FitBounds({ points }: { points: LonLat[] }) {
  const map = useMap();
  useMemo(() => {
    if (points.length === 0) return;
    const latLngs = points.map(([lon, lat]) => [lat, lon] as [number, number]);
    map.fitBounds(latLngs, { padding: [32, 32] });
  }, [points, map]);
  return null;
}

export default function CycloneMap({ track }: { track: TrackFeatureCollection }) {
  const observedTracks = track.features.filter(
    (f) => f.properties?.kind === "observed_track" && f.geometry.type === "LineString",
  );
  const observedPoints = track.features.filter((f) => f.properties?.kind === "observed_point");
  const predictedTracks = track.features.filter(
    (f) => f.properties?.kind === "predicted_track" && f.geometry.type === "LineString",
  );
  const predictedPoints = track.features.filter((f) => f.properties?.kind === "predicted_point");

  const allPoints: LonLat[] = track.features
    .filter((f) => f.geometry.type === "Point")
    .map((f) => f.geometry.coordinates as LonLat);

  const predictedColors = ["#FFB020", "#FF7A45", "#C77DFF"]; // one per model, still != truth colour

  return (
    <MapContainer
      center={[20, -60]}
      zoom={4}
      scrollWheelZoom
      className="h-full w-full rounded-lg"
      style={{ background: "#0B0F17" }}
    >
      <TileLayer
        attribution='&copy; <a href="https://carto.com/attributions">CARTO</a>'
        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
      />
      <FitBounds points={allPoints} />

      {/* OBSERVED -- solid teal line, per UI_UX_ARCHITECTURE.md state colours */}
      {observedTracks.map((f, i) => {
        const coords = (f.geometry.coordinates as LonLat[]).map(
          ([lon, lat]) => [lat, lon] as [number, number],
        );
        return (
          <Polyline
            key={`obs-track-${i}`}
            positions={coords}
            pathOptions={{ color: "#22D3A7", weight: 3, opacity: 0.9 }}
          />
        );
      })}
      {observedPoints.map((f, i) => {
        const [lon, lat] = f.geometry.coordinates as LonLat;
        const p = f.properties ?? {};
        return (
          <CircleMarker
            key={`obs-pt-${i}`}
            center={[lat, lon]}
            radius={4}
            pathOptions={{ color: "#22D3A7", fillColor: "#22D3A7", fillOpacity: 0.9 }}
          >
            <Popup>
              <div className="text-xs">
                <div className="font-semibold">OBSERVED</div>
                <div>{formatTimestamp(p.ts as string)}</div>
                {p.wind_kt != null && <div>Wind: {p.wind_kt as number} kt</div>}
                {p.pressure_hpa != null && <div>Pressure: {p.pressure_hpa as number} hPa</div>}
              </div>
            </Popup>
          </CircleMarker>
        );
      })}

      {/* PREDICTED -- dashed line, distinct colour per model, never teal */}
      {predictedTracks.map((f, i) => {
        const coords = (f.geometry.coordinates as LonLat[]).map(
          ([lon, lat]) => [lat, lon] as [number, number],
        );
        const color = predictedColors[i % predictedColors.length];
        return (
          <Polyline
            key={`pred-track-${i}`}
            positions={coords}
            pathOptions={{ color, weight: 2.5, opacity: 0.85, dashArray: "6 6" }}
          />
        );
      })}
      {predictedPoints.map((f, i) => {
        const [lon, lat] = f.geometry.coordinates as LonLat;
        const p = f.properties ?? {};
        const modelIdx = predictedTracks.findIndex(
          (t) => t.properties?.model_name === p.model_name,
        );
        const color = predictedColors[Math.max(0, modelIdx) % predictedColors.length];
        return (
          <CircleMarker
            key={`pred-pt-${i}`}
            center={[lat, lon]}
            radius={5}
            pathOptions={{ color, fillColor: color, fillOpacity: 0.5, weight: 2 }}
          >
            <Popup>
              <div className="text-xs">
                <div className="font-semibold">MODEL PREDICTION</div>
                <div>
                  {p.model_name as string} {p.model_version as string} · +{p.lead_hours as number}h
                </div>
                {p.pred_wind_kt != null && <div>Predicted wind: {(p.pred_wind_kt as number).toFixed(0)} kt</div>}
                {p.track_error_km != null && (
                  <div>Track error: {(p.track_error_km as number).toFixed(0)} km</div>
                )}
                <div className="mt-1 italic text-[10px] text-gray-500">{p.disclaimer as string}</div>
              </div>
            </Popup>
          </CircleMarker>
        );
      })}
    </MapContainer>
  );
}
