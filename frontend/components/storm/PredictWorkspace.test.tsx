import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

// Phase 12 performance audit: the /predict/[sid] server component already
// fetches the track for the initial forecast origin (`initialTrack` prop);
// PredictWorkspace must not silently re-fetch that SAME data client-side on
// mount. Mocking `getStormTrack` here lets the test assert call COUNT and
// ARGUMENTS precisely, which a live network/DB-backed test could not do
// deterministically.
const getStormTrackMock = vi.fn().mockResolvedValue({ type: "FeatureCollection", features: [] });
vi.mock("@/lib/api", () => ({
  getStormTrack: (...args: unknown[]) => getStormTrackMock(...args),
}));

// The map/satellite/classification/Gemini panels are unrelated to the
// fetch-avoidance logic under test and pull in Leaflet/Recharts/network
// calls of their own -- stubbed out to keep this test isolated and fast,
// matching this project's established pattern of testing one component's
// own behaviour, not its entire child tree (see IntensityChart.test.tsx).
vi.mock("@/components/map/CycloneMapClient", () => ({
  default: () => <div data-testid="mock-map" />,
}));
vi.mock("@/components/satellite/SatelliteViewer", () => ({
  default: () => <div data-testid="mock-satellite" />,
}));
vi.mock("@/components/classification/ClassificationPanel", () => ({
  default: () => <div data-testid="mock-classification" />,
}));
vi.mock("@/components/gemini/GeminiPanel", () => ({
  default: () => <div data-testid="mock-gemini" />,
}));

import PredictWorkspace from "./PredictWorkspace";

const TS = [
  "2010-06-26T00:00:00Z", // origin #1
  "2010-06-26T06:00:00Z", // not an origin
  "2010-06-26T12:00:00Z", // origin #2 (latest -> the initial index)
];

const storm = {
  sid: "2010176N16278", name: null, season: 2010, basin: "NA", subbasin: null,
  start_time: TS[0], end_time: TS[2], n_observations: 3, max_wind_kt: 55.0,
  min_pressure_hpa: 990.0, max_category: 0, made_landfall: null, split: "test" as const,
  has_predictions: true, bbox: [-88.2, 16.9, -86.1, 17.5] as [number, number, number, number],
};

const observations = TS.map((ts, i) => ({
  ts, lat: 16.9 + i, lon: -88.2 + i, wind_kt: 40 + i, pressure_hpa: 1005 - i, category: -1 as const,
  nature: null, storm_speed_kt: null, storm_dir_deg: null, dist2land_km: null,
  is_synoptic: true, is_observed: true, data_kind: "observed" as const,
}));

function predictionRow(originTs: string) {
  return {
    task: "track" as const, origin_ts: originTs, lead_hours: 6,
    valid_ts: TS[2], model_name: "track_cliper", model_version: "v1",
    pred_lat: 17.0, pred_lon: -87.0, pred_wind_kt: null, pred_pressure_hpa: null,
    error_radius_km: 30.0, true_lat: null, true_lon: null, true_wind_kt: null,
    track_error_km: null, wind_error_kt: null, data_kind: "model_prediction" as const,
    disclaimer: "Historical baseline model prediction, not an operational forecast.",
  };
}

const predictionSeries = [predictionRow(TS[0]), predictionRow(TS[2])];

const initialTrack = { type: "FeatureCollection" as const, features: [] };

describe("PredictWorkspace track-fetch behaviour", () => {
  it("does not re-fetch the track on initial mount (initialTrack prop already covers it)", () => {
    render(
      <PredictWorkspace
        storm={storm} observations={observations} predictionSeries={predictionSeries}
        initialTrack={initialTrack}
      />,
    );
    expect(getStormTrackMock).not.toHaveBeenCalled();
  });

  it("does not fetch when scrubbing to a non-origin timestamp", () => {
    getStormTrackMock.mockClear();
    render(
      <PredictWorkspace
        storm={storm} observations={observations} predictionSeries={predictionSeries}
        initialTrack={initialTrack}
      />,
    );
    // Initial index is TS[2] (the latest origin) -- one Previous click moves
    // to TS[1], which is NOT a forecast origin.
    fireEvent.click(screen.getByRole("button", { name: /previous timestamp/i }));
    expect(getStormTrackMock).not.toHaveBeenCalled();
  });

  it("fetches the track when the user scrubs to a DIFFERENT real forecast origin", () => {
    getStormTrackMock.mockClear();
    render(
      <PredictWorkspace
        storm={storm} observations={observations} predictionSeries={predictionSeries}
        initialTrack={initialTrack}
      />,
    );
    const prevButton = screen.getByRole("button", { name: /previous timestamp/i });
    fireEvent.click(prevButton); // TS[2] -> TS[1] (not an origin, no fetch)
    fireEvent.click(prevButton); // TS[1] -> TS[0] (an origin -- must fetch)
    expect(getStormTrackMock).toHaveBeenCalledTimes(1);
    expect(getStormTrackMock).toHaveBeenCalledWith(storm.sid, TS[0]);
  });
});
