import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import IntensityChart from "./IntensityChart";

// Shaped exactly like the real backend response (ObservationOut/PredictionOut).
const observations = [
  { ts: "2010-06-26T00:00:00Z", lat: 16.9, lon: -86.1, wind_kt: 40, pressure_hpa: 1005, category: -1, nature: null, storm_speed_kt: null, storm_dir_deg: null, dist2land_km: null, is_synoptic: true, is_observed: true, data_kind: "observed" as const },
  { ts: "2010-06-26T06:00:00Z", lat: 17.2, lon: -87.2, wind_kt: 45, pressure_hpa: 1000, category: -1, nature: null, storm_speed_kt: null, storm_dir_deg: null, dist2land_km: null, is_synoptic: true, is_observed: true, data_kind: "observed" as const },
];

const predictions = [
  {
    task: "intensity" as const, origin_ts: "2010-06-26T00:00:00Z", lead_hours: 6,
    valid_ts: "2010-06-26T06:00:00Z", model_name: "intensity_lightgbm", model_version: "v1",
    pred_lat: null, pred_lon: null, pred_wind_kt: 44, pred_pressure_hpa: null,
    error_radius_km: null, true_lat: null, true_lon: null, true_wind_kt: 45,
    track_error_km: null, wind_error_kt: -1, data_kind: "model_prediction" as const,
    disclaimer: "Historical baseline model prediction, not an operational forecast.",
  },
];

describe("IntensityChart", () => {
  it("renders without throwing given real API-shaped observation and prediction fixtures", () => {
    expect(() =>
      render(<IntensityChart observations={observations} predictions={predictions} originTs="2010-06-26T00:00:00Z" />),
    ).not.toThrow();
  });

  it("renders nothing when there is no observed wind data at all", () => {
    const { container } = render(
      <IntensityChart
        observations={[{ ...observations[0], wind_kt: null }]}
        predictions={[]}
        originTs={null}
      />,
    );
    expect(container).toBeEmptyDOMElement();
  });
});
