import { afterEach, describe, expect, it, vi } from "vitest";
import {
  ApiError, explainForecast, getModelPerformance, getStorm, getStormPredictionSeries, listStorms,
} from "./api";

describe("API client", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("listStorms resolves with the parsed JSON body on success", async () => {
    const fakeResponse = { items: [{ sid: "2010176N16278" }], total: 1, limit: 50, offset: 0 };
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => fakeResponse,
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await listStorms();
    expect(result).toEqual(fakeResponse);
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/api/v1/cyclones"),
      expect.objectContaining({ cache: "no-store" }),
    );
  });

  it("listStorms appends query parameters correctly", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ items: [], total: 0, limit: 10, offset: 5 }),
    });
    vi.stubGlobal("fetch", fetchMock);

    await listStorms({ season: 2015, limit: 10, offset: 5 });
    const calledUrl = fetchMock.mock.calls[0][0] as string;
    expect(calledUrl).toContain("season=2015");
    expect(calledUrl).toContain("limit=10");
    expect(calledUrl).toContain("offset=5");
  });

  it("throws ApiError with the status code on a non-ok response", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 404,
      json: async () => ({ detail: "Storm 'X' not found", title: "Not Found" }),
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(getStorm("X")).rejects.toBeInstanceOf(ApiError);
    await expect(getStorm("X")).rejects.toMatchObject({ status: 404 });
  });

  it("ApiError message prefers the RFC 7807 detail field", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      json: async () => ({ detail: "boom", title: "Server Error" }),
    });
    vi.stubGlobal("fetch", fetchMock);

    try {
      await getStorm("X");
      expect.fail("should have thrown");
    } catch (err) {
      expect(err).toBeInstanceOf(ApiError);
      expect((err as ApiError).message).toBe("boom");
    }
  });

  it("URL-encodes the storm id", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
    vi.stubGlobal("fetch", fetchMock);

    await getStorm("2010 176N16278");
    const calledUrl = fetchMock.mock.calls[0][0] as string;
    expect(calledUrl).toContain("2010%20176N16278");
  });

  it("getStormPredictionSeries parses a real-shaped response, including nullable geometry fields", async () => {
    // Shaped exactly like a real intensity-task row: pred_lat/pred_lon are
    // null (intensity models never predict position) -- task §18's
    // explicit "pay special attention to nullable fields".
    const realShapedRow = {
      task: "intensity", origin_ts: "2010-06-26T12:00:00Z", lead_hours: 24,
      valid_ts: "2010-06-27T12:00:00Z", model_name: "intensity_lightgbm", model_version: "v1",
      pred_lat: null, pred_lon: null, pred_wind_kt: 92.4, pred_pressure_hpa: null,
      error_radius_km: null, true_lat: null, true_lon: null, true_wind_kt: 90.0,
      track_error_km: null, wind_error_kt: 2.4, data_kind: "model_prediction",
      disclaimer: "Historical baseline model prediction, not an operational forecast.",
    };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => [realShapedRow] });
    vi.stubGlobal("fetch", fetchMock);

    const result = await getStormPredictionSeries("2010176N16278");
    expect(result).toEqual([realShapedRow]);
    expect(fetchMock.mock.calls[0][0]).toContain("/prediction/2010176N16278/series");
  });

  it("explainForecast sends only sid and model-version fields -- never arbitrary text", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ source: "fallback" }) });
    vi.stubGlobal("fetch", fetchMock);

    await explainForecast("2010176N16278", { intensityModelVersion: "v1" });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toContain("/api/v1/explain/forecast");
    expect((init as RequestInit).method).toBe("POST");
    const body = JSON.parse((init as RequestInit).body as string);
    expect(body).toEqual({ sid: "2010176N16278", intensity_model_version: "v1", track_model_version: null });
  });

  it("getModelPerformance resolves the parsed model-performance JSON", async () => {
    const fakeResponse = { intensity: { task: "intensity", models: [] } };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => fakeResponse });
    vi.stubGlobal("fetch", fetchMock);

    const result = await getModelPerformance();
    expect(result).toEqual(fakeResponse);
    expect(fetchMock.mock.calls[0][0]).toContain("/api/v1/analytics/model-performance");
  });
});
