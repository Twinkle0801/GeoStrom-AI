/**
 * Typed API client. Every function's return type comes from `api-types.ts`,
 * which is GENERATED from contracts/openapi.json (never hand-edited --
 * see the "gen:types" script and docs/API_ARCHITECTURE.md §5). A backend
 * field rename becomes a TypeScript error here at build time.
 */
import { API_BASE_URL } from "./config";
import type { paths } from "./api-types";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, { cache: "no-store" });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new ApiError(res.status, body.detail ?? body.title ?? res.statusText);
  }
  return res.json() as Promise<T>;
}

async function post<TBody, TResponse>(path: string, body: TBody): Promise<TResponse> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
  });
  if (!res.ok) {
    const errBody = await res.json().catch(() => ({}));
    throw new ApiError(res.status, errBody.detail ?? errBody.title ?? res.statusText);
  }
  return res.json() as Promise<TResponse>;
}

type CyclonesList =
  paths["/api/v1/cyclones"]["get"]["responses"][200]["content"]["application/json"];
type CycloneDetail =
  paths["/api/v1/cyclones/{sid}"]["get"]["responses"][200]["content"]["application/json"];
type ObservationList =
  paths["/api/v1/cyclones/{sid}/observations"]["get"]["responses"][200]["content"]["application/json"];
type TrackFeatureCollection =
  paths["/api/v1/tracks/{sid}"]["get"]["responses"][200]["content"]["application/json"];
type PredictionList =
  paths["/api/v1/prediction/{sid}"]["get"]["responses"][200]["content"]["application/json"];
type ModelVersionList =
  paths["/api/v1/prediction/models/list"]["get"]["responses"][200]["content"]["application/json"];
type MetaResponse = paths["/api/v1/meta"]["get"]["responses"][200]["content"]["application/json"];
type ExplainRequest =
  paths["/api/v1/explain/forecast"]["post"]["requestBody"]["content"]["application/json"];
type ExplainResponse =
  paths["/api/v1/explain/forecast"]["post"]["responses"][200]["content"]["application/json"];
type ModelPerformanceResponse =
  paths["/api/v1/analytics/model-performance"]["get"]["responses"][200]["content"]["application/json"];

export function listStorms(params?: {
  season?: number;
  split?: string;
  q?: string;
  limit?: number;
  offset?: number;
}): Promise<CyclonesList> {
  const q = new URLSearchParams();
  if (params?.season) q.set("season", String(params.season));
  if (params?.split) q.set("split", params.split);
  if (params?.q) q.set("q", params.q);
  if (params?.limit) q.set("limit", String(params.limit));
  if (params?.offset) q.set("offset", String(params.offset));
  const qs = q.toString();
  return get<CyclonesList>(`/api/v1/cyclones${qs ? `?${qs}` : ""}`);
}

export function getStorm(sid: string): Promise<CycloneDetail> {
  return get<CycloneDetail>(`/api/v1/cyclones/${encodeURIComponent(sid)}`);
}

export function getStormObservations(sid: string): Promise<ObservationList> {
  return get<ObservationList>(`/api/v1/cyclones/${encodeURIComponent(sid)}/observations`);
}

export function getStormTrack(sid: string, originTs?: string): Promise<TrackFeatureCollection> {
  const qs = originTs ? `?t=${encodeURIComponent(originTs)}` : "";
  return get<TrackFeatureCollection>(`/api/v1/tracks/${encodeURIComponent(sid)}${qs}`);
}

export function getStormPrediction(
  sid: string,
  opts?: { originTs?: string; task?: string; model?: string },
): Promise<PredictionList> {
  const q = new URLSearchParams();
  if (opts?.originTs) q.set("t", opts.originTs);
  if (opts?.task) q.set("task", opts.task);
  if (opts?.model) q.set("model", opts.model);
  const qs = q.toString();
  return get<PredictionList>(`/api/v1/prediction/${encodeURIComponent(sid)}${qs ? `?${qs}` : ""}`);
}

/** Every forecast ever issued across the storm's life (all origins), used
 * by the TimeScrubber to know which real timestamps have a real forecast. */
export function getStormPredictionSeries(
  sid: string,
  opts?: { task?: string; model?: string },
): Promise<PredictionList> {
  const q = new URLSearchParams();
  if (opts?.task) q.set("task", opts.task);
  if (opts?.model) q.set("model", opts.model);
  const qs = q.toString();
  return get<PredictionList>(`/api/v1/prediction/${encodeURIComponent(sid)}/series${qs ? `?${qs}` : ""}`);
}

export function listModels(task?: string): Promise<ModelVersionList> {
  const qs = task ? `?task=${encodeURIComponent(task)}` : "";
  return get<ModelVersionList>(`/api/v1/prediction/models/list${qs}`);
}

export function getMeta(): Promise<MetaResponse> {
  return get<MetaResponse>("/api/v1/meta");
}

/** Calls the backend's Gemini explanation endpoint ONLY -- the frontend
 * never talks to Gemini directly (docs/API_ARCHITECTURE.md §6.2; Phase 9). */
export function explainForecast(
  sid: string,
  opts?: { intensityModelVersion?: string; trackModelVersion?: string },
): Promise<ExplainResponse> {
  return post<ExplainRequest, ExplainResponse>("/api/v1/explain/forecast", {
    sid,
    intensity_model_version: opts?.intensityModelVersion ?? null,
    track_model_version: opts?.trackModelVersion ?? null,
  });
}

export function getModelPerformance(): Promise<ModelPerformanceResponse> {
  return get<ModelPerformanceResponse>("/api/v1/analytics/model-performance");
}

export type {
  CyclonesList, CycloneDetail, ObservationList, TrackFeatureCollection,
  PredictionList, ModelVersionList, MetaResponse, ExplainRequest, ExplainResponse,
  ModelPerformanceResponse,
};
