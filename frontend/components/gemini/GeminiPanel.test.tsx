import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import GeminiPanel from "./GeminiPanel";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

const baseExplanation = {
  summary: "GeoStrom AI evidence for TEST (2010, NA). The model predicts 92 kt at +24h.",
  intensity_explanation: "About 92 kt at +24h.",
  track_explanation: "Near 25.9, -88.9 at +24h.",
  classification_explanation: "No classification result is available.",
  limitations: "This is not an operational forecast.",
};

const baseEvidence = {
  evidence_schema_version: "v1",
  generated_at: "2010-06-26T12:00:00Z",
  storm: { sid: "2010176N16278", name: null, season: 2010, basin: "NA", n_observations: 5 },
  current_state: null,
  recent_history: [],
  intensity: null,
  track: null,
  classification: null,
  known_limitations: ["Retrospective research prototype."],
  forbidden_claims: [],
};

function mockFetchOnce(body: unknown, ok = true) {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({ ok, status: ok ? 200 : 500, json: async () => body }),
  );
}

describe("GeminiPanel", () => {
  it("starts idle with a Generate explanation button", () => {
    render(<GeminiPanel sid="2010176N16278" />);
    expect(screen.getByRole("button", { name: /generate explanation/i })).toBeInTheDocument();
  });

  it("shows the loading state immediately after clicking Generate", () => {
    vi.stubGlobal("fetch", vi.fn(() => new Promise(() => {})));
    render(<GeminiPanel sid="2010176N16278" />);
    fireEvent.click(screen.getByRole("button", { name: /generate explanation/i }));
    expect(screen.getByText(/analyzing stored evidence/i)).toBeInTheDocument();
  });

  it("shows the explanation and a grounded badge on a real Gemini response", async () => {
    mockFetchOnce({
      sid: "2010176N16278", generated_at: "2010-06-26T12:00:00Z", evidence_schema_version: "v1",
      intensity_model: null, track_model: null, classification_model: null,
      source: "gemini", fallback_reason: null, validation_violations: [],
      explanation: baseExplanation, evidence: baseEvidence,
      disclaimer: "Retrospective research-prototype model output.",
    });
    render(<GeminiPanel sid="2010176N16278" />);
    fireEvent.click(screen.getByRole("button", { name: /generate explanation/i }));
    await waitFor(() => expect(screen.getByText(baseExplanation.summary)).toBeInTheDocument());
    expect(screen.getByText(/grounded in stored model output/i)).toBeInTheDocument();
  });

  it("clearly identifies a fallback response as a deterministic evidence summary", async () => {
    mockFetchOnce({
      sid: "2010176N16278", generated_at: "2010-06-26T12:00:00Z", evidence_schema_version: "v1",
      intensity_model: null, track_model: null, classification_model: null,
      source: "fallback", fallback_reason: "not_configured", validation_violations: [],
      explanation: baseExplanation, evidence: baseEvidence,
      disclaimer: "Retrospective research-prototype model output.",
    });
    render(<GeminiPanel sid="2010176N16278" />);
    fireEvent.click(screen.getByRole("button", { name: /generate explanation/i }));
    await waitFor(() => expect(screen.getByText(baseExplanation.summary)).toBeInTheDocument());
    expect(screen.getByText(/deterministic evidence summary/i)).toBeInTheDocument();
  });

  it("shows an error state with a retry option when the request fails", async () => {
    mockFetchOnce({ detail: "Storm not found" }, false);
    render(<GeminiPanel sid="2010176N16278" />);
    fireEvent.click(screen.getByRole("button", { name: /generate explanation/i }));
    await waitFor(() =>
      expect(screen.getByText(/explanation unavailable/i)).toBeInTheDocument());
    expect(screen.getByRole("button", { name: /retry/i })).toBeInTheDocument();
  });

  it("opens the evidence drawer when 'View evidence' is clicked", async () => {
    mockFetchOnce({
      sid: "2010176N16278", generated_at: "2010-06-26T12:00:00Z", evidence_schema_version: "v1",
      intensity_model: null, track_model: null, classification_model: null,
      source: "gemini", fallback_reason: null, validation_violations: [],
      explanation: baseExplanation, evidence: baseEvidence,
      disclaimer: "Retrospective research-prototype model output.",
    });
    render(<GeminiPanel sid="2010176N16278" />);
    fireEvent.click(screen.getByRole("button", { name: /generate explanation/i }));
    await waitFor(() => expect(screen.getByText(baseExplanation.summary)).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: /view evidence/i }));
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getByText(/this explanation is grounded in stored project evidence/i)).toBeInTheDocument();
  });

  it("never sends free-form user text -- the request body only ever contains sid and model versions", async () => {
    mockFetchOnce({
      sid: "2010176N16278", generated_at: "2010-06-26T12:00:00Z", evidence_schema_version: "v1",
      intensity_model: null, track_model: null, classification_model: null,
      source: "gemini", fallback_reason: null, validation_violations: [],
      explanation: baseExplanation, evidence: baseEvidence,
      disclaimer: "x",
    });
    render(<GeminiPanel sid="2010176N16278" />);
    fireEvent.click(screen.getByRole("button", { name: /generate explanation/i }));
    await waitFor(() => expect(screen.getByText(baseExplanation.summary)).toBeInTheDocument());
    const fetchMock = vi.mocked(fetch);
    const [, init] = fetchMock.mock.calls[0];
    const sentBody = JSON.parse((init as RequestInit).body as string);
    expect(Object.keys(sentBody).sort()).toEqual(
      ["intensity_model_version", "sid", "track_model_version"].sort(),
    );
  });
});
