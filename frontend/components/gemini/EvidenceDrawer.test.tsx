import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import EvidenceDrawer from "./EvidenceDrawer";

const evidence = {
  evidence_schema_version: "v1",
  generated_at: "2010-06-26T12:00:00Z",
  storm: { sid: "2010176N16278", name: "ALEX", season: 2010, basin: "NA", n_observations: 17 },
  current_state: {
    timestamp: "2010-06-26T12:00:00Z", lat: 25.4, lon: -87.6, wind_kt: 95, pressure_hpa: 948,
  },
  recent_history: [],
  intensity: {
    origin_ts: "2010-06-26T12:00:00Z",
    forecasts: [{ lead_hours: 24, pred_wind_kt: 92.4, true_wind_kt: 90.0 }],
    context: { display_name: "LightGBM", model_version: "v1", dataset_version: "v1" },
  },
  track: null,
  classification: null,
  known_limitations: ["Retrospective research prototype."],
  forbidden_claims: [],
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
} as any;

describe("EvidenceDrawer", () => {
  it("shows the storm identity and observation window", () => {
    render(<EvidenceDrawer evidence={evidence} onClose={vi.fn()} />);
    expect(screen.getByText("ALEX")).toBeInTheDocument();
    expect(screen.getByText("2010176N16278")).toBeInTheDocument();
  });

  it("shows the intensity model and its forecast values", () => {
    render(<EvidenceDrawer evidence={evidence} onClose={vi.fn()} />);
    expect(screen.getByText(/LightGBM v1/)).toBeInTheDocument();
    expect(screen.getByText(/predicted 92.4 kt/)).toBeInTheDocument();
  });

  it("states plainly when no classification result is present -- never fabricates one", () => {
    render(<EvidenceDrawer evidence={evidence} onClose={vi.fn()} />);
    expect(screen.getByText(/no classification result in this packet/i)).toBeInTheDocument();
  });

  it("calls onClose when the Close button is clicked", () => {
    const onClose = vi.fn();
    render(<EvidenceDrawer evidence={evidence} onClose={onClose} />);
    fireEvent.click(screen.getByRole("button", { name: /close evidence drawer/i }));
    expect(onClose).toHaveBeenCalled();
  });

  it("calls onClose when Escape is pressed", () => {
    const onClose = vi.fn();
    render(<EvidenceDrawer evidence={evidence} onClose={onClose} />);
    fireEvent.keyDown(window, { key: "Escape" });
    expect(onClose).toHaveBeenCalled();
  });

  it("never renders raw SQL or a database implementation detail", () => {
    const { container } = render(<EvidenceDrawer evidence={evidence} onClose={vi.fn()} />);
    expect(container.textContent).not.toMatch(/select \*|from predictions|from storms/i);
  });
});
