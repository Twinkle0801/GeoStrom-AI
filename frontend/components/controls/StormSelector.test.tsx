import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import StormSelector from "./StormSelector";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("StormSelector", () => {
  it("shows a loading skeleton before data arrives", () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => new Promise(() => {})), // never resolves
    );
    render(<StormSelector />);
    expect(screen.getByTestId("storm-selector-loading")).toBeInTheDocument();
  });

  it("renders the storm list once data loads", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          items: [
            {
              sid: "2010176N16278", name: null, season: 2010, basin: "NA",
              start_time: "2010-06-26T00:00:00Z", end_time: "2010-06-27T00:00:00Z",
              n_observations: 5, max_wind_kt: 55, min_pressure_hpa: 1000,
              max_category: 0, made_landfall: null, split: "test",
            },
          ],
          total: 1, limit: 50, offset: 0,
        }),
      }),
    );
    render(<StormSelector />);
    await waitFor(() => expect(screen.getByTestId("storm-selector-list")).toBeInTheDocument());
    expect(screen.getByText("2010176N16278")).toBeInTheDocument();
    expect(screen.getByText(/Tropical Storm/)).toBeInTheDocument();
  });

  it("shows an empty state when there are no storms", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ items: [], total: 0, limit: 50, offset: 0 }),
      }),
    );
    render(<StormSelector />);
    await waitFor(() => expect(screen.getByTestId("storm-selector-empty")).toBeInTheDocument());
  });

  it("shows an error state when the API call fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        json: async () => ({ detail: "database unreachable" }),
      }),
    );
    render(<StormSelector />);
    await waitFor(() => expect(screen.getByTestId("storm-selector-error")).toBeInTheDocument());
    expect(screen.getByText(/database unreachable/)).toBeInTheDocument();
  });
});
