import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import ModelSelector from "./ModelSelector";

describe("ModelSelector", () => {
  it("only renders the models actually passed in (never a hardcoded list)", () => {
    render(
      <ModelSelector
        label="Intensity model"
        options={[{ name: "intensity_persistence", version: "v1" }, { name: "intensity_lightgbm", version: "v1" }]}
        value="intensity_lightgbm"
        onChange={vi.fn()}
      />,
    );
    expect(screen.getByText("Persistence")).toBeInTheDocument();
    expect(screen.getByText("LightGBM")).toBeInTheDocument();
    expect(screen.queryByText("GRU")).not.toBeInTheDocument();
  });

  it("marks the recommended production model as 'Best baseline'", () => {
    render(
      <ModelSelector
        label="Intensity model"
        options={[{ name: "intensity_lightgbm", version: "v1" }]}
        value="intensity_lightgbm"
        onChange={vi.fn()}
      />,
    );
    expect(screen.getByText(/best baseline/i)).toBeInTheDocument();
  });

  it("never labels a real per-storm baseline as exploratory", () => {
    render(
      <ModelSelector
        label="Intensity model"
        options={[{ name: "intensity_ridge", version: "v1" }]}
        value="intensity_ridge"
        onChange={vi.fn()}
      />,
    );
    expect(screen.queryByText(/exploratory/i)).not.toBeInTheDocument();
  });

  it("calls onChange with the model name when a button is clicked", () => {
    const onChange = vi.fn();
    render(
      <ModelSelector
        label="Track model"
        options={[{ name: "track_cliper", version: "v1" }, { name: "track_lightgbm", version: "v1" }]}
        value="track_cliper"
        onChange={onChange}
      />,
    );
    fireEvent.click(screen.getByText("LightGBM"));
    expect(onChange).toHaveBeenCalledWith("track_lightgbm");
  });

  it("renders nothing when there are no available models", () => {
    const { container } = render(
      <ModelSelector label="Intensity model" options={[]} value={null} onChange={vi.fn()} />,
    );
    expect(container).toBeEmptyDOMElement();
  });
});
