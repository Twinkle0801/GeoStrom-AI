import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import TimeScrubber from "./TimeScrubber";

const timestamps = [
  "2010-06-26T00:00:00Z", "2010-06-26T06:00:00Z", "2010-06-26T12:00:00Z", "2010-06-26T18:00:00Z",
];

describe("TimeScrubber", () => {
  it("renders the currently selected real timestamp", () => {
    render(
      <TimeScrubber timestamps={timestamps} originTimestamps={new Set()} index={1} onChange={vi.fn()} />,
    );
    expect(screen.getByRole("slider", { name: /storm timeline/i })).toHaveValue("1");
  });

  it("disables Previous at the first index and Next at the last index", () => {
    const { rerender } = render(
      <TimeScrubber timestamps={timestamps} originTimestamps={new Set()} index={0} onChange={vi.fn()} />,
    );
    expect(screen.getByRole("button", { name: /previous timestamp/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /next timestamp/i })).not.toBeDisabled();

    rerender(
      <TimeScrubber
        timestamps={timestamps} originTimestamps={new Set()}
        index={timestamps.length - 1} onChange={vi.fn()}
      />,
    );
    expect(screen.getByRole("button", { name: /next timestamp/i })).toBeDisabled();
  });

  it("calls onChange with the next index when Next is clicked", () => {
    const onChange = vi.fn();
    render(
      <TimeScrubber timestamps={timestamps} originTimestamps={new Set()} index={1} onChange={onChange} />,
    );
    fireEvent.click(screen.getByRole("button", { name: /next timestamp/i }));
    expect(onChange).toHaveBeenCalledWith(2);
  });

  it("calls onChange with the previous index when Previous is clicked", () => {
    const onChange = vi.fn();
    render(
      <TimeScrubber timestamps={timestamps} originTimestamps={new Set()} index={2} onChange={onChange} />,
    );
    fireEvent.click(screen.getByRole("button", { name: /previous timestamp/i }));
    expect(onChange).toHaveBeenCalledWith(1);
  });

  it("calls onChange when the range slider is dragged", () => {
    const onChange = vi.fn();
    render(
      <TimeScrubber timestamps={timestamps} originTimestamps={new Set()} index={0} onChange={onChange} />,
    );
    fireEvent.change(screen.getByRole("slider", { name: /storm timeline/i }), { target: { value: "3" } });
    expect(onChange).toHaveBeenCalledWith(3);
  });

  it("toggles the play/pause button label", () => {
    render(
      <TimeScrubber timestamps={timestamps} originTimestamps={new Set()} index={0} onChange={vi.fn()} />,
    );
    const playButton = screen.getByRole("button", { name: /^play$/i });
    fireEvent.click(playButton);
    expect(screen.getByRole("button", { name: /^pause$/i })).toBeInTheDocument();
  });

  it("renders nothing when there are no timestamps", () => {
    const { container } = render(
      <TimeScrubber timestamps={[]} originTimestamps={new Set()} index={0} onChange={vi.fn()} />,
    );
    expect(container).toBeEmptyDOMElement();
  });
});
