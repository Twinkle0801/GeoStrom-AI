import { describe, expect, it } from "vitest";
import { categoryLabel, formatTimestamp, modelDisplayName } from "./format";

describe("categoryLabel", () => {
  it("maps known categories", () => {
    expect(categoryLabel(-1)).toBe("Tropical Depression");
    expect(categoryLabel(0)).toBe("Tropical Storm");
    expect(categoryLabel(5)).toBe("Category 5");
  });

  it("handles null/undefined", () => {
    expect(categoryLabel(null)).toBe("Unknown");
    expect(categoryLabel(undefined)).toBe("Unknown");
  });
});

describe("modelDisplayName", () => {
  it("strips task prefix and capitalises", () => {
    expect(modelDisplayName("track_cliper")).toBe("Cliper");
    expect(modelDisplayName("intensity_lightgbm")).toBe("Lightgbm");
  });
});

describe("formatTimestamp", () => {
  it("produces a non-empty, human string for a valid ISO timestamp", () => {
    const out = formatTimestamp("2010-06-26T12:00:00Z");
    expect(out).toContain("2010");
  });
});
