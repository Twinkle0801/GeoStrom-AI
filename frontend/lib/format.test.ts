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
  it("uses the correct, established name for every known model", () => {
    expect(modelDisplayName("track_cliper")).toBe("CLIPER-style Ridge");
    expect(modelDisplayName("intensity_lightgbm")).toBe("LightGBM");
    expect(modelDisplayName("intensity_persistence")).toBe("Persistence");
    expect(modelDisplayName("track_gru")).toBe("GRU");
  });

  it("falls back to strip-prefix-and-capitalise for an unknown model name", () => {
    expect(modelDisplayName("intensity_newmodel")).toBe("Newmodel");
  });
});

describe("formatTimestamp", () => {
  it("produces a non-empty, human string for a valid ISO timestamp", () => {
    const out = formatTimestamp("2010-06-26T12:00:00Z");
    expect(out).toContain("2010");
  });
});
