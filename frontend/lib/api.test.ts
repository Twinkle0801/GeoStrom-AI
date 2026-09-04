import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, getStorm, listStorms } from "./api";

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
});
