import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";
import "@testing-library/jest-dom/vitest";

// Vitest (unlike Jest's testEnvironment hook) does not automatically call
// @testing-library/react's cleanup() between tests -- without this, a
// render() from one `it()` stays mounted into the next, and any query for
// an element that repeats across test cases (e.g. a "Play" button
// re-rendered with different props) throws "multiple elements found" for
// reasons that have nothing to do with the component under test. Found
// while adding TimeScrubber.test.tsx (Phase 10).
afterEach(cleanup);
