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

// jsdom does not implement `window.matchMedia` at all (a documented jsdom
// gap, not a bug in this project) -- `lib/motion.ts`'s
// `usePrefersReducedMotion` hook calls it unconditionally on mount, so ANY
// component that uses framer-motion's shared reduced-motion variants
// throws `TypeError: window.matchMedia is not a function` under test
// unless this exists. A minimal, standard polyfill (matches nothing,
// listener registration is a no-op) -- found while giving GeminiPanel a
// motion-aware entrance animation (frontend visual/UX phase).
if (typeof window !== "undefined" && !window.matchMedia) {
  window.matchMedia = (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  }) as MediaQueryList;
}
