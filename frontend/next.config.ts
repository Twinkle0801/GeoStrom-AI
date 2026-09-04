import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Phase 3 vertical slice: no image optimisation service, no rewrites/
  // redirects complexity. Deliberately minimal per the phase's own scope
  // rule (docs/PHASE_3_VERTICAL_SLICE.md).
  reactStrictMode: true,
};

export default nextConfig;
