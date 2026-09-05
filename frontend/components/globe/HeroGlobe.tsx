"use client";

/**
 * Purely decorative atmospheric visualization for the landing hero -- NOT a
 * scientific instrument. No storm markers are plotted: the project's task
 * brief is explicit that inventing cyclone coordinates for a hero graphic
 * would be a fabrication (Phase 10 §5), and no single real storm is
 * privileged as "the" hero storm, so this renders an abstract rotating
 * globe (`cobe`, a tiny canvas/WebGL renderer -- no React Three Fiber, no
 * scene graph, matching the "lightweight 3D globe" instruction) with a
 * plain, marker-free base colour.
 *
 * Respects `prefers-reduced-motion`: rotation is frozen entirely rather
 * than merely slowed, per task §15/§21.
 */
import { useEffect, useRef } from "react";
import createGlobe, { type COBEOptions } from "cobe";
import { usePrefersReducedMotion } from "@/lib/motion";

// cobe@2.0.1 ships a `.d.ts` that omits `onRender`, even though it is part
// of the real, documented runtime API (see node_modules/cobe/README.md's
// own "Quick Start" example) -- a genuine upstream type-definition gap, not
// a reason to fall back to `any`. This documents the exact discrepancy at
// the one call site that needs it.
type COBEOptionsWithRender = COBEOptions & {
  onRender: (state: { phi?: number }) => void;
};

export default function HeroGlobe() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const reducedMotion = usePrefersReducedMotion();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;

    let phi = 0;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const size = 520;

    const globe = createGlobe(canvas, {
      devicePixelRatio: dpr,
      width: size * dpr,
      height: size * dpr,
      phi: 0,
      theta: 0.32,
      dark: 1,
      diffuse: 1.6,
      mapSamples: 14000,
      mapBrightness: 4.5,
      baseColor: [0.16, 0.22, 0.32],
      markerColor: [0.3, 0.55, 1],
      glowColor: [0.29, 0.55, 1],
      markers: [],
      opacity: 0.9,
      onRender: (state: { phi?: number }) => {
        if (!reducedMotion) {
          phi += 0.0022;
        }
        state.phi = phi;
      },
    } as COBEOptionsWithRender);

    return () => globe.destroy();
  }, [reducedMotion]);

  return (
    <canvas
      ref={canvasRef}
      role="img"
      aria-label="Abstract rotating globe, atmospheric visualization"
      style={{ width: 520, height: 520, maxWidth: "100%", aspectRatio: 1 }}
    />
  );
}
