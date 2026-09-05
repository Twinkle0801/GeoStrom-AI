"use client";

/**
 * Code-split wrapper (task §16: "lazy load 3D globe") -- `cobe` and its
 * canvas/WebGL setup are kept out of the landing page's initial JS bundle
 * and loaded only once this component mounts.
 */
import dynamic from "next/dynamic";

const HeroGlobe = dynamic(() => import("./HeroGlobe"), {
  ssr: false,
  loading: () => (
    <div
      aria-hidden
      className="h-[520px] w-[520px] max-w-full animate-pulse rounded-full bg-white/[0.03]"
      style={{ aspectRatio: 1 }}
    />
  ),
});

export default function HeroGlobeClient() {
  return <HeroGlobe />;
}
