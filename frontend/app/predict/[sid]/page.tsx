import Link from "next/link";
import { notFound } from "next/navigation";
import CycloneMapClient from "@/components/map/CycloneMapClient";
import IntensityPanel from "@/components/panels/IntensityPanel";
import StormHeader from "@/components/panels/StormHeader";
import { ApiError, getStorm, getStormPrediction, getStormTrack } from "@/lib/api";

// Phase 0 URL pattern (docs/UI_UX_ARCHITECTURE.md §5.4): /predict/[sid].
// (sid, t) belongs in the URL so every view is shareable -- Phase 3 keeps
// `sid` in the path; the `t` (forecast origin) query param is read by the
// API client but not yet mirrored into the URL bar, a scope cut documented
// in docs/PHASE_3_VERTICAL_SLICE.md's known limitations.

export default async function PredictPage({
  params,
}: {
  params: Promise<{ sid: string }>;
}) {
  const { sid } = await params;

  let storm;
  try {
    storm = await getStorm(sid);
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) notFound();
    throw err;
  }

  const [track, predictions] = await Promise.all([
    getStormTrack(sid),
    storm.has_predictions
      ? getStormPrediction(sid).catch(() => [])
      : Promise.resolve([]),
  ]);

  return (
    <main className="mx-auto max-w-6xl px-6 py-8">
      <Link href="/" className="text-sm text-accent hover:underline">
        ← All storms
      </Link>
      <div className="mt-4">
        <StormHeader storm={storm} />
      </div>

      <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="h-[520px] rounded-xl border border-border-subtle lg:col-span-2">
          <CycloneMapClient track={track} />
        </div>
        <div className="flex flex-col gap-4">
          <MapLegend />
          <IntensityPanel predictions={predictions} />
        </div>
      </div>
    </main>
  );
}

function MapLegend() {
  return (
    <div className="rounded-lg border border-border-subtle bg-white/5 p-4 text-xs">
      <h3 className="mb-2 font-semibold uppercase tracking-wide text-text-secondary">Legend</h3>
      <div className="flex items-center gap-2">
        <span className="inline-block h-0.5 w-6 bg-truth" />
        <span className="text-text-secondary">Observed track (solid)</span>
      </div>
      <div className="mt-1 flex items-center gap-2">
        <span className="inline-block h-0.5 w-6 border-t-2 border-dashed border-predicted" />
        <span className="text-text-secondary">Predicted track (dashed)</span>
      </div>
      <p className="mt-2 text-text-muted">
        Each predicted line is one historical baseline model. Colour and line style both
        distinguish prediction from observation.
      </p>
    </div>
  );
}
