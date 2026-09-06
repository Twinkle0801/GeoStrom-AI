import Link from "next/link";
import { notFound } from "next/navigation";
import PredictWorkspace from "@/components/storm/PredictWorkspace";
import { ArrowRightIcon } from "@/components/ui/Icons";
import {
  ApiError, getStorm, getStormObservations, getStormPredictionSeries, getStormTrack,
} from "@/lib/api";

// Phase 0 URL pattern (docs/UI_UX_ARCHITECTURE.md §5.4): /predict/[sid].
// Phase 10 upgrades this route to the flagship analysis workspace
// (PredictWorkspace) while keeping the same URL shape and the same
// typed API client Phase 3 established.

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

  const [observations, predictionSeries, initialTrack] = await Promise.all([
    getStormObservations(sid),
    storm.has_predictions ? getStormPredictionSeries(sid).catch(() => []) : Promise.resolve([]),
    getStormTrack(sid),
  ]);

  return (
    <main>
      <div className="mx-auto max-w-7xl px-6 pt-4">
        <Link
          href="/storms"
          className="group inline-flex items-center gap-1.5 text-sm text-text-secondary transition-colors hover:text-accent-soft"
        >
          <ArrowRightIcon width={14} height={14} className="rotate-180 transition-transform duration-200 group-hover:-translate-x-0.5" />
          Storm Explorer
        </Link>
      </div>
      <PredictWorkspace
        storm={storm}
        observations={observations}
        predictionSeries={predictionSeries}
        initialTrack={initialTrack}
      />
    </main>
  );
}
