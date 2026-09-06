import Capabilities from "@/components/home/Capabilities";
import FeaturedStorm from "@/components/home/FeaturedStorm";
import Hero from "@/components/home/Hero";
import HowItWorks from "@/components/home/HowItWorks";
import LiveOverview from "@/components/home/LiveOverview";
import ModelIntelligence from "@/components/home/ModelIntelligence";
import Pillars from "@/components/home/Pillars";
import ScientificIntegrity from "@/components/home/ScientificIntegrity";
import {
  getModelPerformance, getStorm, listStorms,
  type CycloneDetail, type CyclonesList, type ModelPerformanceResponse,
} from "@/lib/api";

/**
 * Every section below either needs no data (Hero, Capabilities, Pillars,
 * HowItWorks, ScientificIntegrity -- static/descriptive copy about
 * already-documented facts) or real data fetched here, server-side, with
 * every call independently wrapped so ONE unreachable endpoint degrades
 * only its own section (never a fabricated fallback, never a whole-page
 * crash) -- task §35/§6's "use ONLY existing API data".
 */
async function safeListStorms(): Promise<CyclonesList | null> {
  try {
    return await listStorms({ limit: 500 });
  } catch {
    return null;
  }
}

async function safeModelPerformance(): Promise<ModelPerformanceResponse | null> {
  try {
    return await getModelPerformance();
  } catch {
    return null;
  }
}

async function safeFeaturedStorm(storms: CyclonesList | null): Promise<CycloneDetail | null> {
  const items = storms?.items ?? [];
  if (items.length === 0) return null;
  // Deterministic, real selection criterion -- the storm with the highest
  // recorded lifetime max wind in the fetched catalogue -- never a hand-
  // picked or invented "hero" storm.
  const withWind = items.filter((s) => s.max_wind_kt != null);
  if (withWind.length === 0) return null;
  const mostIntense = withWind.reduce((best, s) => (s.max_wind_kt! > best.max_wind_kt! ? s : best));
  try {
    return await getStorm(mostIntense.sid);
  } catch {
    return null;
  }
}

export default async function HomePage() {
  const storms = await safeListStorms();
  const [modelPerformance, featuredStorm] = await Promise.all([
    safeModelPerformance(),
    safeFeaturedStorm(storms),
  ]);

  return (
    <main>
      <Hero />
      <LiveOverview storms={storms} />
      <Capabilities />
      <Pillars />
      <HowItWorks />
      <FeaturedStorm storm={featuredStorm} />
      <ModelIntelligence data={modelPerformance} />
      <ScientificIntegrity />
    </main>
  );
}
