import StormExplorer from "@/components/storm/StormExplorer";
import SectionHeader from "@/components/ui/SectionHeader";

export const metadata = {
  title: "Storm Explorer — GeoStrom AI",
  description: "Search and filter historical North Atlantic tropical cyclones.",
};

export default function StormsPage() {
  return (
    <main className="relative mx-auto max-w-7xl px-6 py-10">
      <div
        aria-hidden
        className="pointer-events-none absolute inset-x-0 top-0 -z-10 h-72 bg-[radial-gradient(60%_60%_at_50%_0%,rgba(76,141,255,0.08),transparent_70%)]"
      />
      <SectionHeader
        eyebrow="Storm Explorer"
        title="Historical North Atlantic storms"
        description="Every storm below has real IBTrACS observations. Select one to open its full analysis workspace."
      />
      <div className="mt-6">
        <StormExplorer />
      </div>
    </main>
  );
}
