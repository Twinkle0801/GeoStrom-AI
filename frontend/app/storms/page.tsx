import StormExplorer from "@/components/storm/StormExplorer";
import SectionHeader from "@/components/ui/SectionHeader";

export const metadata = {
  title: "Storm Explorer — GeoStrom AI",
  description: "Search and filter historical North Atlantic tropical cyclones.",
};

export default function StormsPage() {
  return (
    <main className="mx-auto max-w-7xl px-6 py-10">
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
