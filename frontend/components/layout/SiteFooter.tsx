import Link from "next/link";

const LINKS = [
  { href: "/storms", label: "Storm Explorer" },
  { href: "/models", label: "Model Performance" },
  { href: "/methodology", label: "Methodology" },
] as const;

export default function SiteFooter() {
  return (
    <footer className="border-t border-border-subtle">
      <div className="mx-auto max-w-7xl px-6 py-10">
        <div className="flex flex-col gap-6 sm:flex-row sm:items-start sm:justify-between">
          <div className="max-w-xl">
            <div className="flex items-center gap-2 text-text-primary">
              <span className="inline-block h-1.5 w-1.5 rounded-full bg-accent" />
              <span className="text-sm font-semibold tracking-tight">GeoStrom AI</span>
            </div>
            <p className="mt-3 text-xs leading-relaxed text-text-muted">
              GeoStrom AI is a retrospective research prototype. All storm data is historical; all
              predictions are historical baseline model output, evaluated against known outcomes.
              This is not an operational forecasting system, weather warning, or safety advisory.
            </p>
          </div>
          <nav aria-label="Footer" className="flex shrink-0 gap-x-6 gap-y-2 text-xs sm:flex-col">
            {LINKS.map((l) => (
              <Link key={l.href} href={l.href} className="text-text-secondary hover:text-text-primary">
                {l.label}
              </Link>
            ))}
          </nav>
        </div>
        <div className="mt-6 flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-border-subtle pt-4 font-mono text-[11px] text-text-muted">
          <span>North Atlantic basin · 1980–2015</span>
          <span aria-hidden>·</span>
          <span>IBTrACS</span>
          <span aria-hidden>·</span>
          <span>HURSAT-B1</span>
          <span aria-hidden>·</span>
          <span>ADT-HURSAT</span>
        </div>
      </div>
    </footer>
  );
}
