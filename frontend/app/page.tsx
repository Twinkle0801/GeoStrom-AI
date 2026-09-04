import StormSelector from "@/components/controls/StormSelector";

export default function HomePage() {
  return (
    <main className="mx-auto max-w-2xl px-6 py-12">
      <header className="mb-8">
        <h1 className="text-4xl font-semibold tracking-tight text-text-primary">GeoStrom AI</h1>
        <p className="mt-2 text-text-secondary">
          Retrospective tropical cyclone research prototype — Phase 3 vertical slice.
        </p>
        <p className="mt-1 text-sm text-text-muted">
          Select a storm to compare its observed track against historical baseline model
          predictions.
        </p>
      </header>
      <section className="rounded-xl border border-border-subtle bg-bg-elevated">
        <StormSelector />
      </section>
      <footer className="mt-8 text-xs text-text-muted">
        Not an operational forecasting system. Predictions are historical baseline model output,
        retrospectively evaluated — never live, never guaranteed.
      </footer>
    </main>
  );
}
