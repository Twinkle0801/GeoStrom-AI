export default function SiteFooter() {
  return (
    <footer className="border-t border-border-subtle">
      <div className="mx-auto max-w-7xl px-6 py-8 text-xs text-text-muted">
        <p>
          GeoStrom AI is a retrospective research prototype. All storm data is historical; all
          predictions are historical baseline model output, evaluated against known outcomes. This
          is not an operational forecasting system, weather warning, or safety advisory.
        </p>
        <p className="mt-2">North Atlantic basin, 1980-2015 · IBTrACS · HURSAT-B1 · ADT-HURSAT</p>
      </div>
    </footer>
  );
}
