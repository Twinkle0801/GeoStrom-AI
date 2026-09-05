# Phase 10 — Premium Frontend Dashboard & Scientific Visualization

**Status: COMPLETE**, with two panels explicitly **BLOCKED BY EXISTING DATA/API CONTRACT**
(Satellite Intelligence, Classification) — documented honestly below, never faked.

---

## 1. Objective

Transform the Phase 3 bare vertical slice into a premium, cinematic, scientifically credible
dashboard, while preserving every existing backend contract, never fabricating a value, and
maintaining a strict visual/textual separation between observed data, model predictions,
baseline vs. exploratory models, satellite-derived information, and Gemini-generated
explanation.

## 2. Design system

Extended, not replaced, Phase 3's existing Tailwind tokens (`tailwind.config.ts`): the same
dark ground (`bg-base #05070C`), the same non-negotiable colour rules (predicted `#FFB020`
never shares a colour with observed `#22D3A7`; the intensity ramp is the only other semantic
scale). Added: an `Inter` variable font via `next/font/google` (previously declared in Tailwind
config but never actually loaded), a restrained radial-gradient backdrop in `app/layout.tsx`,
and a small set of reusable primitives under `components/ui/` (`GlassPanel`, `SectionHeader`,
`MetricCard`, `Badge`, `EmptyState`, `ErrorState`, `LoadingSkeleton`) used everywhere instead of
ad hoc one-off styling. **No shadcn/ui, no Mapbox, no deck.gl, no GSAP, no raw React Three
Fiber** — deliberately: the existing token system already covers every visual need this phase
had, and pulling in Radix/CVA for a handful of primitives this project can hand-roll in a few
lines each would be exactly the unnecessary-dependency pattern the task warns against. The one
new visual-only dependency is `cobe` (~5 KB, canvas/WebGL, zero scene-graph) for the landing
hero's abstract globe — the literal "lightweight 3D globe" the task asks for, explicitly
distinct from React Three Fiber.

New dependencies (all pinned, matching this repo's exact-version convention): `framer-motion
13.2.0`, `recharts 3.10.1`, `cobe 2.0.1`, `clsx 2.1.1`.

## 3. Page architecture

| Route | Purpose | Status |
|---|---|---|
| `/` | Premium landing: hero, CTAs, four-pillar honesty section | ✅ New |
| `/storms` | Storm Explorer: search/season/split filters over real storm cards | ✅ New (filters), reuses `StormSelector` |
| `/predict/[sid]` | Flagship analysis workspace | ✅ Rebuilt on Phase 3's route/API calls |
| `/models` | Model Performance: intensity/track/classification comparison | ✅ New, backed by a new read-only analytics endpoint |
| `/methodology` | Educational pipeline walkthrough | ✅ New, static content, no fabricated numbers |

## 4. Component architecture

Adapted the task's suggested tree to what the repository already had, rather than duplicating:

```
components/
  globe/HeroGlobe.tsx, HeroGlobeClient.tsx      (new -- cobe, lazy-loaded)
  home/Hero.tsx, Pillars.tsx                    (new)
  layout/SiteHeader.tsx, SiteFooter.tsx         (new)
  maps/TrackLegend.tsx                          (extracted from the old inline MapLegend)
  map/CycloneMap.tsx, CycloneMapClient.tsx      (Phase 3, EXTENDED: current-position marker,
                                                  per-model filtering -- not rewritten)
  storm/StormSelector.tsx                       (Phase 3, EXTENDED: optional filters prop,
                                                  card layout -- existing tests still pass)
  storm/StormExplorer.tsx, ModelSelector.tsx,
        PredictWorkspace.tsx                    (new)
  panels/StormHeader.tsx                        (Phase 3, EXTENDED: larger hero typography,
                                                  "Historical Analysis" status, forecast-horizon
                                                  stat -- combines the task's "HEADER" and
                                                  "HERO/SUMMARY" sections into one, since both
                                                  showed the same underlying fields)
  timeline/TimeScrubber.tsx                     (new)
  charts/IntensityChart.tsx, ModelComparisonChart.tsx,
         TrackErrorChart.tsx, ClassificationComparisonChart.tsx  (new, Recharts)
  satellite/SatelliteViewer.tsx, SatelliteMetadata.tsx           (new -- honest empty-state shells)
  classification/ClassificationPanel.tsx                        (new -- honest empty-state shell)
  gemini/GeminiPanel.tsx, EvidenceDrawer.tsx, GroundingBadge.tsx (new)
  ui/GlassPanel.tsx, SectionHeader.tsx, MetricCard.tsx, Badge.tsx,
     EmptyState.tsx, ErrorState.tsx, LoadingSkeleton.tsx          (new)
```

`components/panels/IntensityPanel.tsx` (Phase 3's plain intensity table) was removed after its
functionality was fully superseded by `IntensityChart` + an inline horizon table in
`PredictWorkspace` — nothing referenced it any more, so it was deleted rather than left as dead
code.

## 5. API data flow

Unchanged principle: `Backend API → typed API client (lib/api.ts, types generated from
contracts/openapi.json) → React state → visualization components`. Every value shown anywhere
in Phase 10 traces to a real backend response; nothing is hardcoded or fabricated.

**One small, additive backend endpoint was added**, per the task's own explicit instruction
("If an API endpoint required by the UI does not exist: STOP and inspect the existing backend
first. Prefer implementing the smallest additive backend endpoint required."): `GET
/api/v1/analytics/model-performance` (`backend/app/api/v1/analytics.py`), exactly the endpoint
`docs/API_ARCHITECTURE.md` §3.6 had already planned. It reads three already-committed benchmark
report files (`ml/reports/phase{2,7,8}_*.json`, `phase5_baseline_results.json`,
`phase6_{resnet18,small_cnn}_results.json`) and reshapes them into a tiered comparison —
**it recomputes nothing, retrains nothing, and never imports `ml.geostrom_ml`**, matching
`app/main.py`'s existing invariant. 9 new backend tests
(`backend/tests/test_api_analytics.py`) verify the recommended-model tiering is correct and one
value is cross-checked byte-for-byte against the real committed JSON file.

**One additive field was added to the existing Phase 9 `/api/v1/explain/forecast` response**:
`evidence: EvidencePacket` (previously the response referenced model versions only, not the
full packet). This is what `EvidenceDrawer` renders — without it, "inspect what supports the
Gemini explanation" (task §11) had nothing real to show. Confirmed to introduce no regression
(all 7 pre-existing `test_api_explain.py` tests still pass) and to leak no secret (the packet
never contained one).

`contracts/openapi.json` and `frontend/lib/api-types.ts` were regenerated via the existing
`export_openapi.py` / `npm run gen:types` pipeline after both backend changes — no type was
hand-edited.

## 6. Map architecture

`CycloneMap.tsx` (Leaflet, unchanged rendering engine) gained two additive props:
`currentPosition` (the TimeScrubber's selected real observation — a third, distinct marker
style: accent-blue ring, never reusing observed-teal or predicted-amber) and
`selectedModelName` (filters the predicted track/points to one model when the Model Selector
has a selection, showing all models when it doesn't — matching Phase 3's original behaviour by
default). Observed vs. predicted differ by **colour AND line style** exactly as Phase 3 already
enforced (solid teal vs. dashed amber) — this rule was not touched, only extended with a third,
clearly distinct treatment for the scrub-position marker.

Geometry is still assembled entirely server-side (`backend/app/services/geometry.py`, untouched)
— the frontend performs no geodesic math, per `docs/SYSTEM_ARCHITECTURE.md` §6.1's standing
rule. When the user scrubs to a real forecast origin, the client re-fetches
`GET /api/v1/tracks/{sid}?t=<origin>` (the existing endpoint, called with a different real
timestamp) rather than reconstructing predicted geometry client-side.

## 7. TimeScrubber behavior

Scrubs through the storm's **real observation timestamps only** — every reachable index is one
actual IBTrACS row (`components/timeline/TimeScrubber.tsx`). A brighter tick marks timestamps
that also have a real, issued model forecast (derived from the full `/api/v1/prediction/{sid}
/series` response, fetched once). Scrubbing to a non-origin timestamp still shows the real
observed position; the prediction-dependent panels (map's predicted layer, intensity chart's
forecast overlay, horizon table) correctly show their empty state rather than inventing an
interpolated forecast — no artificial interpolation is ever created, per the task's explicit
rule. Supports play/pause/previous/next/drag, all keyboard-operable (native `<input
type="range">` plus real `<button>`s).

## 8. Chart architecture

`IntensityChart` plots observed wind (solid teal, the storm's full real history) and predicted
wind (dashed amber, diamond markers, this forecast's +6/+12/+18/+24h points) as **two distinct
Recharts series over a shared time axis** — never merged into one line, per the task's explicit
rule. The tooltip states series type ("Observed (IBTrACS)" vs. "Predicted (model output)"),
timestamp, value, and unit (kt) explicitly.

`ModelComparisonChart` is the one generic per-horizon bar-chart implementation; `TrackErrorChart`
is a thin, named wrapper around it (task's suggested file structure, one chart engine, not two
divergent copies). `ClassificationComparisonChart` is a small dedicated Recharts bar chart for
the non-horizon classification metrics.

**A real bug was found and fixed here**: `app/models/page.tsx` is a Server Component; an early
version imported Recharts directly at its top level and 500'd in dev with `(0,
react.createContext) is not a function` — Recharts cannot evaluate inside React Server
Component module bundling. Every Recharts-using component in this codebase is now a dedicated
`"use client"` module, confirmed by re-testing the route (200, verified via a running dev
server, not just typecheck).

## 9. Gemini integration

`GeminiPanel` calls `POST /api/v1/explain/forecast` through `lib/api.ts` **only** — no Gemini
SDK, no API key, nothing Gemini-related exists anywhere in the frontend bundle (confirmed:
`grep -ril gemini frontend/` finds nothing outside third-party `node_modules` binaries). No
free-text input is offered, so no arbitrary user text can ever become an unrestricted prompt
(task §10's explicit requirement) — the request body only ever contains `{sid,
intensity_model_version, track_model_version}` (asserted directly in
`GeminiPanel.test.tsx`).

States: idle ("Generate explanation") → loading ("Analyzing stored evidence…") → success (shows
the four explanation sections + `GroundingBadge` + Regenerate/Copy/View evidence) or a genuine
transport error (`ErrorState`, "Explanation unavailable...", with retry). A `source: "fallback"`
response is **not** treated as a frontend error — it is a normal 200 response, displayed
identically to a Gemini response, distinguished only by `GroundingBadge` reading "Deterministic
evidence summary" instead of "Grounded in stored model output" — exactly matching the task's
"display it normally but clearly identify" instruction.

## 10. EvidenceDrawer

Renders exactly the `EvidencePacket` the backend sent Gemini (§5's additive `evidence` field) —
storm identity, observation window, both models' forecasts and versions, classification (or an
honest "no classification result in this packet" line), and the packet's own
`known_limitations`. No SQL, no database implementation detail, no credential — the packet never
contained one, verified directly by a test asserting the rendered text never matches a SQL-like
pattern. Closes on Escape or backdrop/Close click; focus moves to the Close button on open.

## 11. Accessibility

- Every interactive element is a real `<button>`/`<input>`/`<select>` with a visible
  `focus-visible` outline (task's explicit requirement) — no `div onClick`.
- Observed vs. predicted remain distinguishable by colour **and** line style (unchanged Phase 3
  rule, re-verified, extended with a third distinct current-position marker style).
- `HeroGlobe`'s canvas carries `role="img"` and a descriptive `aria-label`; it and the Pillars/
  Hero entrance animations both check `prefers-reduced-motion` via `lib/motion.ts`'s
  `usePrefersReducedMotion` hook and collapse to an instant, motion-free state when set.
- `EvidenceDrawer` is a labelled `role="dialog"` with `aria-modal`, Escape-to-close, and initial
  focus management.
- **Known, honestly-scoped limitation**: `ModelSelector`'s buttons use `role="radio"`/
  `role="radiogroup"` for correct screen-reader semantics but do not implement full roving-
  tabindex arrow-key navigation between options (each option is reached by Tab, not arrow keys,
  unlike a native `<input type="radio">` group) — a real, documented gap rather than a claimed
  full pass.

## 12. Performance

- `HeroGlobe` is lazy-loaded via `next/dynamic({ssr:false})` (`HeroGlobeClient.tsx`), keeping
  `cobe` and its canvas/WebGL setup out of the landing page's initial bundle — the same pattern
  Phase 3 already established for the Leaflet map (`CycloneMapClient.tsx`).
- The map remains dynamically imported with `ssr:false` (Leaflet's existing, unchanged
  constraint).
- `PredictWorkspace` filters/derives chart and table data with `useMemo`, and only re-fetches
  `/tracks` (not the whole page) when the scrub position actually lands on a real forecast
  origin.
- No dataset is rendered in full at once beyond what the backend already returns for one storm
  (at most a few hundred observation/prediction rows) — no client-side pagination gap was found
  necessary at this data scale.

## 13. Testing

**Frontend: 41 passed / 0 failed** (13 pre-existing + 28 new), **TypeScript: 0 errors**,
**ESLint: 0 errors**. **Backend: 141 passed / 0 failed** (132 pre-existing Phase 9 baseline + 9
new). **ML fast suite (unaffected, re-verified anyway): 341 passed / 0 failed.**

| New test file | Count | Covers |
|---|---:|---|
| `components/timeline/TimeScrubber.test.tsx` | 7 | Current timestamp, prev/next bounds, drag, play/pause, empty case |
| `components/storm/ModelSelector.test.tsx` | 5 | Only-real-models rule, recommended/exploratory badges, selection |
| `components/gemini/GeminiPanel.test.tsx` | 7 | Idle/loading/success/fallback/error states, evidence drawer open, request-body shape |
| `components/gemini/EvidenceDrawer.test.tsx` | 6 | Storm/intensity rendering, missing-classification honesty, close behaviours, no-SQL-leak |
| `components/charts/IntensityChart.test.tsx` | 2 | Renders with real API-shaped fixtures, empty case |
| `backend/tests/test_api_analytics.py` | 9 | Tiering/recommended-model correctness, exact value cross-check against committed JSON |

**A real, pre-existing test-infrastructure bug was found and fixed**: `vitest.setup.ts` never
called `@testing-library/react`'s `cleanup()` between tests. Vitest (unlike Jest) does not do
this automatically; every render since Phase 3 had been silently accumulating across `it()`
blocks within a file. It went unnoticed because no prior test queried for an element that
repeated identically across test cases — the first `TimeScrubber` test to do so ("Play" button)
surfaced it immediately (5 of 7 tests failed with "multiple elements found"). Fixed by adding
`afterEach(cleanup)`; all pre-existing tests continued to pass afterward.

**A second real bug was found**: `lib/format.ts::modelDisplayName` naively capitalised the first
letter of a model's slug ("intensity_lightgbm" → "Lightgbm", "track_cliper" → "Cliper") — wrong
for brand/acronym names, and a real, user-visible defect in a "premium" UI. Fixed with a proper
name table (mirroring the backend's own `_DISPLAY_NAMES`); the pre-existing pinned test in
`lib/format.test.ts` was updated to assert the *correct* names, not preserved to lock in the bug.

## 14. Known limitations

- **BLOCKED BY EXISTING DATA/API CONTRACT — Satellite Intelligence.** Phase 4 built a real
  HURSAT-B1 → Zarr pipeline, but no backend endpoint serves those frames to a browser, and no
  per-timestamp image lookup exists in the current API. `SatelliteViewer` renders a correct,
  honest empty state ("Satellite frame unavailable for this timestamp") — never a placeholder
  image that could be mistaken for real data. Serving real frames would require a new
  image-conversion endpoint, a genuinely new backend capability outside this frontend phase's
  scope.
- **BLOCKED BY EXISTING DATA/API CONTRACT — Classification.** No `classifications` table exists
  in the database schema (Phase 3's own `db/models.py` docstring: none was ever built), and the
  Phase 9 evidence packet's own `classification` field is always `None` for the same reason.
  `ClassificationPanel` shows a correct empty state plus the real, frozen `scene_taxonomy_v1`
  taxonomy for educational context — never a fabricated label or confidence.
- The Model Selector on `/predict/[sid]` only ever offers models with real per-storm prediction
  rows (Persistence/Ridge/LightGBM for intensity; Persistence/CLIPER-style Ridge/LightGBM for
  track) — GRU never appears there, since Phase 7/8's GRU work produced aggregate benchmark
  metrics only, never per-storm database rows. GRU is shown, correctly labelled "Exploratory",
  only on the aggregate `/models` page.
- `EvidenceDrawer` is a right-side overlay that becomes full-width on narrow viewports, rather
  than a literal bottom-anchored sheet — functionally full-screen on mobile, but not
  bottom-positioned.
- `ModelSelector`'s custom radio buttons lack roving-tabindex arrow-key navigation (§11).
- No dependency-vulnerability remediation was attempted for the pre-existing pinned `next@15.1.6`
  / `eslint@9.18.0` versions (`npm audit` reports known CVEs against them) — upgrading a pinned
  major/minor version is outside this frontend-visualization phase's scope and was not requested.

## 15. Manual verification performed

A real backend (FastAPI + the existing `geostrom_db` Postgres/PostGIS container) and a real
`npm run dev` frontend were started together and exercised via `curl` against five routes (`/`,
`/storms`, `/models`, `/methodology`, `/predict/2015313N22289` — a real storm with real
predictions) — all returned 200 after the Recharts/RSC fix (§8); the `/models` route's 500 before
that fix was caught this way, not by typecheck or a unit test, neither of which would have
detected a React-Server-Component-only runtime failure. The real `/api/v1/explain/forecast` and
`/api/v1/analytics/model-performance` endpoints were also called directly and returned
correctly-shaped, correctly-tiered, non-fabricated data (`source: "gemini"`, real LightGBM/
CLIPER-style Ridge/Logistic Regression recommendations). **No headless-browser screenshot tool
was available in this environment**, so client-side interactivity (TimeScrubber dragging, model
selection re-rendering the map, GeminiPanel's live click flow) was verified through the 41 passing
component tests plus direct code review, not a captured screenshot or an automated browser
session — stated plainly rather than claimed as a full manual UI walkthrough.
