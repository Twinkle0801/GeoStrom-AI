# UI / UX ARCHITECTURE — GeoStrom AI

**Phase:** 0 (Architecture) · **Status:** Complete · No frontend code has been written.

---

## 1. Design Direction

### 1.1 The reference, and how it is used

The brief cites **designcode.io** as the quality bar. **We do not copy it.** We extract the
*principles* that make it read as premium and re-apply them to a scientific instrument:

| Principle observed | How GeoStrom AI applies it |
|---|---|
| Deep dark ground, content floating above it | Dark canvas so satellite imagery, track lines, and intensity colours are the brightest things on screen |
| Generous whitespace, restrained element count | Each view answers one question. No dashboard-of-everything |
| Large, confident typography with a wide scale | Storm name and current intensity are display-scale; supporting data is quiet |
| Glass surfaces with blur and thin luminous borders | Map overlay panels float above the map without hiding it |
| Subtle multi-stop gradients, never flat black | Gradients carry meaning — the intensity scale is the accent ramp |
| Motion that reveals structure, not motion for its own sake | Elements enter as data arrives; the map animates along the storm's timeline |
| Depth via layering and shadow rather than skeuomorphism | Clear z-hierarchy: map → overlay → modal |

**The critical adaptation:** designcode.io is a marketing and course site — its job is to impress.
GeoStrom AI's job is to **communicate uncertain scientific results without overstating them.**
Where the two conflict, legibility and honesty win. Concretely: the cinematic treatment lives on the
landing page and in transitions; **the analysis surfaces stay calm, dense, and readable.** A forecast
cone must never be hard to read because of a gradient.

### 1.2 Design tokens

```
COLOUR — dark ground
  --bg-base        #05070C     near-black, faintly blue
  --bg-elevated    #0B0F17
  --surface-glass  rgba(255,255,255,0.045)  + backdrop-blur 20px
  --border-subtle  rgba(255,255,255,0.09)
  --border-lume    rgba(120,180,255,0.22)

TEXT
  --text-primary   #F2F5FA
  --text-secondary #9BA6B8
  --text-muted     #5E6979

ACCENT — cool, instrument-like
  --accent         #4C8DFF
  --accent-soft    #7FB0FF
  --accent-glow    radial, rgba(76,141,255,0.28)

INTENSITY RAMP — the one semantic colour scale (also the "brand" gradient)
  TD  #4CC9F0   TS  #4895EF   C1 #4361EE   C2 #7209B7
  C3  #B5179E   C4 #F72585   C5 #FF5400
  (perceptually ordered; verify against colour-blind simulation before locking)

STATE
  --truth          #22D3A7   observed / ground truth
  --predicted      #FFB020   model output — always visually distinct from truth
  --uncertainty    rgba(255,176,32,0.14)   cone fill

TYPE  Inter or Geist (variable) + JetBrains Mono for numerics
  display  clamp(2.75rem, 6vw, 5.5rem)   tight tracking, 600
  h1 2.25rem · h2 1.5rem · body 0.95rem · caption 0.8rem
  All numeric readouts use tabular figures so values do not jitter when updating

SPACE   4 / 8 / 12 / 16 / 24 / 32 / 48 / 64 / 96
RADIUS  8 (control) · 16 (card) · 24 (panel)
MOTION  120ms micro · 240ms element · 480ms view
        ease-out cubic-bezier(0.22, 1, 0.36, 1)
```

**Two non-negotiable colour rules:**

1. **Predicted and observed must never share a colour.** A viewer must distinguish model output from
   ground truth at a glance, without a legend.
2. **The intensity ramp is the only semantic scale.** It is used identically on the map, in charts,
   and on badges, and nothing else uses those hues.

---

## 2. Framework Stack

| Layer | Choice | Rationale |
|---|---|---|
| Framework | **Next.js 15, App Router** | Server components fetch storm data server-side, keeping the client bundle small — which matters because the map and chart libraries are already heavy |
| Language | **TypeScript, strict** | API types are generated from OpenAPI; strictness is what makes that generation valuable |
| Styling | **Tailwind CSS** + CSS custom properties for tokens | Tokens as CSS variables (not just Tailwind config) so the map layers and canvas code read the same values |
| State | **TanStack Query** for server state; React state/`nuqs` for view state | Caching, deduplication, and loading states solved. **No Redux/Zustand** — there is almost no client state beyond "which storm, which time", which belongs in the URL |
| URL state | Storm and timestep in the URL | Every view is shareable and linkable — genuinely valuable for a demo |

---

## 3. Frontend Technology Evaluation

The brief lists candidate libraries. **Recommending all of them would be the wrong answer.**
Each is assessed on value delivered versus cost incurred.

| Technology | Verdict | Reasoning |
|---|---|---|
| **Framer Motion** | ✅ **Adopt** | Highest premium-feel-per-kilobyte in the stack. Layout animations, shared-element transitions, scroll reveals, and `AnimatePresence` cover the entire motion brief. Respects `prefers-reduced-motion` natively |
| **Leaflet + react-leaflet** | ✅ **Adopt** | The map is the product. Leaflet is small, dependency-light, needs no token, and handles polylines, markers, and polygons — precisely our workload of a few hundred vertices. Mature React bindings |
| **Recharts** | ✅ **Adopt** | Intensity timelines, error-growth curves, class probability bars. Declarative, themeable, well-typed. Sufficient at our data volume |
| **react-globe.gl** *(or `cobe`)* | ✅ **Adopt — one instance only** | The **one** place 3D earns its place: a rotating globe on the landing page showing historical tracks as arcs. Genuinely striking, immediately communicates what the project is, and needs no interaction logic. `cobe` is ~5 kB if a static globe suffices; `react-globe.gl` if arcs are wanted |
| **Three.js / React Three Fiber (direct)** | ⚠️ **Only via the globe wrapper** | Writing raw R3F means owning a render loop, camera rig, disposal, and mobile GPU performance. That is days of work competing directly with the ML deliverables, for decoration. **Use the wrapper; do not hand-roll a scene** |
| **GSAP** | ❌ **Reject** | Overlaps almost entirely with Framer Motion. Two animation systems means two mental models, two bundles, and conflicting scroll handlers. Framer Motion covers everything the brief describes |
| **Mapbox GL JS** | ❌ **Reject** | Requires an access token and a billing account. MapLibre GL JS is the open fork with the same API if vector tiles are ever needed |
| **Lenis (smooth scroll)** | ⚪ **Optional — landing page only** | ~3 kB, and it is much of what makes premium sites *feel* premium. **Must be disabled on data pages** — hijacked scrolling in a dense analysis view is actively hostile |
| **D3 (direct)** | ⚪ **Only if Recharts blocks something** | Powerful but a large time sink. Reach for it only for a specific visual Recharts cannot express |
| **shadcn/ui** | ✅ **Adopt** | Copy-in Radix components: accessible primitives (dialog, tooltip, select, tabs) with no runtime dependency and full styling control. Accessibility for free is the main argument |
| **deck.gl** | ❌ **Reject** | Built for hundreds of thousands of points. We have hundreds. Large bundle, steep learning curve, no benefit |

### 3.1 Where complexity would be wasted

Being explicit, since the brief asked:

- **A fully 3D globe as the primary analysis interface.** It looks impressive and is *worse* for
  reading a track: occlusion, distorted distance perception, and hard interaction. **2D map for
  analysis; 3D globe for the landing page.**
- **Animating the map itself in 3D** (tilt, extruded storms). High cost, low information gain.
- **A physics/particle wind field.** Beautiful, and we have no wind-field data — only point
  intensity. Rendering one would be **fabricating data visually**, which is the same failure mode as
  a Gemini hallucination and is disallowed on the same grounds.
- **Scroll-jacked storytelling on data pages.** Fights the analyst.
- **A custom WebGL chart engine.** Recharts is sufficient at this volume.

---

## 4. Component Architecture

```
components/
├── map/
│   ├── CycloneMap.tsx           Leaflet container, dark basemap, layer orchestration
│   ├── TrackLayer.tsx           observed polyline, intensity-coloured segments
│   ├── PredictedTrackLayer.tsx  dashed predicted line + horizon markers
│   ├── UncertaintyCone.tsx      polygon from stored geometry
│   ├── ActualFutureLayer.tsx    ground-truth comparison line
│   ├── StormMarker.tsx          current position, pulsing, intensity-coloured
│   └── MapLegend.tsx            intensity ramp + truth/predicted key
├── charts/
│   ├── IntensityTimeline.tsx    wind & pressure vs time; predicted overlay + error band
│   ├── ErrorGrowthChart.tsx     mean error vs lead time, model vs baseline
│   ├── ClassProbabilityBar.tsx  classification distribution
│   └── DetectionStrip.tsx       detection probability along the storm's life
├── panels/
│   ├── StormHeader.tsx          name, season, basin, current category — display type
│   ├── StateReadout.tsx         wind / pressure / position / motion, mono tabular
│   ├── ForecastTable.tsx        per-horizon prediction vs truth vs error
│   ├── ModelBadge.tsx           model_version + measured skill — ALWAYS shown with a prediction
│   ├── ConfidenceMeter.tsx      calibrated confidence, with an honest label
│   ├── GeminiPanel.tsx          streamed narrative + "View the data behind this"
│   └── EvidenceDrawer.tsx       the raw evidence packet, expandable
├── controls/
│   ├── TimeScrubber.tsx         the central interaction — see §5.4
│   ├── StormSelector.tsx        searchable combobox
│   ├── LayerToggles.tsx
│   └── ModelSwitcher.tsx        compare model versions
├── imagery/
│   ├── IRFrame.tsx              satellite thumbnail for the selected timestep
│   └── FrameFilmstrip.tsx       adjacent frames
└── marketing/
    ├── GlobeHero.tsx            the single 3D element
    ├── ScrollReveal.tsx
    └── ArchitectureDiagram.tsx
```

**`ModelBadge` is a required companion to every prediction surface.** A predicted number shown
without its model version and measured error is a misleading number. This is a design rule, not a
nice-to-have.

---

## 5. Page Architecture

### 5.1 `/` — Landing

| | |
|---|---|
| **Purpose** | Communicate what GeoStrom AI is in under ten seconds; establish credibility; route to the product |
| **Components** | `GlobeHero` (rotating globe, historical track arcs) · display-scale headline · four capability cards (detection / classification / intensity / track) · scroll-revealed architecture diagram · honest metrics strip (real numbers from `/analytics/model-performance`, never invented) · dataset attribution · CTA to the Monitor |
| **Data** | `GET /analytics/model-performance` · `GET /tracks/bulk` (a curated handful of storms for the globe) |
| **Interactions** | Globe auto-rotates and pauses on hover · scroll-triggered reveals · magnetic CTA |
| **Notes** | The only page with cinematic motion at full strength. Globe is lazy-loaded below the fold with a static fallback so it never blocks LCP |

### 5.2 `/monitor` — Cyclone Monitor

| | |
|---|---|
| **Purpose** | The entry point to the data: browse and select a storm, see it in context |
| **Components** | Full-bleed `CycloneMap` · glass filter panel (season, basin, category, landfall) · storm list synced to the map · hover preview card · selected-storm summary |
| **Data** | `GET /cyclones` (filtered, paged) · `GET /tracks/bulk` |
| **API deps** | `/cyclones`, `/tracks/bulk` |
| **Interactions** | Filter → map and list update together · hover a row → highlight the track · click → open Analysis · map bounds can drive the filter |
| **MVP** | ✅ Required |

### 5.3 `/analysis/[sid]` — Cyclone Analysis

| | |
|---|---|
| **Purpose** | Understand one storm in depth: what happened, what the models saw, how they classified it |
| **Components** | `StormHeader` · `CycloneMap` with the full observed track · `TimeScrubber` · `IRFrame` + `FrameFilmstrip` · `IntensityTimeline` · `ClassProbabilityBar` · `DetectionStrip` · lifecycle ribbon (Tier-C derived stage, **explicitly badged "derived"**) · `GeminiPanel` (storm summary) |
| **Data** | `/cyclones/{sid}`, `/cyclones/{sid}/observations`, `/tracks/{sid}`, `/classification/{sid}`, `/detection/{sid}`, `POST /explain/storm` |
| **Interactions** | Scrub time → map marker, IR frame, readouts, and charts all move together · click a chart point to jump · toggle layers · expand the evidence drawer |
| **MVP** | ✅ Required |

### 5.4 `/predict/[sid]` — Prediction Dashboard

| | |
|---|---|
| **Purpose** | **The core demonstration.** At time T, show what the models forecast, what actually happened, and the error between them |
| **Components** | `CycloneMap` with five layers (observed · current · predicted · actual future · uncertainty cone) · `TimeScrubber` positioned as the forecast origin · `ForecastTable` (per-horizon predicted vs actual vs error in km and kt) · `IntensityTimeline` with the forecast overlaid and its error band · `ErrorGrowthChart` vs baselines · `ModelSwitcher` · `ModelBadge` · `GeminiPanel` (forecast explanation) |
| **Data** | `/prediction/{sid}?t=`, `/tracks/{sid}?t=`, `/prediction/{sid}/series`, `/analytics/error-by-leadtime`, `POST /explain/forecast` |
| **Interactions** | Move the origin time → the whole forecast recomputes from stored rows · switch model version → layers re-render for comparison · toggle "hide the actual future" for a blind-forecast presentation mode · hover a prediction point → error tooltip |
| **MVP** | ✅ Required — this page is the project |

> **`TimeScrubber` is the single most important interaction in the product.** Every view is a
> function of `(sid, t)`. It must be keyboard-accessible (arrow keys step, Home/End jump), show data
> availability along the track, and drive URL state so any moment is linkable.

### 5.5 `/explorer` — Historical Explorer

| | |
|---|---|
| **Purpose** | Aggregate view across seasons: patterns, climatology, comparison |
| **Components** | Season selector · multi-track map (`/tracks/bulk`) · storms-per-season and category-distribution charts · sortable/filterable table · storm-vs-storm comparison |
| **Data** | `/analytics/season-summary`, `/cyclones`, `/tracks/bulk`, optionally `POST /explain/compare` |
| **Interactions** | Brush a season range · select storms to compare · click through to Analysis |
| **MVP** | ⚪ **Reduced scope** — ship the table + multi-track map. Defer the comparison view if time is short |

### 5.6 `/methodology` — Methodology & About

| | |
|---|---|
| **Purpose** | **The honesty page, and a required deliverable.** States what the system does, what data it used, how well it performs, and what it cannot do |
| **Components** | Architecture diagram · dataset descriptions with attribution and licensing · **the live benchmark table** from `/analytics/model-performance` (every model, every metric, with baselines) · split methodology and leakage-control statement · label-provenance section (which labels are dataset-derived vs heuristic) · known limitations · **prominent non-operational-use disclaimer** · team and stack |
| **Data** | `/analytics/model-performance`, `/analytics/dataset-summary`, `/meta` |
| **MVP** | ✅ **Required.** A system that reports uncertain predictions without documenting its limitations is not finished. This page is also what makes the work credible to a technical judge |

### 5.7 Page dependency map

```
   /  (landing)
   └──▶ /monitor ──▶ /analysis/[sid] ──▶ /predict/[sid]
            │                                  │
            └──▶ /explorer ────────────────────┘
   /methodology  ── linked from every page footer and from every ModelBadge
```

---

## 6. Motion Specification

| Surface | Motion | Budget |
|---|---|---|
| Landing hero | Globe rotation, staggered headline reveal, gradient drift | Full cinematic |
| Page transitions | Shared-element storm name, 480 ms fade+lift | Moderate |
| Panel entry | Stagger 40 ms, 240 ms fade+lift | Moderate |
| Map layers | Predicted track draws in over 600 ms; cone fades in | Purposeful — the draw *communicates* direction of forecast |
| Scrubbing | **No easing.** 1:1 with input | **None** — latency here reads as broken |
| Numeric readouts | Cross-fade only, no count-up | Minimal — count-up animations make values unreadable while settling |
| Charts | Animate on mount, not on data change | Minimal |

**Rules:** `prefers-reduced-motion` disables everything non-essential (globe rotation stops, reveals
become instant). Motion never delays access to data. Nothing important is behind a scroll animation.

---

## 7. Performance

| Risk | Mitigation |
|---|---|
| Globe blocks LCP | `next/dynamic` with `ssr:false`, lazy-mounted below the fold, static image fallback, `IntersectionObserver`-gated |
| Map libraries in the initial bundle | Dynamic import; the map only loads on pages that show one |
| Long tracks kill pan/zoom | **Server-side simplification** with a zoom-dependent tolerance; the client never simplifies |
| Many tracks on the explorer map | Cap the visible set, use canvas rendering, aggressively simplify at low zoom |
| Chart re-render on every scrub tick | Memoise series; scrubbing moves a cursor, it does not rebuild the chart |
| Gemini latency | Stream tokens; skeleton state; cached responses are near-instant |
| IR thumbnails | Pre-rendered 256² PNG/WebP, `next/image`, adjacent frames prefetched |
| 3D on low-end GPUs | Detect `deviceMemory`/WebGL support; fall back to the static image |

**Targets:** LCP < 2.5 s · CLS < 0.1 · TBT < 200 ms · 60 fps map interaction · initial JS < 250 kB
gzip excluding lazy chunks.

---

## 8. Accessibility

The design direction (dark, glass, heavy motion) has three well-known accessibility failure modes.
Each is addressed explicitly:

| Risk | Requirement |
|---|---|
| **Low contrast on glass surfaces** | Every text/background pair verified at **≥ 4.5:1** *over the actual blurred backdrop*, not over the nominal token. Glass panels get a solid low-opacity base layer beneath the blur to guarantee a floor |
| **Motion sensitivity** | `prefers-reduced-motion` fully honoured, and a persistent in-app motion toggle |
| **Colour as the only encoding** | The intensity ramp is always accompanied by a text label or numeric value. Predicted vs observed differs by **line style (dashed vs solid)** as well as colour — the map must be readable in greyscale |
| Keyboard access | Full keyboard path through the scrubber, storm selector, map layer toggles, and all charts |
| Screen readers | Charts carry text summaries; the map has an accessible tabular equivalent of the track |
| Focus visibility | Visible focus rings that survive the dark/glass treatment |

**Colour-blind verification of the intensity ramp is a Phase 2 gate**, not a final polish item — the
ramp is used everywhere and is expensive to change late.

---

## 9. Frontend Boundaries

**The frontend must not:**

- compute any meteorological quantity (distance, error, category, tendency) — all arrive precomputed;
- perform geodesic maths or reprojection — geometry arrives correct and simplified;
- hold or use the Gemini API key — it calls `/api/v1/explain/*`;
- display a prediction without its `model_version` and stated uncertainty;
- render a visual that implies data the system does not have (wind fields, rainfall, surge, damage);
- hard-code the classification class list — it comes from `/classification/classes`.

**The frontend must:**

- generate its API types from `contracts/openapi.json`;
- keep `(sid, t)` in the URL so every view is shareable;
- degrade gracefully when a prediction, image, or Gemini response is unavailable;
- surface the non-operational-use disclaimer where predictions are displayed, not only on About.
