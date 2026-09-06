# Phase 11 — End-to-End Test

## Tested environment

- OS: Windows 11, PowerShell + Git Bash.
- Backend: `cd backend && python -m uvicorn app.main:app --port 8001` (dev mode, no reload).
- Frontend: `cd frontend && npm run dev` (`next dev -p 3001`).
- Database: `geostrom_db` (Docker, `postgis/postgis:16-3.4`), restarted via Docker Desktop this
  session (a recurring environment characteristic of this workstation, not an application defect
  — the container itself was healthy and unmodified once running).
- Frontend `.env.local`: `NEXT_PUBLIC_API_BASE_URL=http://localhost:8001` (unchanged).
- Backend `.env`: real local `DATABASE_URL` + a real local `GEMINI_API_KEY` (already present from
  Phase 9/10 setup; never printed, never committed).

## Database state

88 storms in `storms` (all `split='test'`, confirmed to match the frozen `splits_v1.json`
manifest exactly — 0 mismatches, checked live via a direct query against both the DB and the
manifest in the same process). 216 prediction rows for the storm used in this test (9 real
forecast origins × 4 horizons × 6 model/task combinations).

## Real storm used

**`2015313N22289`** — season 2015, North Atlantic, split=test, `has_predictions=true`, real
predictions from `intensity_{persistence,ridge,lightgbm}` and `track_{persistence,cliper,
lightgbm}`.

## API checks (all against the running backend, port 8001)

| Endpoint | Result |
|---|---|
| `GET /health` | `{"status":"ok","database":"ok"}` |
| `GET /api/v1/cyclones?limit=1` | 200, timestamps serialize with explicit `Z` (UTC), never naive |
| `GET /api/v1/cyclones/2015313N22289` | 200, `has_predictions: true`, real `bbox` |
| `GET /api/v1/cyclones/NOPE` | 404, RFC 7807 problem+json body |
| `GET /api/v1/tracks/2015313N22289` | 200, 29 features, `[lon, lat]` order confirmed (`[-75.4, 29.5]`) |
| `GET /api/v1/prediction/2015313N22289/series` | 200, 216 rows, 2 tasks, 6 model/version combos, 9 distinct origins |
| `GET /api/v1/analytics/model-performance` | 200, LightGBM/CLIPER-style Ridge/Logistic Regression all correctly `is_recommended: true`; every GRU/CNN row `tier: "exploratory"` |
| `POST /api/v1/explain/forecast {"sid":"2015313N22289"}` | 200, `source: "gemini"` (real API call succeeded and passed grounding), correct model identity, real evidence content |
| `POST /api/v1/explain/forecast {"sid":"NOPE"}` | 404 |
| `POST /api/v1/explain/forecast {}` | 422 (missing required `sid`) |
| `OPTIONS /api/v1/cyclones` (CORS preflight, `Origin: http://localhost:3001`) | `access-control-allow-origin: http://localhost:3001`, methods `GET, POST` — not a wildcard |

## Frontend journey (real storm, real backend, curl-driven SSR verification)

| Step | Route | Result |
|---|---|---|
| HOME | `/` | 200 |
| STORM EXPLORER | `/storms` | 200 |
| SELECT REAL STORM → STORM ANALYSIS | `/predict/2015313N22289` | 200; page HTML contains `Historical Analysis`, `Observed timestamp` (TimeScrubber), `Intensity model`/`Track model` (ModelSelector), `AI Analysis` (GeminiPanel), `2015313N22289` |
| LOAD OBSERVED TRACK / LOAD PREDICTIONS | (same page, server-fetched) | Confirmed via the same 200 response — `PredictWorkspace` receives real `observations`/`predictionSeries`/`initialTrack` server-side |
| SELECT MODEL / SCRUB TIME / UPDATE MAP / UPDATE INTENSITY CHART | Client-side interactivity | Verified via the 44 passing component tests (`TimeScrubber.test.tsx`, `ModelSelector.test.tsx`, `IntensityChart.test.tsx`) plus direct code review — **no headless-browser tool was available in this environment**, so live drag/click interaction was not captured as video/screenshot; stated plainly rather than overclaimed |
| OPEN GEMINI / RECEIVE EXPLANATION | `POST /api/v1/explain/forecast` | Real call, `source: "gemini"`, ~12.5s wall-clock (see Performance Baseline) |
| OPEN EVIDENCE DRAWER | (client-side, same response) | `evidence` field present and correctly populated (verified directly via curl on the raw JSON; drawer rendering verified via `EvidenceDrawer.test.tsx`) |
| `/models` | Model Performance page | 200 |
| `/methodology` | Methodology page | 200 |

## Gemini end-to-end test

Traced: stored DB rows → `evidence_builder.py` → `GeminiExplanationService` → real Gemini API →
`validate_grounding` → `ExplainResponse` → (would flow to) `GeminiPanel`/`EvidenceDrawer`. New
integration test (`test_integration_evidence_chain.py`, 5 tests) asserts the **exact seeded
numeric values** (a deliberately unusual `pred_wind_kt=92.4`, `pred_lat`/`pred_lon`) survive this
entire chain unchanged, for both the fallback path and a real (mocked) Gemini-success path, and
that `current_state` is sourced from `Observation` rows while `forecasts` are sourced from
`Prediction` rows — never confused. A real, non-mocked Gemini call was also made directly against
the live API (above) and returned `source: "gemini"` with zero grounding violations.

## Failure-mode tests (task §17, all pre-existing + re-verified this phase)

| # | Scenario | Test | Result |
|---|---|---|---|
| 1 | Gemini success | `test_gemini_service.py::test_1_valid_gemini_response...` | PASS |
| 2 | Gemini timeout | `...test_11_timeout_falls_back_without_retry` | PASS |
| 3 | Gemini unavailable | `...test_12_api_exception_falls_back` | PASS |
| 4 | Invalid Gemini output | `...test_2_malformed_json_falls_back_after_retry`, `test_3_schema_validation_failure_falls_back` | PASS |
| 5 | Unsupported numeric claim | `...test_4_unsupported_number_triggers_fallback` | PASS |
| 6 | Unsupported model claim | `...test_7_unsupported_model_name_triggers_fallback` | PASS |
| 7 | Unsupported classification claim | `...test_6_unsupported_classification_triggers_fallback` | PASS |
| 8 | Malformed response | (same as #4) | PASS |
| 9 | Empty/missing evidence packet | `...test_10_missing_evidence_still_produces_a_safe_fallback` | PASS |
| 10 | Frontend API failure | `GeminiPanel.test.tsx::shows an error state with a retry option...` | PASS |

## Regression tests (exact counts, this run)

| Suite | Result |
|---|---|
| Backend | **178 passed / 0 failed** (141 Phase 10 baseline + 37 new Phase 11 tests) |
| Frontend | **44 passed / 0 failed** (41 Phase 10 baseline + 3 new Phase 11 tests) |
| ML fast suite | **341 passed / 0 failed** (unchanged — no ML code touched) |
| ML satellite integration (real-data, ~23 min) | **11 passed / 0 failed** (unchanged) |
| TypeScript | 0 errors |
| ESLint | 0 errors |

No random seed, evaluation split, or ML metric was changed. `Storm.split` in the database was
re-verified live against the frozen `splits_v1.json` manifest: 88/88 storms match, 0 mismatches.

## Performance baseline (recorded, not optimized — Lighthouse deferred to Phase 12)

| Endpoint/page | Observed latency |
|---|---:|
| `GET /api/v1/cyclones?limit=50` | ~254 ms |
| `GET /api/v1/cyclones/{sid}` | ~255 ms |
| `GET /api/v1/prediction/{sid}/series` | ~253 ms |
| `GET /api/v1/analytics/model-performance` | ~222 ms |
| `POST /api/v1/explain/forecast` (real Gemini call) | **~12.5 s** — the one clear bottleneck, dominated by the external Gemini API round-trip, not backend code |
| Frontend `/` (SSR) | ~550 ms |
| Frontend `/predict/{sid}` (SSR, 4 parallel backend calls) | ~1.02 s |

**Obvious bottleneck identified, not fixed (out of scope per task §28)**: the real Gemini call
latency (~12.5 s) is the single largest contributor to a user's "Generate explanation" wait time.
No premature optimization (caching, streaming) was attempted this phase — both are already
documented as deferred in `docs/PHASE_9_GEMINI_INTEGRATION.md` §17.

## Known limitations

- No headless-browser/screenshot tool was available; client-side interactivity (drag, click,
  live re-render) was verified through passing component tests and code review, not a captured
  browser session.
- The satellite/classification panels remain BLOCKED BY EXISTING CONTRACT (§14/§15 of the task) —
  confirmed via a live OpenAPI path listing (11 real paths, no satellite/classification route
  exists) rather than assumed from Phase 10's documentation.
- Antimeridian handling has no real North Atlantic storm to exercise it end-to-end (the basin
  never crosses ±180°); verified instead with a synthetic fixture at the backend geometry layer
  (`test_antimeridian_crossing_coordinates_pass_through_unmangled`) plus the pre-existing 26
  passing `ml/tests/test_geo.py` cases at the computation layer.
