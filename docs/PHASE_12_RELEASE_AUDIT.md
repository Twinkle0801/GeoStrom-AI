# Phase 12 — Final Hardening, Performance, Security & Release Readiness

## 1. Executive summary

Phase 12 audited the entire integrated system built across Phases 0-11 for
release readiness, then made a small number of narrow, measured-evidence
fixes: (1) a real 12.5s-class Gemini latency was addressed with a bounded,
process-local, TTL cache that stores only validated (`source="gemini"`)
responses; (2) a lightweight, process-local rate limiter now protects the
one Gemini-backed endpoint; (3) a genuine 500-error bug (naive-vs-aware
datetime comparison) was found and fixed in the observations endpoint;
(4) a genuine duplicate client-side network request on every `/predict/[sid]`
page load was found (via a live Lighthouse network trace, not guesswork)
and fixed. No ML code, model, dataset, frozen split, seed, or metric was
touched. No frontend visual/layout/library change was made. The frontend
was not redesigned.

## 2. Baseline commit

- Commit: `8517e3dc5436e42f1c9bb34eb75ac0959ed05974` ("Complete end-to-end
  integration", Phase 11's final commit).
- Branch: `master`.
- Working tree at Phase 12 start: clean (verified via `git status --short`).
- No TODO/FIXME/debug-flag/wildcard-CORS residue found anywhere in the
  repository at audit start (`grep` sweep across `backend/`, `frontend/`,
  excluding generated/vendor directories).

## 3. Changes implemented

| # | Change | File(s) | Why |
|---|---|---|---|
| 1 | Bounded, thread-safe, in-process TTL cache for validated Gemini explanations | `backend/app/gemini/cache.py` (new) | Avoid repeating a ~1.3-6.5s (measured; Phase 11 recorded ~12.5s on its own run) real Gemini call for evidence-identical requests |
| 2 | Process-local, per-IP fixed-window rate limiter | `backend/app/gemini/ratelimit.py` (new) | Protect Gemini quota from accidental request storms |
| 3 | Wired cache + rate limiter into the route; cache checked first (a hit is never rate-limited); only `source="gemini"` results are ever cached | `backend/app/api/v1/explain.py` | Task's explicit "do not break normal frontend usage" + "do not cache validation/transport failures" |
| 4 | New `Settings` fields for cache/limiter bounds (defaults: 128 entries / 1h TTL; 20 req / 60s) | `backend/app/core/config.py` | Configurable without code changes |
| 5 | `HTTPException.headers` now propagated by the RFC 7807 exception handler | `backend/app/main.py` | Needed for the 429 response's `Retry-After` header (a real, previously-latent gap: any future `HTTPException(headers=...)` would have been silently dropped) |
| 6 | Fixed a real 500-error: naive-vs-timezone-aware `datetime` comparison crashed `GET /cyclones/{sid}/observations?from=...&to=...` when the query value carried no UTC offset | `backend/app/api/v1/cyclones.py` | Found via the API robustness audit (task §11's "invalid date ranges"); naive values are now assumed UTC, per this project's UTC-only convention |
| 7 | Fixed a real, measured duplicate network request: `PredictWorkspace` re-fetched the track for the initial forecast origin on every mount, even though the server component already fetched and passed the same data as `initialTrack` | `frontend/components/storm/PredictWorkspace.tsx` | Found via a live Lighthouse network-request trace on `/predict/2015313N22289` (see §3 Performance) |
| 8 | Test coverage for all of the above | `backend/tests/test_gemini_cache.py`, `test_gemini_ratelimit.py`, `test_api_explain.py`, `test_api_cyclones.py`, `test_api_prediction.py` (new tests added), `frontend/components/storm/PredictWorkspace.test.tsx` (new) | Task §14 |
| 9 | Regenerated `contracts/openapi.json` and `frontend/lib/api-types.ts` | — | The 429 response is now part of the `/explain/forecast` contract |

Nothing else changed. No ML file, no frontend visual/layout file, no
dataset, split, seed, or metric was touched.

## 4. Performance

### 4.1 Backend read endpoints (unchanged from Phase 11 — re-measured, not re-optimized)

| Endpoint | Phase 11 | Phase 12 (re-measured) |
|---|---:|---:|
| `GET /cyclones?limit=50` | ~254 ms | ~268 ms |
| `GET /cyclones/{sid}` | ~255 ms | ~258 ms |
| `GET /prediction/{sid}/series` | ~253 ms | ~287 ms |
| `GET /analytics/model-performance` | ~222 ms | ~291 ms |
| `GET /tracks/{sid}` | (not separately measured) | ~260 ms |

Within normal single-machine run-to-run variance; no regression, no DB
change, no code change to these paths this phase. **PASS, unchanged.**

### 4.2 Gemini latency: cold call vs. cache hit (real API, real storm, this session)

| Call | Storm | Result |
|---|---|---:|
| Cold (guaranteed cache miss, first time this storm was ever explained) | `2015270N27291` | **6.505 s**, `source: "gemini"` |
| Immediate repeat (identical evidence, cache hit) | `2015270N27291` | **1.340 s**, `source: "gemini"`, byte-identical `explanation` text |

A ~5.2s (80%) latency reduction on the cache-hit path, confirmed not just
by timing but by the response's `explanation.summary` text being
byte-identical across three consecutive calls despite Gemini's nonzero
temperature (0.2) — proof the second and third calls never reached Gemini
at all. (Phase 11 recorded ~12.5s for its own cold call; real external API
latency varies run to run — both numbers are reported honestly rather than
reconciled.)

### 4.3 Frontend

Production build (`next build`) succeeded; route sizes:

| Route | Size | First Load JS |
|---|---:|---:|
| `/` | 42.7 kB | 152 kB |
| `/storms` | 2.91 kB | 112 kB |
| `/predict/[sid]` | 18.7 kB | 231 kB |
| `/models` | 10.4 kB | 219 kB |
| `/methodology` | 137 B | 106 kB |

Unchanged from before the fix (the duplicate-request fix removes a
network round trip, not JS bytes). Leaflet (`CycloneMapClient`) and the
Cobe globe (`HeroGlobeClient`) are already lazy-loaded via
`next/dynamic({ ssr: false })` — verified by direct code inspection,
predating this phase. Recharts (`IntensityChart`/`ModelComparisonChart`/
`ClassificationComparisonChart`) is loaded eagerly; left as-is, since it is
genuinely above-the-fold, immediately-visible content on every route that
uses it, not dead weight — dynamically importing visible core content
would trade one loading state for another with no demonstrated net
benefit, which the task's "do not optimize blindly" instructs against.

### 4.4 Lighthouse (real CLI run, `lighthouse@13.4.1`, headless Chrome, against `next start` production build, on real backend data for storm `2015313N22289`)

| Route | Performance | Accessibility | Best Practices | SEO |
|---|---:|---:|---:|---:|
| `/` | 0.80 | 0.95 | 0.96 | 1.00 |
| `/storms` | 0.99 | 0.96 | 0.96 | 1.00 |
| `/predict/2015313N22289` (before fix) | 0.65 | 0.95 | 0.96 | 1.00 |
| `/predict/2015313N22289` (after fix) | **0.70** | 0.95 | 0.96 | 1.00 |
| `/models` | 0.83 | 0.96 | 0.96 | 1.00 |
| `/methodology` | 0.99 | 0.96 | 0.96 | 1.00 |

`/predict/[sid]` before/after, directly attributable to the fix in §3.7:

| Metric | Before | After |
|---|---:|---:|
| Performance score | 0.65 | 0.70 |
| Largest Contentful Paint | 5.9 s | 4.9 s |
| Total Blocking Time | 500 ms | 450 ms |
| Time to Interactive | 5.9 s | 5.0 s |
| Total network requests | 43 | 42 |
| `GET /api/v1/tracks/{sid}` requests visible in the browser trace | **1** (the redundant client-side duplicate) | **0** |

`/predict/[sid]` remains the lowest-scoring route among the five audited
— expected and accepted: it is the one page carrying a real Leaflet map,
an intensity chart, a model selector, a satellite panel, a classification
panel, and the Gemini panel simultaneously, none of which were removed or
simplified (explicitly forbidden this phase). No further frontend change
was made without additional measured justification, per the task's "do
not optimize blindly."

Accessibility/Best-Practices/SEO scores are all comfortably high (0.95+)
across every route; no fix was made in this area, as none of the flagged
items (all minor, e.g. a handful of low-severity color-contrast notes on
translucent glass-panel text) rise to a correctness or accessibility
defect, and fixing them would mean touching the visual design system,
explicitly forbidden this phase. Documented, not silently dropped.

## 5. Gemini hardening

### 5.1 Caching

- **Design**: `backend/app/gemini/cache.py`. Bounded (default 128 entries),
  LRU + TTL (default 1 hour), thread-safe (a single lock guards every
  access — FastAPI runs sync path operations across a worker-thread pool).
- **Cache key**: SHA-256 of the *entire* evidence packet content (storm,
  origin timestamp, resolved model name/version for both tasks, observed
  history, forecast values, evidence-schema version), excluding only
  `generated_at`. This is a deliberate over-approximation of the task's
  "at minimum consider storm_id/origin/model/version/schema-version" list:
  any change to any evidence field produces a different key, so a hit is
  only ever returned for evidence that is byte-identical (module the
  timestamp) to what was previously validated.
- **What is cached**: only a `GeminiExplanationService` result whose
  `source == "gemini"` (i.e. it already passed `validate_grounding`).
  A `"fallback"` result (`timeout`, `api_error`, `malformed_json`,
  `ungrounded_claim`, `not_configured`) is **never** cached — enforced in
  the route (`if result.source == "gemini": cache.set(...)`), and covered
  by `test_fallback_results_are_never_cached_each_request_calls_gemini_again`.
- **Restart semantics**: plain in-process dict — cleared on every backend
  restart, not shared across multiple worker processes if the app is ever
  run with more than one uvicorn/gunicorn worker. Documented in the
  module's own docstring. Accepted for this single-process retrospective
  research prototype; introducing Redis purely for this would be new
  infrastructure disproportionate to the project's actual traffic.

### 5.2 Rate limiting

- **Design**: `backend/app/gemini/ratelimit.py`. Fixed window, per client
  IP (`request.client.host`) — there is no user identity in this
  no-auth architecture, so IP is the only available key. Default: 20
  requests / 60 s. Process-local counters, reset on restart, not shared
  across multiple worker processes — documented, not oversold, in the
  module's own docstring.
- **Interaction with the cache**: the cache is checked *first*. A hit
  never reaches the rate limiter or Gemini at all — repeated views of an
  already-explained forecast stay free, per the task's "do not break
  normal frontend usage." Only a genuine cache-miss (an attempted real
  Gemini call) is rate-limited. Covered by
  `test_cache_hit_bypasses_the_rate_limiter_entirely`.
  `test_rate_limiter_reset_allows_requests_again` covers the reset path.
- **Response**: `429` with a `Retry-After` header (required fixing the
  RFC 7807 exception handler to propagate `HTTPException.headers`, which
  it silently dropped before — see §3 item 5).

### 5.3 Grounding regression

Re-ran the full existing grounding/validator/adversarial suite
(`test_gemini_validator.py`, `test_gemini_service.py`,
`test_gemini_evidence_builder.py`, `test_gemini_fallback.py`,
`test_integration_evidence_chain.py`) — **all still pass, unchanged**.
This suite already covers, verbatim, every category the Phase 12 task
lists (unsupported wind/latitude/classification/model/confidence/
forecast-horizon claims, prompt-injection-shaped evidence text, negation
cases on both sides of a forbidden term, negative-coordinate sign
handling, categorical claim validation) — a genuinely comprehensive
inheritance from Phase 9, not a gap. No new grounding test was added,
since none of the task's named categories were actually uncovered; adding
a redundant test purely to hit a count would violate "do not create
meaningless tests just to increase test count." **PASS, no change
required, explicitly documented rather than silently skipped.**

## 6. Security

Full sweep across `backend/`, `frontend/`, `ml/` (excluding
`node_modules`/`__pycache__`/build output):

| Check | Result |
|---|---|
| Hardcoded `GEMINI_API_KEY`/`GOOGLE_API_KEY`/password/secret/token | None found |
| `debug=True` / Django-style `DEBUG = True` | None found (FastAPI app has no debug mode) |
| Wildcard CORS (`allow_origins=["*"]`) | None — `settings.allowed_origins` is an explicit localhost allowlist |
| `.env`/`.env.local` tracked in git | No — only `.env.example` templates are tracked |
| Gemini SDK/API key/system prompt/EvidencePacket-construction logic in the frontend bundle | None — confirmed by both source grep and a full production build inspection |
| Raw SQL / string-built queries | None beyond one parameterless `SELECT 1` health check; every repository query is SQLAlchemy ORM with bound parameters (verified again this phase, including the `q` substring filter, which uses `.ilike()`, not string concatenation) |
| Path traversal / unsafe subprocess calls | None — `sid` path/body parameters flow only into `db.get(Storm, sid)`/ORM `WHERE` clauses, never a filesystem path or shell command; verified live with adversarial `sid` values (`../../etc/passwd`, a SQL-comment-shaped string, a 500-char string, emoji) — all correctly 404, never a 500 |
| Prompt injection | Already covered end-to-end by `TestPromptInjectionLikeEvidence` (§5.3) — the deterministic validator is the backstop regardless of prompt-level defenses |
| Secrets in logs | `GeminiExplanationService._log_safe` logs only status/category fields, never the raw response or API key (unchanged, re-verified) |

No new secret-handling code was introduced this phase; the cache stores
only Gemini's validated structured explanation object, never a request
header, API key, or raw HTTP response.

## 7. Scientific integrity

Re-verified, not re-litigated (no ML code was touched, so this is a
confirmation pass, not a new evaluation):

- Frozen split (`splits_v1.json`) unchanged; no split-generation code was
  touched.
- Model registry tiering unchanged: LightGBM (intensity), CLIPER-style
  Ridge (track), Logistic Regression (classification) remain
  `tier="baseline"`/`is_recommended=true`; every GRU/CNN variant remains
  `tier="exploratory"`, `is_recommended=false` — reconfirmed live via
  `GET /api/v1/analytics/model-performance` this phase.
- Observed vs. predicted separation: unchanged code path; re-covered by
  the pre-existing `test_integration_evidence_chain.py` (Phase 11).
- UTC-only timestamps, kt/km units, `[lon, lat]` GeoJSON order,
  antimeridian handling: unchanged; no code in these paths was touched
  this phase (the one date-handling fix in §3 item 6 concerns query-
  parameter parsing, not any stored or computed scientific value).

**PASS across the board — no ML code was modified, so no ML conclusion
could have changed.**

## 8. Satellite status

Unchanged from Phase 11: pipeline exists and is real (12 storms / 627
fused samples), 531/547 frozen-split storms have HURSAT archive coverage,
no frame-serving endpoint exists. Confirmed still true via a live OpenAPI
path listing this phase (still 11 paths, no satellite route). **No
endpoint was added.** No fake imagery, no invented per-storm result.

## 9. Classification status

Unchanged from Phase 11: frozen `scene_taxonomy_v1` and offline
baseline/CNN results exist; no `classifications` table or per-storm
serving endpoint exists. Confirmed still true this phase. **No migration,
no fake row, no invented classification result was added.**

## 10. Tests

| Suite | Result |
|---|---:|
| Backend (`pytest`) | **206 passed / 0 failed** (178 Phase 11 baseline + 28 new: 9 cache unit tests, 5 rate-limiter unit tests, 5 cache/rate-limit API-interaction tests, 9 API-robustness edge-case tests) |
| Frontend (`vitest`) | **47 passed / 0 failed** (44 Phase 11 baseline + 3 new `PredictWorkspace.test.tsx`) |
| ML fast suite | **341 passed / 0 failed** (unchanged — no ML code touched) |
| ML satellite integration (real-data) | see final regression, §16 below |
| TypeScript (`tsc --noEmit`) | 0 errors |
| ESLint | 0 errors |
| Frontend production build (`next build`) | Succeeds |

## 11. Known limitations

- The Gemini cache and rate limiter are **process-local**: they reset on
  every backend restart and are not shared if the app is ever run with
  more than one worker process. Acceptable for this single-process
  research prototype; would need to change (e.g. Redis-backed) before any
  multi-worker deployment.
- No headless-browser tool was available for interactive (click/drag)
  capture; Lighthouse (a real, network-driven headless-Chrome tool) *was*
  available this phase and was used for the performance/accessibility
  audit, closing part of the Phase 11 gap, but manual TimeScrubber
  drag-interaction capture still relies on the existing component tests
  plus code review, not a captured live session.
- Backend/frontend/Gemini latency numbers are real, single-machine,
  single-run measurements on a Windows development workstation, not a
  load-tested or multi-run statistical baseline — reported as such, not
  as a guaranteed SLA.
- Recharts remains eagerly bundled on chart-bearing routes (§4.3) — a
  documented, deliberate non-change, not an oversight.

## 12. Deployment/readiness assessment

The backend, frontend, and their contract are internally consistent, pass
their full test suites, contain no known secret leakage, and gracefully
degrade under Gemini failure (timeout/API error/malformed output/
ungrounded claim all still return `200` with a deterministic fallback,
never a `500`). The one real crash bug found this phase (naive-datetime
comparison) is fixed and regression-tested. The one real duplicate-request
bug found this phase is fixed and regression-tested, with before/after
Lighthouse evidence. Satellite/classification gaps remain honestly
unimplemented rather than fabricated. The system is ready for the planned
final frontend visual/UX refinement phase; no further backend hardening
is blocking that work.

## 13. Recommended next step

Proceed to the final frontend visual/UX refinement phase. Suggested
follow-ups for a later, separate infrastructure phase (not urgent, not
blocking): if the app is ever deployed with multiple worker processes,
replace the in-process Gemini cache/rate-limiter with a shared store
(Redis or equivalent); consider adding automated Lighthouse CI now that a
working invocation recipe exists (`npx lighthouse@13.4.1 <url> --output=json
--chrome-flags="--headless"`, run against `next start`, not `next dev`).
