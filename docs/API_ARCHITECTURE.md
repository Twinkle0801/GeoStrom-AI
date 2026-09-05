# API ARCHITECTURE — GeoStrom AI

**Phase:** 0 (Architecture) · **Status:** Complete · No endpoint has been implemented.

---

## 1. Stack and Rationale

| Component | Choice | Why |
|---|---|---|
| Framework | **FastAPI** | Async, and — decisively — it generates the OpenAPI schema from Pydantic models, which generates the frontend's TypeScript types. The contract cannot silently drift |
| Validation | **Pydantic v2** | Response models are the API contract; v2 is substantially faster at serialisation, which matters for GeoJSON payloads |
| ORM | **SQLAlchemy 2.0** (typed) | Mature PostGIS support via GeoAlchemy2; the repository pattern lets Phase 1 run on SQLite before Postgres exists |
| Migrations | **Alembic** | Schema will change as dataset verification lands |
| Server | **Uvicorn** behind the platform's proxy | Standard |
| Inference | **ONNX Runtime** — *reserved, not used in MVP* | Predictions are precomputed. The dependency is present so live inference is a small change later |
| Cache | **In-process TTL cache**, Redis only if measured to be needed | Data is static and small. Adding Redis for a read-only dataset would be complexity without benefit |

### 1.1 The API's job

**It transports precomputed results.** It performs no meteorological computation. Every scientific
number originates in `ml/` and is stored; the API selects, shapes, and serialises. The single
exception is geometry assembly (GeoJSON construction, simplification, cone polygons), which is a
presentation concern that belongs server-side so all clients get identical geometry.

---

## 2. Conventions

| Aspect | Rule |
|---|---|
| Base path | `/api/v1` — versioned from day one |
| Time | ISO 8601 UTC with `Z`. No local time anywhere |
| Coordinates | WGS84, `lon, lat` order in GeoJSON (per spec), `lat, lon` in scalar JSON fields (named explicitly to avoid confusion) |
| Units | Declared in field names or the response envelope: `wind_kt`, `pressure_hpa`, `error_km` |
| Pagination | `limit` / `offset` with a `total` count; default 50, max 500 |
| Errors | RFC 7807 problem+json: `type`, `title`, `status`, `detail`, `instance` |
| Model attribution | Every model-derived payload carries `model_version` |
| Nulls | A missing measurement is `null`, never `0`, never `-999` |
| Caching | `Cache-Control` + `ETag`; historical data is immutable, so long TTLs are safe |

---

## 3. API Groups

### 3.1 `/api/v1/cyclones` — storm catalogue

| Endpoint | Purpose | Input | Output | MVP |
|---|---|---|---|---|
| `GET /cyclones` | List/search storms for the explorer and pickers | `season`, `basin`, `name`, `min_category`, `made_landfall`, `q`, `limit`, `offset`, `sort` | Paged storm summaries: `sid`, `name`, `season`, `basin`, `start/end`, `max_wind_kt`, `min_pressure_hpa`, `max_category`, `n_observations`, `split` | ✅ **Required** |
| `GET /cyclones/{sid}` | Full metadata for one storm | `sid` | Summary + lifecycle bounds + availability flags (`has_imagery`, `has_predictions`) + bbox | ✅ **Required** |
| `GET /cyclones/{sid}/observations` | The raw best-track time series for charts and the timeline scrubber | `sid`, `from`, `to`, `synoptic_only` | Array of observations: `ts`, `lat`, `lon`, `wind_kt`, `pressure_hpa`, `category`, `nature`, `image_key` | ✅ **Required** |
| `GET /cyclones/seasons` | Facets for filter UI | — | Seasons with storm counts | ⚪ Nice to have |

### 3.2 `/api/v1/tracks` — geospatial payloads

| Endpoint | Purpose | Input | Output | MVP |
|---|---|---|---|---|
| `GET /tracks/{sid}` | **The map's primary payload.** One GeoJSON FeatureCollection containing observed track, current position, actual future, predicted track, prediction points, and the uncertainty cone | `sid`, `t` (origin time), `simplify` (zoom-derived tolerance), `include` (comma list of feature kinds) | `FeatureCollection` per SYSTEM_ARCHITECTURE.md §6.2 | ✅ **Required** |
| `GET /tracks/bulk` | Many tracks at once for a season-overview map | `season`, `basin`, `simplify` | FeatureCollection of simplified LineStrings | ⚪ Nice to have |

**Design note:** one composite endpoint rather than five small ones. The map needs all layers
simultaneously; five round trips would produce visible staggered rendering. `include` keeps it
flexible without fragmenting the contract.

### 3.3 `/api/v1/detection`

| Endpoint | Purpose | Input | Output | MVP |
|---|---|---|---|---|
| `GET /detection/{sid}` | Detection probability across a storm's lifetime (for the timeline strip) | `sid`, `model_version?` | Array of `{ts, probability, is_positive, model_version}` | ✅ **Required** |
| `GET /detection/{sid}/{ts}` | Single-frame result | `sid`, `ts` | Probability + calibrated confidence + thumbnail URL | ✅ **Required** |
| `POST /detection/analyze` | Run detection on an **uploaded** image | multipart image | Probability + confidence | ❌ **Advanced.** Requires live ONNX inference and an unclear preprocessing contract for arbitrary user imagery — a false-confidence risk. Deliberately deferred |

### 3.4 `/api/v1/classification`

| Endpoint | Purpose | Input | Output | MVP |
|---|---|---|---|---|
| `GET /classification/{sid}/{ts}` | Class for one frame | `sid`, `ts`, `model_version?` | `predicted_class`, `confidence`, full `probabilities`, `true_class`, `model_version` | ✅ **Required** |
| `GET /classification/{sid}` | Class sequence over the storm's life — drives the lifecycle ribbon | `sid` | Array of the above | ✅ **Required** |
| `GET /classification/classes` | The active label set, **served from config, not hard-coded** | — | Class names, ordering, descriptions, ordinal flag, colours | ✅ **Required** — this is what allows the class list to change after dataset verification without a frontend release |

### 3.5 `/api/v1/prediction`

| Endpoint | Purpose | Input | Output | MVP |
|---|---|---|---|---|
| `GET /prediction/{sid}` | Forecast issued at origin time `t` | `sid`, `t`, `model_version?` | Per lead time: `lead_hours`, `valid_ts`, `pred_lat/lon`, `pred_wind_kt`, `error_radius_km`, `true_*`, `track_error_km`, `wind_error_kt` | ✅ **Required** |
| `GET /prediction/{sid}/series` | Every forecast issued across the storm's life — powers the error-growth chart | `sid`, `lead_hours?` | Array of prediction records | ✅ **Required** |
| `GET /prediction/compare` | Same storm/time under multiple model versions — the model-comparison view | `sid`, `t`, `models` | Grouped predictions with per-model error | ⚪ **Strongly recommended** — it is the visible payoff of the swappable-model design and is nearly free once predictions are stored with `model_id` |

### 3.6 `/api/v1/analytics`

| Endpoint | Purpose | Input | Output | MVP |
|---|---|---|---|---|
| `GET /analytics/model-performance` | The benchmark table: every model, every metric, on the frozen test split | `task?` | Per model: metrics, skill vs baseline, error by lead time, dataset build, trained-at | ✅ **Required** — this is the methodology page's core content and the project's honesty guarantee |
| `GET /analytics/error-by-leadtime` | Error growth curves | `model_version?` | Mean/median/percentile error per horizon | ✅ **Required** |
| `GET /analytics/season-summary` | Storm counts, ACE-style aggregates, category distribution per season | `basin?`, `from`, `to` | Aggregates | ⚪ Nice to have |
| `GET /analytics/basin-climatology` | Spatial density of tracks | `basin`, `season_range` | Gridded density GeoJSON | ❌ Advanced |
| `GET /analytics/dataset-summary` | Dataset build stats, class balance, split sizes, QC report | — | Manifest summary | ⚪ Nice to have — high transparency value, cheap |

### 3.7 `/api/v1/explain` — Gemini (see §6–8)

| Endpoint | Purpose | Input | Output | MVP |
|---|---|---|---|---|
| `POST /explain/forecast` | Natural-language explanation of one stored forecast | `sid`, `t`, `model_version?`, `audience` | Streamed text + the evidence packet used + validation status | ✅ **Required** |
| `POST /explain/storm` | Narrative summary of a storm's full lifecycle | `sid` | Streamed text + evidence packet | ✅ **Required** |
| `POST /explain/compare` | Compare two storms using stored values only | `sid_a`, `sid_b` | Streamed text + evidence packet | ⚪ Nice to have |
| `POST /explain/ask` | Constrained Q&A over the evidence packet for the current view | `sid`, `t`, `question` | Streamed text, or an explicit refusal | ⚪ **Guarded.** Highest hallucination surface; ship only if §8 guardrails are proven |

### 3.8 System

| Endpoint | Purpose | MVP |
|---|---|---|
| `GET /health` | Liveness + DB connectivity | ✅ Required |
| `GET /meta` | Active model versions, dataset build, API version | ✅ Required |
| `GET /openapi.json` | Contract source for frontend type generation | ✅ Required |

---

## 4. Cross-Cutting Concerns

| Concern | Approach |
|---|---|
| **CORS** | Explicit origin allowlist from env. No wildcard in production |
| **Rate limiting** | Only on `/explain/*` — the only endpoint with a per-call cost. Per-IP token bucket, plus a global daily cap so a runaway client cannot exhaust the Gemini quota |
| **Caching** | Historical results are immutable → long `Cache-Control` + `ETag` on all read endpoints. `/explain/*` responses cached by `(sid, t, scope, model_version, audience)` hash, which typically removes most Gemini calls during a demo |
| **Auth** | **None for MVP.** All data is public and read-only. Adding auth would be complexity without a threat it mitigates. `/explain/*` is protected by rate limiting rather than identity |
| **Observability** | Structured JSON logs with a request id; log Gemini token usage and guardrail outcomes explicitly |
| **Errors** | Never leak a stack trace. Unknown `sid` → 404 with a problem document; unavailable prediction → 404 with a `detail` explaining *why* (e.g. "no forecast issued at this time — fewer than 8 prior observations") |

---

## 5. Contract Generation

```
Pydantic v2 models  ──▶  FastAPI /openapi.json  ──▶  contracts/openapi.json
                                                          │
                                                openapi-typescript
                                                          ▼
                                            frontend/lib/api-types.ts  (generated)
```

**Rule: `api-types.ts` is generated and never hand-edited.** A backend field rename becomes a
frontend TypeScript error at build time instead of `undefined` on a dashboard during a demo.
Regeneration is a CI step and a Makefile target.

---

## 6. Gemini Architecture

### 6.1 The boundary — stated as a hard rule

```
   ┌─────────────────────────────────────────────────────────────┐
   │  ML MODELS  produce every number.                           │
   │  GEMINI     produces only sentences about those numbers.    │
   └─────────────────────────────────────────────────────────────┘

   Detection · Classification · Intensity · Track
                      │
                      ▼   (stored rows only)
            EVIDENCE PACKET BUILDER
                      │   structured JSON, backend-assembled
                      ▼
                   GEMINI
                      │   narrative text
                      ▼
             GUARDRAIL VALIDATOR
                      │   every number checked against the packet
                      ▼
                     UI
```

**Gemini never:** forecasts a track, estimates an intensity, classifies an image, queries the
database, executes code, or receives a question without an accompanying evidence packet.
**Gemini only:** explains, summarises, contextualises, and translates numbers into prose.

Architecturally, this is enforced because **Gemini has no access to anything except the JSON packet
the backend constructs.** It is not given database credentials, tools, or retrieval. The restriction
is structural, not merely instructed — a prompt rule can be talked around; a missing capability cannot.

### 6.2 Where it runs — **backend only**

| Consideration | Backend | Frontend |
|---|---|---|
| API key exposure | Server env var, never shipped | **Any key in a browser bundle is public.** `NEXT_PUBLIC_*` is readable by anyone |
| Evidence grounding | Backend already holds the verified rows | Client would have to be trusted to assemble facts — trivially tampered with |
| Guardrail validation | Runs before the user sees text | Client-side validation is bypassable |
| Rate limiting / cost control | Enforceable | Not enforceable |
| Caching | Shared across users | Per-browser only |
| Prompt confidentiality | Server-side | Visible in the bundle |

**DECISION: all Gemini calls originate in `backend/app/gemini/`.** The frontend calls
`POST /api/v1/explain/*` and renders a stream. It never holds a key and never talks to Gemini
directly. There is no MVP scenario where a frontend call is the right choice.

### 6.3 Secrets handling

- `GEMINI_API_KEY` lives only in the deployment platform's secret store and in a local `.env`.
- `.env` is git-ignored; `.env.example` contains **names only, never values**.
- Startup fails loudly if the key is missing rather than silently degrading.
- The key never appears in logs, error responses, or the OpenAPI schema.
- A separate, low-quota key is used for local development.

---

## 7. Evidence Packet Design

The packet is the **entire** universe of facts available to the model for a given call. It is built
from the same stored rows the UI is displaying, so text and visuals cannot disagree.

```jsonc
{
  "schema_version": "1.0",
  "generated_at": "…",
  "storm": {
    "sid": "…", "name": "…", "season": 2005, "basin": "North Atlantic",
    "lifecycle": { "start": "…", "end": "…", "n_observations": 62 }
  },
  "current_state": {
    "timestamp": "…", "lat": 25.1, "lon": -87.6,
    "wind_kt": 95, "pressure_hpa": 948,
    "category": 2, "nature": "TS",
    "motion": { "speed_kt": 11, "direction_deg": 315 },
    "dist2land_km": 340
  },
  "recent_history": [ { "timestamp": "…", "wind_kt": 80, "lat": …, "lon": … } ],
  "model_outputs": {
    "classification": { "class": "Category 2", "confidence": 0.81,
                        "model_version": "cls_resnet18_v1" },
    "detection":      { "probability": 0.97, "model_version": "det_resnet18_v1" },
    "forecast": [
      { "lead_hours": 6,  "pred_lat": …, "pred_lon": …, "pred_wind_kt": 101,
        "error_radius_km": 38, "model_version": "trk_gru_v1" }
    ]
  },
  "model_context": {
    "trk_gru_v1": {
      "test_mae_km_24h": 172, "baseline_persistence_km_24h": 231,
      "skill_vs_persistence_pct": 25.5, "trained_on_seasons": "1998–2012"
    }
  },
  "known_limitations": [
    "No sea-surface temperature, ocean heat content, or vertical wind shear predictors are available; intensity skill is correspondingly limited.",
    "Trained on North Atlantic storms only; behaviour in other basins is unvalidated.",
    "This is a retrospective research prototype and is not an operational forecast."
  ],
  "allowed_claims": ["…"],
  "forbidden_claims": [
    "landfall timing or location", "casualty or damage estimates",
    "evacuation or safety advice", "comparison to any storm not in this packet",
    "any numeric value not present in this packet"
  ]
}
```

**Deliberate inclusions:**

- **`model_context`** — the model's *measured* error and skill. Without it, Gemini describes a
  forecast as though it were certain. With it, the narrative can say *"this model's 24-hour track
  error averages 172 km, about 26% better than persistence"* — which is both more useful and true.
- **`known_limitations`** — pushes the honesty requirement into the generated text automatically.
- **`forbidden_claims`** — an explicit, machine-checkable negative list, not a vague instruction.

---

## 8. Preventing Fabrication — five layers

A prompt instruction alone is not a control. Five independent layers, each of which fails safe:

**Layer 1 — Structural isolation.** Gemini's only input is the packet. It has no tools, no
retrieval, no database access, no web access. It cannot look anything up, so it cannot import an
outside "fact".

**Layer 2 — System prompt constraints.** Explicit, repeated, and specific:

> You are a scientific explanation assistant for a tropical cyclone research prototype.
> You will receive a JSON evidence packet. **Every factual statement you make must be traceable to a
> value in that packet.** You must not use outside knowledge about specific storms, and you must not
> estimate, infer, or interpolate any numeric value that is not present. If asked something the
> packet does not answer, reply exactly: *"That information is not available in this system's data."*
> Always state that predictions are model output with quantified uncertainty. Never give safety,
> evacuation, or operational guidance. Report the model's stated error alongside any forecast value.

**Layer 3 — Structured output.** Request a constrained response schema (e.g. `summary`,
`key_drivers[]`, `uncertainty_note`, `caveats[]`) rather than free prose. A schema narrows the space
in which invention can occur and makes the output directly renderable as UI components.

**Layer 4 — Guardrail validator (the load-bearing layer).** A deterministic post-processing check:

1. Extract every numeric token from the response.
2. Assert each appears in the packet, within a rounding tolerance, or is a trivially derived
   quantity from packet values (a difference or percentage of two present numbers, computed and
   checked by the validator).
3. Scan for forbidden-claim patterns — landfall timing, casualty/damage language, evacuation or
   safety directives, named storms absent from the packet.
4. On failure: **do not show the response.** Retry once with the violation quoted back; on a second
   failure, fall back to a deterministic template-generated summary built directly from the packet.
5. Log every violation with the packet hash — the violation rate becomes a reportable metric.

**Layer 5 — UI transparency.** Gemini output is visually distinguished as AI-generated narrative,
carries a "Grounded in stored model output" indicator, and offers an expandable **"View the data
behind this"** panel showing the evidence packet. The user can always audit the claim.

**Why the template fallback matters:** it guarantees the demo never shows a hallucinated cyclone
statement, and it means an outage or quota exhaustion degrades the UI to a plainer summary instead of
breaking the page. Gemini becomes an enhancement layer, not a dependency.

### 8.1 Gemini configuration

| Setting | Value | Reason |
|---|---|---|
| Temperature | Low (~0.2–0.3) | Explanation, not creativity. Low temperature reduces embellishment |
| Max output tokens | Bounded per endpoint | Cost control; long answers drift further from the packet |
| Streaming | Enabled | Perceived latency. **Validate the full text before marking the stream complete**, and retract with the fallback if validation fails |
| Safety settings | Defaults | No adversarial content is expected |
| Model choice | The current cost-effective Gemini text model | Explanation from a structured packet is not a frontier-capability task; select on latency and cost, and make it configurable |

---

## 9. What is Deliberately Not Built

| Not built | Why |
|---|---|
| WebSocket / SSE live updates (beyond Gemini streaming) | Data is static. Polling is unnecessary; push is meaningless |
| GraphQL | The access patterns are few and known. REST + generated types delivers the same safety with less machinery |
| Redis / Celery / task queue | No long-running online work exists — inference is offline |
| User accounts, auth, personalisation | No per-user state in the MVP |
| Model-serving framework (TorchServe, Triton, BentoML) | Nothing is served online |
| Multi-tenancy, quotas, billing | Out of scope |

Each of these would be defensible in a production system with different requirements. **None of them
solves a problem this system has**, and the brief explicitly asked that no technology be introduced
for the sake of complexity.

---

## 10. Update (Phase 9): Gemini Implementation Status

§6-8 above were written at Phase 0, before any model existed to explain — they were the plan.
Phase 9 implemented them; this section records what shipped exactly as specified vs. what remains
deliberately deferred, without editing the historical design above.

**Built, matching §6-8 exactly:** the evidence packet (`backend/app/gemini/schemas.py::
EvidencePacket`, versioned `"v1"`, the same field groups §7 sketched), the system prompt (§8 Layer
2, verbatim rule-for-rule), structured output via the SDK's native schema support (§8 Layer 3), the
Guardrail Validator (§8 Layer 4 — every numeric/categorical/model-identity claim checked, forbidden
claims scanned), the deterministic template fallback (§8's "why the template fallback matters"),
backend-only execution with the API key confined to `backend/app/gemini/` (§6.2/§6.3), and only one
endpoint, `POST /api/v1/explain/forecast` (§3.7's first row).

**Deliberately deferred, not silently dropped:** response streaming, the `(sid, t, scope,
model_version, audience)` response cache, per-IP/global rate limiting (§4/§8.1), and
`/explain/storm`/`/explain/compare`/`/explain/ask` (§3.7's other rows) — all out of the task's
explicitly minimal Phase 9 contract. `Layer 5` (UI transparency: the "Grounded in stored model
output" indicator, the "View the data behind this" panel) is a frontend concern and was not built,
per Phase 9's explicit no-frontend-work instruction.

Full detail, exact test counts, and three real bugs found via a live-API smoke test:
[PHASE_9_GEMINI_INTEGRATION.md](PHASE_9_GEMINI_INTEGRATION.md).
