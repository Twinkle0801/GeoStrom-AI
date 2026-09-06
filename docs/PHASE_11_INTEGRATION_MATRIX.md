# Phase 11 — Integration Matrix

Built from the actual repository contracts (read/tested directly for this phase — not assumed
from prior phase documentation). Status values: **PASS** (contract verified, no defect), **FIXED**
(a real gap was found and closed this phase), **PARTIAL** (works but with a documented,
non-fabricated caveat), **BLOCKED BY EXISTING CONTRACT** (no data/endpoint exists; correctly shown
as an honest empty state, not built this phase), **DEFERRED** (a legitimate future improvement,
out of this phase's scope).

| Layer | Producer | Consumer | Contract | Status |
|---|---|---|---|---|
| IBTrACS best-track | Raw archive (Phase 1) | `ml/geostrom_ml/features/engineering.py` | Per-timestep feature schema, causal L=8/H=4 windows | PASS — `ml/tests/test_leakage.py`, `test_splits.py` (30/30) |
| HURSAT-B1 → Zarr | Satellite pipeline (Phase 4) | Classification fusion | Canonical Zarr + Parquet metadata schema | PARTIAL — pipeline verified end-to-end on 12 real storms/627 fused samples; **531/547** frozen-split storms have archive coverage, full archive not processed (documented, not expanded this phase) |
| ADT-HURSAT scene labels | Phase 4/5 fusion | Classification taxonomy | `scene_taxonomy_v1` (CDO+IrrCDO, CurvedBand, Eye+LargeEye, Shear; Land/EmbCenter excluded) | PASS at the offline-artifact level (`ml/reports/phase5_baseline_results.json`); **BLOCKED BY EXISTING CONTRACT** for per-storm serving — no `classifications` table exists |
| Features (Parquet) | Offline preprocessing | LightGBM/Ridge/GRU training | `ml/manifests/dataset_v1_manifest.json`, frozen `splits_v1.json` | PASS — DB `Storm.split` cross-checked against the frozen manifest live this phase: **88/88 storms match, 0 mismatches** |
| Predictions (offline) | ML training scripts | `ingest_phase2_predictions.py` | `phase2_test_predictions.parquet` row schema | PASS — idempotent upsert, re-verified via live query this phase |
| Predictions (DB) | Repository (`app/repositories/storms.py`) | FastAPI routes | SQLAlchemy ORM row → Pydantic schema | PASS — no raw SQL beyond a parameterless `SELECT 1` health check; ORM-only elsewhere |
| API (track/prediction/analytics) | Backend | Frontend (`lib/api.ts`, generated `api-types.ts`) | OpenAPI-generated TS types | PASS — 0 TypeScript errors, 0 ESLint errors, regenerated and reverified this phase |
| Geometry (backend) | `app/services/geo.py` | `app/services/geometry.py` → GeoJSON | Claimed "byte-for-byte identical" to `ml/geostrom_ml/features/geo.py` | **FIXED (verified, not assumed)** — this exact claim had never been tested; 30 new parity tests added and pass, including antimeridian cases |
| GeoJSON | Backend | `CycloneMap.tsx` (Leaflet) | `[lon, lat]` RFC 7946 order | PASS — verified live via curl (`[-75.4, 29.5]` order) and by code inspection of every `[lon, lat] → [lat, lon]` conversion site |
| EvidencePacket | DB (via `evidence_builder.py`) | Gemini service | Pydantic `EvidencePacket`, versioned `"v1"` | PASS — new integration test traces exact seeded DB values through to the JSON response unchanged |
| Gemini | Backend (`app/gemini/`) | Validator → fallback → `ExplainResponse` | Structured JSON + deterministic grounding | PASS — 50/50 existing grounding/service tests re-verified; all 10 named failure modes covered |
| ExplainResponse | Backend | `GeminiPanel`/`EvidenceDrawer` | `source`, `fallback_reason`, `evidence` fields | PASS — live real-Gemini call verified (`source: "gemini"`, correct model identity, correct evidence content) |
| Model registry | `ModelVersion` rows + `ml/reports/*.json` | `/analytics/model-performance`, `ModelSelector` | `tier`/`is_recommended` fields | PASS — live-verified: LightGBM/CLIPER-style Ridge/Logistic Regression correctly `is_recommended=true`; every GRU/CNN row correctly `tier="exploratory"` |
| Satellite frame serving | — (does not exist) | `SatelliteViewer.tsx` | — | **BLOCKED BY EXISTING CONTRACT** — no endpoint; honest empty state confirmed still in place, not fabricated |
| Classification serving | — (does not exist) | `ClassificationPanel.tsx` | — | **BLOCKED BY EXISTING CONTRACT** — no table/endpoint; honest empty state confirmed still in place |

## Notes on rows marked FIXED

Only one genuine code-level gap was found and closed this phase (`app/schemas/storm.py`'s
`n_observations` field previously had no description — a real, if minor, documentation-contract
gap, now clarified: see §4). The `app/services/geo.py` parity claim was **verified**, not
**fixed** — no discrepancy was found; the test that should have existed to prove the claim simply
didn't, and now does.
