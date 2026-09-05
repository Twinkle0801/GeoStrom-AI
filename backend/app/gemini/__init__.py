"""Phase 9: Gemini natural-language explanation layer.

Per docs/API_ARCHITECTURE.md §6.2 ("DECISION: all Gemini calls originate in
`backend/app/gemini/`"), every Gemini-touching import in this codebase lives
under this package. No other module calls the Gemini SDK directly.

Gemini never produces a number: it explains numbers the rest of GeoStrom AI
already computed. See docs/PHASE_9_GEMINI_INTEGRATION.md for the full
architecture.
"""
