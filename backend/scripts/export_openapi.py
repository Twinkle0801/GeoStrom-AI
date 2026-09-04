"""Export the OpenAPI contract from the actual FastAPI app object.

Per docs/API_ARCHITECTURE.md §5: Pydantic models -> FastAPI /openapi.json ->
contracts/openapi.json -> generated frontend types. This script generates
the schema directly from `app.main:app` (no server needs to be running),
so the committed contract can never drift from what the code actually
declares.

Usage:
    cd backend && python scripts/export_openapi.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.main import app  # noqa: E402

OUT_PATH = REPO_ROOT / "contracts" / "openapi.json"


def main() -> int:
    schema = app.openapi()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(f"Wrote {len(schema['paths'])} paths to {OUT_PATH}")
    for path in schema["paths"]:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
