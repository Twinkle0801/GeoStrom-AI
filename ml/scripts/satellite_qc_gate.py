"""Pretty-print the Phase 4 satellite QC gate report.

The QC report itself is produced as part of `build_satellite_dataset.py`
(via `ml.geostrom_ml.satellite.qc.build_qc_report`); this script only
re-renders an already-written report in the same tabular style as Phase 1's
`ml/scripts/qc_gate.py`, so the gate can be inspected without re-running the
full (expensive, real-NetCDF-I/O-bound) pipeline.

Usage:
    python ml/scripts/satellite_qc_gate.py
    python ml/scripts/satellite_qc_gate.py --report path/to/other_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.geostrom_ml.config import REPORT_DIR  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, default=REPORT_DIR / "satellite_qc_gate.json")
    args = ap.parse_args()

    if not args.report.exists():
        print(f"No QC report at {args.report}. Run build_satellite_dataset.py first.", file=sys.stderr)
        return 1

    report = json.loads(args.report.read_text(encoding="utf-8"))

    print(f"{'ID':<5}{'B':<3}{'RESULT':<8}{'CHECK':<62}VALUE")
    print("-" * 110)
    for c in report["checks"]:
        res = "PASS" if c["passed"] else ("FAIL" if c["passed"] is False else "n/a")
        print(f"{c['id']:<5}{'*' if c['blocking'] else ' ':<3}{res:<8}{c['name'][:60]:<62}{c['value']}")

    s = report["summary"]
    print("-" * 110)
    print(f"GATE: {s['gate_status']}   passed {s['passed']}/{s['total_checks']}   "
          f"blocking failures: {s['blocking_failures'] or 'none'}   (* = blocking)\n")

    counts = report["counts"]
    print("18-POINT REPORT:")
    for k, v in counts.items():
        label = k.split("_", 1)[1].replace("_", " ")
        if isinstance(v, dict):
            print(f"  {counts_key_num(k):>2}. {label:<32} {json.dumps(v, default=str)[:120]}")
        else:
            print(f"  {counts_key_num(k):>2}. {label:<32} {v}")
    return 0 if s["gate_status"] == "PASS" else 1


def counts_key_num(k: str) -> str:
    return k.split("_", 1)[0]


if __name__ == "__main__":
    raise SystemExit(main())
