"""Phase 5 Task 9: runtime leakage validators.

Real, callable validator functions (not just test assertions) so scripts
can defend themselves at run time, and so the adversarial tests in
`ml/tests/test_classification_leakage.py` have an actual validator to prove
catches a deliberately-introduced violation -- per the task's explicit
"include at least one adversarial test that proves the validator catches
it" instruction.
"""

from __future__ import annotations

import pandas as pd


def find_storm_split_violations(df: pd.DataFrame) -> list[dict]:
    """Storms that appear in more than one split. Empty list = clean.

    Applies to either the Phase 4 sample index or the Phase 5
    classification index -- both carry `storm_id` and `split`.
    """
    per_storm_splits = df.groupby("storm_id")["split"].unique()
    violations = []
    for storm_id, splits in per_storm_splits.items():
        if len(splits) > 1:
            violations.append({"storm_id": storm_id, "splits": sorted(splits.tolist())})
    return violations


def find_excluded_rows_in_selection(df: pd.DataFrame) -> list[str]:
    """`sample_id`s that are NOT qc_status=='included' but are present in a
    DataFrame that is about to be used for training/evaluation. Empty list
    = clean. Call this on whatever DataFrame a baseline script is about to
    slice `X`/`y` from, right before doing so.
    """
    if "qc_status" not in df.columns:
        raise ValueError("DataFrame has no qc_status column -- cannot validate")
    bad = df[df["qc_status"] != "included"]
    return bad["sample_id"].tolist()


def assert_no_storm_split_leakage(df: pd.DataFrame) -> None:
    """Raise with full detail if any storm crosses a split boundary."""
    violations = find_storm_split_violations(df)
    if violations:
        raise ValueError(f"Storm(s) appear in more than one split: {violations}")


def assert_no_excluded_rows(df: pd.DataFrame) -> None:
    """Raise with full detail if an excluded row would enter training/eval."""
    bad = find_excluded_rows_in_selection(df)
    if bad:
        raise ValueError(f"{len(bad)} excluded sample(s) present in a training/eval "
                         f"selection: {bad[:10]}{'...' if len(bad) > 10 else ''}")
