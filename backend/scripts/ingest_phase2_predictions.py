"""Ingest Phase 2 baseline predictions into PostgreSQL/PostGIS.

This is the ONE place the offline (ml/) and online (backend/) worlds meet,
matching docs/SYSTEM_ARCHITECTURE.md's "the only handoff" between the
offline and online halves of the system. It is a standalone script, run
manually or as a build step -- it is NOT imported by the running FastAPI
application (`backend/app/`), which never touches ml/ or this script.

Inputs (all already-committed Phase 1/2 artifacts; nothing is downloaded or
recomputed):
    ml/reports/phase2_test_predictions.parquet   -- raw per-window predictions
    ml/reports/phase2_benchmark_results.json     -- per-model, per-horizon metrics
    ml/manifests/splits_v1.json                  -- frozen storm->split map
    ml/manifests/dataset_v1_manifest.json        -- dataset/feature version

Design note -- why the observed track needs no separate IBTrACS read:
    Phase 2's sliding windows (L=8, H=4, stride 1) mean that, for a given
    storm, the sorted `t_ref` values of every window are an exact,
    gap-free 6-hourly sequence (verified: 0/88 test storms have a gap
    between consecutive window t_ref values). The union of every window's
    reference point, plus the last window's +6/+12/+18/+24h ground-truth
    targets, therefore reconstructs the storm's COMPLETE observed synoptic
    track for its test-eligible span -- with zero re-derivation of Phase 2's
    causal feature/window logic and zero import of ml/geostrom_ml.

Idempotency: every insert goes through `INSERT ... ON CONFLICT ... DO
UPDATE`, keyed on the same UNIQUE constraints Alembic created. Re-running
this script against unchanged input is a true no-op (row counts and content
identical); running it again after the source Parquet legitimately changes
updates rows in place rather than creating duplicates.

Usage:
    cd backend && python scripts/ingest_phase2_predictions.py [--dry-run]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.db.base import SessionLocal, engine  # noqa: E402
from app.db.models import ModelVersion, Observation, Prediction, Storm  # noqa: E402
from app.services.geo import (  # noqa: E402
    displace, haversine_km, validate_lat, validate_lon,
)

PHASE2_PREDICTIONS = REPO_ROOT / "ml" / "reports" / "phase2_test_predictions.parquet"
PHASE2_BENCHMARK = REPO_ROOT / "ml" / "reports" / "phase2_benchmark_results.json"
SPLITS_MANIFEST = REPO_ROOT / "ml" / "manifests" / "splits_v1.json"
DATASET_MANIFEST = REPO_ROOT / "ml" / "manifests" / "dataset_v1_manifest.json"

BASIN = "NA"  # Phase 3 task: North Atlantic is locked as the MVP basin
HORIZONS = (6, 12, 18, 24)
INTENSITY_MODELS = ["intensity_persistence_v1", "intensity_ridge_v1", "intensity_lightgbm_v1"]
TRACK_MODELS = ["track_persistence_v1", "track_cliper_v1", "track_lightgbm_v1"]
ALL_MODELS = INTENSITY_MODELS + TRACK_MODELS

SID_RE = re.compile(r"^\d{7}[NS]\d{5}$")  # IBTrACS SID grammar, per Phase 1 verification

# Standard Saffir-Simpson thresholds on 1-minute wind (kt), matching the
# public definition IBTrACS itself uses for USA_SSHS (Phase 1 column
# documentation). Applying this fixed, published formula to a wind value
# already in the dataset is a deterministic DERIVED classification, not a
# fabricated one.
def category_from_wind(wind_kt: float | None) -> int | None:
    if wind_kt is None:
        return None
    if wind_kt < 34:
        return -1  # tropical depression
    if wind_kt < 64:
        return 0   # tropical storm
    if wind_kt < 83:
        return 1
    if wind_kt < 96:
        return 2
    if wind_kt < 113:
        return 3
    if wind_kt < 137:
        return 4
    return 5


class ValidationError(RuntimeError):
    pass


def load_and_validate_predictions() -> pd.DataFrame:
    if not PHASE2_PREDICTIONS.exists():
        raise FileNotFoundError(
            f"{PHASE2_PREDICTIONS} not found. This script ingests the existing "
            "Phase 2 artifact and must not regenerate it -- run Phase 2's "
            "ml/scripts/run_phase2_benchmark.py yourself first if it is missing."
        )
    df = pd.read_parquet(PHASE2_PREDICTIONS)

    required_base = {"sid", "t_ref", "season", "ref_lat", "ref_lon", "ref_wind", "ref_pres"}
    required_targets = {f"y_{f}_{h}h_true"
                        for h in HORIZONS
                        for f in ("wind_abs", "lat_future", "lon_future")}
    required_model_cols = set()
    for m in INTENSITY_MODELS:
        required_model_cols |= {f"{m}__wind_{h}h" for h in HORIZONS}
    for m in TRACK_MODELS:
        required_model_cols |= {f"{m}__dlat_{h}h" for h in HORIZONS} \
            | {f"{m}__dlon_{h}h" for h in HORIZONS}
    required = required_base | required_targets | required_model_cols

    missing = required - set(df.columns)
    if missing:
        raise ValidationError(f"Phase 2 artifact missing required columns: {sorted(missing)}")

    # --- dtype / value validation -----------------------------------------
    if not pd.api.types.is_datetime64_any_dtype(df["t_ref"]):
        raise ValidationError(f"t_ref is not a datetime column (dtype={df['t_ref'].dtype})")

    bad_sids = sorted(set(df["sid"]) - {s for s in df["sid"].unique() if SID_RE.match(s)})
    if bad_sids:
        raise ValidationError(f"{len(bad_sids)} storm id(s) do not match the IBTrACS SID "
                              f"grammar: {bad_sids[:5]}")

    for col in ("ref_lat",) + tuple(f"y_lat_future_{h}h_true" for h in HORIZONS):
        bad = df[col].notna() & ~df[col].between(-90, 90)
        if bad.any():
            raise ValidationError(f"{int(bad.sum())} rows have out-of-range {col}")
    for col in ("ref_lon",) + tuple(f"y_lon_future_{h}h_true" for h in HORIZONS):
        bad = df[col].notna() & ~df[col].between(-180, 180)
        if bad.any():
            raise ValidationError(f"{int(bad.sum())} rows have out-of-range {col}")

    dup = df.duplicated(subset=["sid", "t_ref"])
    if dup.any():
        raise ValidationError(f"{int(dup.sum())} duplicate (sid, t_ref) rows in source artifact")

    # gap-free cadence assumption the observed-track reconstruction depends on
    for sid, grp in df.groupby("sid"):
        diffs = grp.sort_values("t_ref")["t_ref"].diff().dropna()
        if len(diffs) and not (diffs == pd.Timedelta(hours=6)).all():
            raise ValidationError(
                f"Storm {sid}: window t_ref values are not a gap-free 6h "
                f"sequence -- the observed-track reconstruction assumption "
                f"does not hold for this artifact. Refusing to ingest."
            )

    return df


def load_benchmark_metrics() -> list[dict]:
    if not PHASE2_BENCHMARK.exists():
        raise FileNotFoundError(f"{PHASE2_BENCHMARK} not found.")
    records = json.loads(PHASE2_BENCHMARK.read_text(encoding="utf-8"))
    names = {r["model_name"] for r in records}
    missing = set(ALL_MODELS) - names
    if missing:
        raise ValidationError(f"Benchmark results missing model(s): {sorted(missing)}")
    return records


def split_model_name(full_name: str) -> tuple[str, str]:
    m = re.match(r"^(.+)_v(\d+)$", full_name)
    if not m:
        raise ValidationError(f"Model name '{full_name}' does not end in _v<N>")
    return m.group(1), f"v{m.group(2)}"


def build_model_versions(benchmark: list[dict]) -> dict[str, dict]:
    """One dict per model, keyed by full model_name, ready for upsert."""
    out: dict[str, dict] = {}
    for full_name in ALL_MODELS:
        rows = [r for r in benchmark if r["model_name"] == full_name]
        if not rows:
            raise ValidationError(f"No benchmark rows for model {full_name}")
        name, version = split_model_name(full_name)
        task = rows[0]["task"]
        metrics_by_horizon = {str(r["forecast_horizon_h"]): r["metrics"] for r in rows}
        error_radii = None
        if task == "track":
            error_radii = {str(r["forecast_horizon_h"]): r["metrics"]["mean_track_error_km"]
                           for r in rows}
        r24 = next(r for r in rows if r["forecast_horizon_h"] == 24)
        out[full_name] = {
            "name": name, "version": version, "task": task,
            "trained_at": dt.datetime.fromisoformat(r24["timestamp_utc"]),
            "dataset_build": r24["dataset_version"],
            "split_version": r24["split_version"],
            "feature_version": r24["feature_version"],
            "config": r24["config"],
            "metrics": metrics_by_horizon,
            "error_radii_km": error_radii,
            "is_active": True,
        }
    return out


def upsert_model_versions(db: Session, model_defs: dict[str, dict]) -> dict[str, int]:
    ids: dict[str, int] = {}
    for full_name, d in model_defs.items():
        stmt = pg_insert(ModelVersion).values(**d)
        stmt = stmt.on_conflict_do_update(
            index_elements=["name", "version"],
            set_={k: stmt.excluded[k] for k in d if k not in ("name", "version")},
        ).returning(ModelVersion.id)
        row_id = db.execute(stmt).scalar_one()
        ids[full_name] = row_id
    db.commit()
    return ids


def reconstruct_observed_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Every window's reference point, plus the last window's future
    targets per storm. See module docstring for why this is complete and
    exact, with no re-derivation of Phase 2 logic."""
    ref_rows = df[["sid", "t_ref", "ref_lat", "ref_lon", "ref_wind", "ref_pres"]].rename(
        columns={"t_ref": "ts", "ref_lat": "lat", "ref_lon": "lon",
                 "ref_wind": "wind_kt", "ref_pres": "pressure_hpa"})

    future_frames = []
    last = df.sort_values("t_ref").groupby("sid", as_index=False).last()
    for h in HORIZONS:
        future_frames.append(pd.DataFrame({
            "sid": last["sid"],
            "ts": last["t_ref"] + pd.Timedelta(hours=h),
            "lat": last[f"y_lat_future_{h}h_true"],
            "lon": last[f"y_lon_future_{h}h_true"],
            "wind_kt": last[f"y_wind_abs_{h}h_true"],
            # Phase 2 artifact does not carry future pressure truth
            "pressure_hpa": pd.Series([float("nan")] * len(last), dtype="float64"),
        }))

    all_rows = pd.concat([ref_rows] + future_frames, ignore_index=True)
    all_rows = all_rows.drop_duplicates(subset=["sid", "ts"]).sort_values(["sid", "ts"])
    all_rows["category"] = all_rows["wind_kt"].apply(category_from_wind)
    all_rows["step_index"] = all_rows.groupby("sid").cumcount()
    return all_rows.reset_index(drop=True)


def build_storms(obs: pd.DataFrame, seasons: pd.Series, split_map: dict[str, str]) -> list[dict]:
    storms = []
    for sid, g in obs.groupby("sid"):
        g = g.sort_values("ts")
        for _, row in g.iterrows():
            validate_lat(row["lat"], field=f"{sid} lat"); validate_lon(row["lon"], field=f"{sid} lon")
        season = int(seasons.loc[sid])
        split = split_map.get(sid)
        wind_vals = g["wind_kt"].dropna()
        pres_vals = g["pressure_hpa"].dropna()
        cat_vals = g["category"].dropna()
        coords = list(zip(g["lon"].tolist(), g["lat"].tolist()))
        linestring_wkt = "SRID=4326;LINESTRING(" + ",".join(f"{x} {y}" for x, y in coords) + ")"
        min_lon, max_lon = g["lon"].min(), g["lon"].max()
        min_lat, max_lat = g["lat"].min(), g["lat"].max()
        bbox_wkt = (f"SRID=4326;POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, "
                   f"{max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))")
        storms.append({
            "sid": sid, "name": None,  # not present in the Phase 2 artifact -- see docs
            "season": season, "basin": BASIN, "subbasin": None,
            "start_time": g["ts"].min().to_pydatetime(), "end_time": g["ts"].max().to_pydatetime(),
            "n_observations": len(g),
            "max_wind_kt": float(wind_vals.max()) if len(wind_vals) else None,
            "min_pressure_hpa": float(pres_vals.min()) if len(pres_vals) else None,
            "max_category": int(cat_vals.max()) if len(cat_vals) else None,
            "made_landfall": None,  # not present in the Phase 2 artifact -- see docs
            "split": split,
            "track_geom": linestring_wkt, "bbox": bbox_wkt,
        })
    return storms


def upsert_storms(db: Session, storms: list[dict]) -> None:
    for chunk_start in range(0, len(storms), 200):
        chunk = storms[chunk_start:chunk_start + 200]
        stmt = pg_insert(Storm).values(chunk)
        stmt = stmt.on_conflict_do_update(
            index_elements=["sid"],
            set_={k: stmt.excluded[k] for k in chunk[0] if k != "sid"},
        )
        db.execute(stmt)
    db.commit()


def upsert_observations(db: Session, obs: pd.DataFrame) -> int:
    records = []
    for _, row in obs.iterrows():
        validate_lat(row["lat"]); validate_lon(row["lon"])
        wkt = f"SRID=4326;POINT({row['lon']} {row['lat']})"
        records.append({
            "sid": row["sid"], "ts": row["ts"].to_pydatetime(), "step_index": int(row["step_index"]),
            "lat": float(row["lat"]), "lon": float(row["lon"]), "geom": wkt,
            "wind_kt": None if pd.isna(row["wind_kt"]) else float(row["wind_kt"]),
            "pressure_hpa": None if pd.isna(row["pressure_hpa"]) else float(row["pressure_hpa"]),
            "category": None if pd.isna(row["category"]) else int(row["category"]),
            "nature": None, "storm_speed_kt": None, "storm_dir_deg": None, "dist2land_km": None,
            "is_synoptic": True, "is_observed": True,
        })
    for i in range(0, len(records), 500):
        chunk = records[i:i + 500]
        stmt = pg_insert(Observation).values(chunk)
        stmt = stmt.on_conflict_do_update(
            index_elements=["sid", "ts"],
            set_={k: stmt.excluded[k] for k in chunk[0] if k not in ("sid", "ts")},
        )
        db.execute(stmt)
    db.commit()
    return len(records)


def build_predictions(df: pd.DataFrame, model_ids: dict[str, int]) -> list[dict]:
    records = []
    for row in df.itertuples(index=False):
        row = row._asdict()
        sid, t_ref = row["sid"], row["t_ref"]
        ref_lat, ref_lon = row["ref_lat"], row["ref_lon"]

        for h in HORIZONS:
            true_lat = row[f"y_lat_future_{h}h_true"]
            true_lon = row[f"y_lon_future_{h}h_true"]
            true_wind = row[f"y_wind_abs_{h}h_true"]
            valid_ts = t_ref + pd.Timedelta(hours=h)

            for m in INTENSITY_MODELS:
                pred_wind = row[f"{m}__wind_{h}h"]
                records.append({
                    "sid": sid, "task": "intensity", "origin_ts": t_ref.to_pydatetime(),
                    "lead_hours": h, "valid_ts": valid_ts.to_pydatetime(),
                    "model_id": model_ids[m],
                    "pred_lat": None, "pred_lon": None, "pred_geom": None,
                    "pred_wind_kt": float(pred_wind), "pred_pressure_hpa": None,
                    "error_radius_km": None,
                    "true_lat": float(true_lat), "true_lon": float(true_lon),
                    "true_wind_kt": float(true_wind),
                    "track_error_km": None,
                    "wind_error_kt": float(pred_wind - true_wind),
                })

            for m in TRACK_MODELS:
                dlat = row[f"{m}__dlat_{h}h"]
                dlon = row[f"{m}__dlon_{h}h"]
                pred_lat, pred_lon = displace(ref_lat, ref_lon, dlat, dlon)
                validate_lat(pred_lat, field=f"{sid}/{m}/{h}h pred_lat")
                validate_lon(pred_lon, field=f"{sid}/{m}/{h}h pred_lon")
                track_err = haversine_km(true_lat, true_lon, pred_lat, pred_lon)
                records.append({
                    "sid": sid, "task": "track", "origin_ts": t_ref.to_pydatetime(),
                    "lead_hours": h, "valid_ts": valid_ts.to_pydatetime(),
                    "model_id": model_ids[m],
                    "pred_lat": float(pred_lat), "pred_lon": float(pred_lon),
                    "pred_geom": f"SRID=4326;POINT({pred_lon} {pred_lat})",
                    "pred_wind_kt": None, "pred_pressure_hpa": None,
                    "error_radius_km": None,  # filled in below from model metadata
                    "true_lat": float(true_lat), "true_lon": float(true_lon),
                    "true_wind_kt": float(true_wind),
                    "track_error_km": float(track_err),
                    "wind_error_kt": None,
                })
    return records


def upsert_predictions(db: Session, records: list[dict]) -> int:
    for i in range(0, len(records), 2000):
        chunk = records[i:i + 2000]
        stmt = pg_insert(Prediction).values(chunk)
        stmt = stmt.on_conflict_do_update(
            index_elements=["sid", "origin_ts", "lead_hours", "model_id"],
            set_={k: stmt.excluded[k] for k in chunk[0]
                 if k not in ("sid", "origin_ts", "lead_hours", "model_id")},
        )
        db.execute(stmt)
    db.commit()
    return len(records)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="validate and report counts without writing to the database")
    args = ap.parse_args()

    print(f"Loading + validating {PHASE2_PREDICTIONS.name} ...")
    df = load_and_validate_predictions()
    print(f"  {len(df)} windows, {df['sid'].nunique()} storms -- validation passed")

    benchmark = load_benchmark_metrics()
    model_defs = build_model_versions(benchmark)
    print(f"  {len(model_defs)} model versions from {PHASE2_BENCHMARK.name}")

    splits = json.loads(SPLITS_MANIFEST.read_text(encoding="utf-8"))
    split_map: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        for sid in splits[split_name]["storm_ids"]:
            split_map[sid] = split_name
    unmapped = sorted(set(df["sid"]) - set(split_map))
    if unmapped:
        raise ValidationError(f"{len(unmapped)} storm(s) in predictions absent from the "
                              f"frozen split manifest: {unmapped[:5]}")
    not_test = sorted(sid for sid in df["sid"].unique() if split_map[sid] != "test")
    if not_test:
        print(f"  WARNING: {len(not_test)} storms are not in the 'test' split "
              f"(unexpected for phase2_test_predictions.parquet): {not_test[:5]}")

    obs = reconstruct_observed_rows(df)
    seasons = df.groupby("sid")["season"].first()
    storms = build_storms(obs, seasons, split_map)

    print(f"  Reconstructed {len(obs)} observed points across {len(storms)} storms")

    if args.dry_run:
        print("\n--dry-run: no database writes performed.")
        return 0

    db = SessionLocal()
    try:
        print("Upserting model_versions ...")
        model_ids = upsert_model_versions(db, model_defs)
        print(f"  {len(model_ids)} model_versions upserted: {model_ids}")

        print("Upserting storms ...")
        upsert_storms(db, storms)
        print(f"  {len(storms)} storms upserted")

        print("Upserting observations ...")
        n_obs = upsert_observations(db, obs)
        print(f"  {n_obs} observations upserted")

        print("Building + upserting predictions ...")
        pred_records = build_predictions(df, model_ids)
        # attach the model's empirical error radius per horizon (track only)
        radii_by_model_id = {model_ids[k]: v.get("error_radii_km") for k, v in model_defs.items()}
        for r in pred_records:
            radii = radii_by_model_id.get(r["model_id"])
            if radii:
                r["error_radius_km"] = radii.get(str(r["lead_hours"]))
        n_pred = upsert_predictions(db, pred_records)
        print(f"  {n_pred} predictions upserted")

        # --- post-write sanity counts ---
        n_storms_db = db.execute(select(Storm)).scalars().all()
        print(f"\nFinal DB counts: storms={len(n_storms_db)}")
    finally:
        db.close()

    print("\nIngestion complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
