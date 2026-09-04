"""Ingestion validation and end-to-end integration tests.

Uses small SYNTHETIC data shaped exactly like the real Phase 2 artifact
(not the real 1732-row Parquet file, to keep this suite fast and
independent of ml/reports/ being present) to exercise the actual ingestion
functions -- validation, observed-track reconstruction, and the full
artifact -> database -> API path, plus the required idempotency guarantee.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import select

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT / "scripts"))

import ingest_phase2_predictions as ingest  # noqa: E402
from app.db.models import ModelVersion, Observation, Prediction, Storm  # noqa: E402


def _synthetic_predictions_df(n_windows: int = 3, sid: str = "2010176N16278") -> pd.DataFrame:
    """Two windows of one synthetic storm, columns matching the real
    Phase 2 artifact schema exactly (docs/PHASE_2_FORECASTING_BASELINES.md)."""
    base = dt.datetime(2010, 6, 26, 0, tzinfo=dt.timezone.utc)
    rows = []
    for i in range(n_windows):
        t_ref = base + dt.timedelta(hours=6 * i)
        row = {
            "sid": sid, "t_ref": t_ref, "season": 2010,
            "ref_lat": 16.0 + i * 0.3, "ref_lon": -86.0 - i * 0.5,
            "ref_wind": 40.0 + i * 5, "ref_pres": 1005.0 - i,
        }
        for h in ingest.HORIZONS:
            row[f"y_wind_abs_{h}h_true"] = 45.0 + i + h * 0.1
            row[f"y_lat_future_{h}h_true"] = 16.0 + i * 0.3 + h * 0.02
            row[f"y_lon_future_{h}h_true"] = -86.0 - i * 0.5 - h * 0.03
        for m in ingest.INTENSITY_MODELS:
            for h in ingest.HORIZONS:
                row[f"{m}__wind_{h}h"] = 44.0 + i + h * 0.1
        for m in ingest.TRACK_MODELS:
            for h in ingest.HORIZONS:
                row[f"{m}__dlat_{h}h"] = h * 0.02
                row[f"{m}__dlon_{h}h"] = -h * 0.03
        rows.append(row)
    return pd.DataFrame(rows)


def _synthetic_benchmark(models=None) -> list[dict]:
    models = models or ingest.ALL_MODELS
    out = []
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    for m in models:
        task = "intensity" if m.startswith("intensity") else "track"
        for h in ingest.HORIZONS:
            metrics = ({"n": 10, "mae_kt": 5.0, "rmse_kt": 6.0, "bias_kt": 0.1} if task == "intensity"
                      else {"n": 10, "mean_track_error_km": 50.0 + h, "median_track_error_km": 40.0})
            out.append({
                "model_name": m, "model_version": "v1", "task": task,
                "dataset_version": "v1", "split_version": "v1", "feature_version": "v1",
                "forecast_horizon_h": h, "sample_count": 10, "metrics": metrics,
                "timestamp_utc": now, "config": {"alpha": 1.0},
            })
    return out


class TestValidation:
    def test_missing_column_is_rejected(self, tmp_path, monkeypatch):
        df = _synthetic_predictions_df().drop(columns=["ref_wind"])
        p = tmp_path / "bad.parquet"
        df.to_parquet(p)
        monkeypatch.setattr(ingest, "PHASE2_PREDICTIONS", p)
        with pytest.raises(ingest.ValidationError, match="missing required columns"):
            ingest.load_and_validate_predictions()

    def test_malformed_sid_is_rejected(self, tmp_path, monkeypatch):
        df = _synthetic_predictions_df(sid="NOT_A_VALID_SID")
        p = tmp_path / "bad.parquet"
        df.to_parquet(p)
        monkeypatch.setattr(ingest, "PHASE2_PREDICTIONS", p)
        with pytest.raises(ingest.ValidationError, match="SID grammar"):
            ingest.load_and_validate_predictions()

    def test_out_of_range_latitude_is_rejected(self, tmp_path, monkeypatch):
        df = _synthetic_predictions_df()
        df.loc[0, "ref_lat"] = 95.0
        p = tmp_path / "bad.parquet"
        df.to_parquet(p)
        monkeypatch.setattr(ingest, "PHASE2_PREDICTIONS", p)
        with pytest.raises(ingest.ValidationError, match="out-of-range"):
            ingest.load_and_validate_predictions()

    def test_out_of_range_longitude_is_rejected(self, tmp_path, monkeypatch):
        df = _synthetic_predictions_df()
        df.loc[0, "ref_lon"] = 200.0
        p = tmp_path / "bad.parquet"
        df.to_parquet(p)
        monkeypatch.setattr(ingest, "PHASE2_PREDICTIONS", p)
        with pytest.raises(ingest.ValidationError, match="out-of-range"):
            ingest.load_and_validate_predictions()

    def test_duplicate_sid_t_ref_is_rejected(self, tmp_path, monkeypatch):
        df = _synthetic_predictions_df()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        p = tmp_path / "bad.parquet"
        df.to_parquet(p)
        monkeypatch.setattr(ingest, "PHASE2_PREDICTIONS", p)
        with pytest.raises(ingest.ValidationError, match="duplicate"):
            ingest.load_and_validate_predictions()

    def test_gapped_t_ref_cadence_is_rejected(self, tmp_path, monkeypatch):
        df = _synthetic_predictions_df(n_windows=3)
        df.loc[2, "t_ref"] = df.loc[2, "t_ref"] + dt.timedelta(hours=6)  # introduce a gap
        p = tmp_path / "bad.parquet"
        df.to_parquet(p)
        monkeypatch.setattr(ingest, "PHASE2_PREDICTIONS", p)
        with pytest.raises(ingest.ValidationError, match="gap-free"):
            ingest.load_and_validate_predictions()

    def test_valid_artifact_passes(self, tmp_path, monkeypatch):
        df = _synthetic_predictions_df()
        p = tmp_path / "good.parquet"
        df.to_parquet(p)
        monkeypatch.setattr(ingest, "PHASE2_PREDICTIONS", p)
        result = ingest.load_and_validate_predictions()
        assert len(result) == 3

    def test_benchmark_missing_model_is_rejected(self):
        incomplete = _synthetic_benchmark(models=ingest.ALL_MODELS[:-1])
        with pytest.raises(ingest.ValidationError, match="No benchmark rows"):
            ingest.build_model_versions(incomplete)

    def test_load_benchmark_metrics_upfront_check(self, tmp_path, monkeypatch):
        p = tmp_path / "incomplete_benchmark.json"
        p.write_text(
            __import__("json").dumps(_synthetic_benchmark(models=ingest.ALL_MODELS[:-1])))
        monkeypatch.setattr(ingest, "PHASE2_BENCHMARK", p)
        with pytest.raises(ingest.ValidationError, match="missing model"):
            ingest.load_benchmark_metrics()

    def test_model_name_version_split(self):
        assert ingest.split_model_name("track_cliper_v1") == ("track_cliper", "v1")
        assert ingest.split_model_name("intensity_lightgbm_v1") == ("intensity_lightgbm", "v1")
        with pytest.raises(ingest.ValidationError):
            ingest.split_model_name("no_version_suffix")

    def test_category_from_wind_thresholds(self):
        assert ingest.category_from_wind(None) is None
        assert ingest.category_from_wind(20) == -1
        assert ingest.category_from_wind(50) == 0
        assert ingest.category_from_wind(70) == 1
        assert ingest.category_from_wind(140) == 5


class TestObservedTrackReconstruction:
    def test_reconstructs_ref_points_plus_final_future_points(self):
        df = _synthetic_predictions_df(n_windows=3)
        obs = ingest.reconstruct_observed_rows(df)
        # 3 ref points + 4 future points from the LAST window = 7
        assert len(obs) == 7
        assert obs["sid"].nunique() == 1

    def test_no_duplicate_timestamps(self):
        df = _synthetic_predictions_df(n_windows=5)
        obs = ingest.reconstruct_observed_rows(df)
        assert obs.duplicated(subset=["sid", "ts"]).sum() == 0

    def test_track_is_temporally_sorted_and_gap_free(self):
        df = _synthetic_predictions_df(n_windows=4)
        obs = ingest.reconstruct_observed_rows(df)
        ts = obs.sort_values("ts")["ts"]
        diffs = ts.diff().dropna()
        assert (diffs == pd.Timedelta(hours=6)).all()

    def test_future_points_are_not_confused_with_predictions(self):
        """The reconstructed observed rows carry no model attribution --
        they are OBSERVED ground truth, structurally distinct from a
        Prediction row even though the numeric values originate in the
        same y_*_true columns a Prediction row also copies for context."""
        df = _synthetic_predictions_df(n_windows=2)
        obs = ingest.reconstruct_observed_rows(df)
        assert "model_id" not in obs.columns
        assert "pred_lat" not in obs.columns


@pytest.mark.usefixtures("engine")
class TestEndToEndIngestion:
    """The full artifact -> database path, using the real upsert functions
    against the real (test) PostgreSQL/PostGIS database."""

    def test_full_ingestion_writes_expected_rows(self, db_session, tmp_path, monkeypatch):
        df = _synthetic_predictions_df(n_windows=3)
        model_defs = ingest.build_model_versions(_synthetic_benchmark())

        model_ids = ingest.upsert_model_versions(db_session, model_defs)
        assert len(model_ids) == 6

        obs = ingest.reconstruct_observed_rows(df)
        seasons = df.groupby("sid")["season"].first()
        split_map = {"2010176N16278": "test"}
        storms = ingest.build_storms(obs, seasons, split_map)
        ingest.upsert_storms(db_session, storms)

        n_obs = ingest.upsert_observations(db_session, obs)
        assert n_obs == 7

        pred_records = ingest.build_predictions(df, model_ids)
        # 3 windows * 4 horizons * 6 models = 72
        assert len(pred_records) == 72
        n_pred = ingest.upsert_predictions(db_session, pred_records)
        assert n_pred == 72

        db_session.commit()
        assert db_session.scalar(select(Storm).where(Storm.sid == "2010176N16278")) is not None
        assert len(db_session.scalars(select(Observation)).all()) == 7
        assert len(db_session.scalars(select(Prediction)).all()) == 72

    def test_reingestion_is_idempotent(self, db_session):
        df = _synthetic_predictions_df(n_windows=2)
        model_defs = ingest.build_model_versions(_synthetic_benchmark())

        def run_once():
            model_ids = ingest.upsert_model_versions(db_session, model_defs)
            obs = ingest.reconstruct_observed_rows(df)
            seasons = df.groupby("sid")["season"].first()
            storms = ingest.build_storms(obs, seasons, {"2010176N16278": "test"})
            ingest.upsert_storms(db_session, storms)
            ingest.upsert_observations(db_session, obs)
            preds = ingest.build_predictions(df, model_ids)
            ingest.upsert_predictions(db_session, preds)
            db_session.commit()
            return model_ids

        ids_1 = run_once()
        n_storms_1 = len(db_session.scalars(select(Storm)).all())
        n_obs_1 = len(db_session.scalars(select(Observation)).all())
        n_pred_1 = len(db_session.scalars(select(Prediction)).all())

        ids_2 = run_once()
        n_storms_2 = len(db_session.scalars(select(Storm)).all())
        n_obs_2 = len(db_session.scalars(select(Observation)).all())
        n_pred_2 = len(db_session.scalars(select(Prediction)).all())

        assert ids_1 == ids_2
        assert (n_storms_1, n_obs_1, n_pred_1) == (n_storms_2, n_obs_2, n_pred_2)
