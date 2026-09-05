"""GET /api/v1/analytics/model-performance -- reads real, committed
ml/reports/*.json benchmark files. No mocking: these files are already part
of the repository (Phase 2/5/6/7/8's own outputs), so this test exercises
the real read path directly, the same way `test_ingestion.py` exercises the
real Phase 2 benchmark file."""

from __future__ import annotations


def test_model_performance_returns_200_and_expected_shape(client):
    r = client.get("/api/v1/analytics/model-performance")
    assert r.status_code == 200
    body = r.json()
    assert body["intensity"]["task"] == "intensity"
    assert body["track"]["task"] == "track"
    assert body["classification"]["task"] == "classification"


def test_intensity_lightgbm_is_the_recommended_baseline(client):
    body = client.get("/api/v1/analytics/model-performance").json()
    lgbm = next(m for m in body["intensity"]["models"] if m["model_name"] == "intensity_lightgbm_v1")
    assert lgbm["tier"] == "baseline"
    assert lgbm["is_recommended"] is True
    assert body["intensity"]["recommended_model"] == "LightGBM"


def test_intensity_gru_is_exploratory_and_not_recommended(client):
    body = client.get("/api/v1/analytics/model-performance").json()
    gru = next(m for m in body["intensity"]["models"] if m["model_name"] == "intensity_gru_v1")
    assert gru["tier"] == "exploratory"
    assert gru["is_recommended"] is False


def test_track_cliper_is_the_recommended_baseline(client):
    body = client.get("/api/v1/analytics/model-performance").json()
    cliper = next(m for m in body["track"]["models"] if m["model_name"] == "track_cliper_v1")
    assert cliper["tier"] == "baseline"
    assert cliper["is_recommended"] is True


def test_track_gru_is_exploratory(client):
    body = client.get("/api/v1/analytics/model-performance").json()
    gru = next(m for m in body["track"]["models"] if m["model_name"] == "track_gru_v1")
    assert gru["tier"] == "exploratory"


def test_classification_logistic_regression_is_recommended(client):
    body = client.get("/api/v1/analytics/model-performance").json()
    lr = next(m for m in body["classification"]["models"] if m["model_name"] == "logistic_regression")
    assert lr["tier"] == "baseline"
    assert lr["is_recommended"] is True


def test_classification_cnn_models_are_exploratory(client):
    body = client.get("/api/v1/analytics/model-performance").json()
    names_tiers = {m["model_name"]: m["tier"] for m in body["classification"]["models"]}
    assert names_tiers["resnet18"] == "exploratory"
    assert names_tiers["small_cnn"] == "exploratory"


def test_intensity_metrics_by_horizon_include_all_four_horizons(client):
    body = client.get("/api/v1/analytics/model-performance").json()
    lgbm = next(m for m in body["intensity"]["models"] if m["model_name"] == "intensity_lightgbm_v1")
    assert set(lgbm["metrics_by_horizon"].keys()) == {"6", "12", "18", "24"}
    assert lgbm["metrics_by_horizon"]["24"]["mae_kt"] == 8.535242906682745


def test_no_metric_is_fabricated_matches_committed_report(client):
    """Cross-check one exact value against the real committed JSON file
    directly -- proves the endpoint reads, not invents."""
    import json
    from pathlib import Path
    report = json.loads(
        (Path(__file__).resolve().parents[2] / "ml" / "reports" /
         "phase7_intensity_gru_results.json").read_text(encoding="utf-8")
    )
    expected = report["comparison_vs_phase2"]["24"]["intensity_lightgbm_v1"]["mae_kt"]
    body = client.get("/api/v1/analytics/model-performance").json()
    lgbm = next(m for m in body["intensity"]["models"] if m["model_name"] == "intensity_lightgbm_v1")
    assert lgbm["metrics_by_horizon"]["24"]["mae_kt"] == expected
