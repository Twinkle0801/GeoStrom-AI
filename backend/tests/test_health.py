"""Application startup and health endpoint."""

from __future__ import annotations


def test_app_starts_and_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"


def test_meta_endpoint(client, sample_model, sample_intensity_model):
    r = client.get("/api/v1/meta")
    assert r.status_code == 200
    body = r.json()
    assert body["project_name"] == "GeoStrom AI"
    assert "track_cliper_v1" in body["active_models"]
    assert "not operational" in body["note"].lower() or "not" in body["note"].lower()
