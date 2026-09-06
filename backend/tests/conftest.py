"""Shared pytest fixtures for backend tests.

Uses a REAL PostgreSQL/PostGIS database (`geostrom_test`, a sibling to the
dev `geostrom` database), not SQLite and not a mock -- the task brief
explicitly asks for the "Phase 2 artifact -> database -> FastAPI endpoint"
path to be integration-tested, and PostGIS geometry columns have no SQLite
equivalent worth trusting for that. `TEST_DATABASE_URL` may be overridden
via environment variable for CI.
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg2://geostrom:geostrom_local_dev_pw_9f3a@localhost:5434/geostrom_test",
)
os.environ["DATABASE_URL"] = TEST_DATABASE_URL  # must be set before app.core.config is imported

from app.db.base import Base  # noqa: E402
from app.db.models import ModelVersion, Observation, Prediction, Storm  # noqa: E402
from app.main import app  # noqa: E402
from app.db.base import get_db  # noqa: E402
from app.api.v1.explain import get_explain_cache, get_rate_limiter  # noqa: E402
from app.gemini.cache import ExplainCache  # noqa: E402
from app.gemini.ratelimit import RateLimiter  # noqa: E402


@pytest.fixture(scope="session")
def engine():
    eng = create_engine(TEST_DATABASE_URL, future=True)
    Base.metadata.drop_all(eng)   # clean slate; this database holds nothing but test fixtures
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def db_session(engine):
    """One session per test, wrapped in a transaction that is rolled back --
    tests never see each other's writes and the schema is reused."""
    connection = engine.connect()
    trans = connection.begin()
    SessionLocal = sessionmaker(bind=connection, future=True)
    session = SessionLocal()
    yield session
    session.close()
    if trans.is_active:  # a test that triggered an IntegrityError already
        trans.rollback()  # rolled back the connection-level transaction
    connection.close()


@pytest.fixture
def client(db_session):
    """Every test gets its OWN fresh `ExplainCache`/`RateLimiter` instance,
    with generous bounds -- otherwise the module-level singletons in
    `app.api.v1.explain` would persist Gemini-explanation cache entries and
    rate-limit counters across unrelated tests (and files), since they live
    for the whole pytest process. Tests that specifically exercise caching
    or rate-limiting override these again, with tighter bounds, inside the
    test body -- `dependency_overrides` is just a dict, so a later
    assignment simply replaces this one for that test only."""
    def _override_get_db():
        yield db_session
    app.dependency_overrides[get_db] = _override_get_db
    # NOTE: each override must return the SAME instance on every call -- a
    # lambda that constructs a fresh ExplainCache()/RateLimiter() inline
    # would build a brand-new, empty one on every dependency resolution
    # (i.e. every request), silently defeating caching/rate-limiting
    # entirely. Construct once here, close over it.
    test_cache = ExplainCache(maxsize=100, ttl_seconds=3600.0)
    test_limiter = RateLimiter(max_requests=1000, window_seconds=60.0)
    app.dependency_overrides[get_explain_cache] = lambda: test_cache
    app.dependency_overrides[get_rate_limiter] = lambda: test_limiter
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def sample_model(db_session) -> ModelVersion:
    m = ModelVersion(
        name="track_cliper", version="v1", task="track",
        trained_at=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
        dataset_build="v1", split_version="v1", feature_version="v1",
        config={"alpha": 1.0}, metrics={"24": {"mean_track_error_km": 200.4}},
        error_radii_km={"6": 30.0, "12": 75.0, "18": 133.0, "24": 200.0},
        is_active=True,
    )
    db_session.add(m)
    db_session.flush()
    return m


@pytest.fixture
def sample_intensity_model(db_session) -> ModelVersion:
    m = ModelVersion(
        name="intensity_lightgbm", version="v1", task="intensity",
        trained_at=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
        dataset_build="v1", split_version="v1", feature_version="v1",
        config={"n_estimators": 300}, metrics={"24": {"mae_kt": 8.5}},
        error_radii_km=None, is_active=True,
    )
    db_session.add(m)
    db_session.flush()
    return m


@pytest.fixture
def sample_storm(db_session) -> Storm:
    s = Storm(
        sid="2010176N16278", name=None, season=2010, basin="NA", subbasin=None,
        start_time=dt.datetime(2010, 6, 26, 0, tzinfo=dt.timezone.utc),
        end_time=dt.datetime(2010, 6, 27, 0, tzinfo=dt.timezone.utc),
        n_observations=5, max_wind_kt=55.0, min_pressure_hpa=1000.0, max_category=0,
        made_landfall=None, split="test",
        track_geom="SRID=4326;LINESTRING(-86.1 16.9, -87.2 17.2, -88.2 17.5)",
        bbox="SRID=4326;POLYGON((-88.2 16.9,-86.1 16.9,-86.1 17.5,-88.2 17.5,-88.2 16.9))",
    )
    db_session.add(s)
    db_session.flush()
    return s


@pytest.fixture
def sample_observations(db_session, sample_storm) -> list[Observation]:
    rows = []
    base = dt.datetime(2010, 6, 26, 12, tzinfo=dt.timezone.utc)
    coords = [(16.9, -86.1, 40.0), (17.2, -87.2, 45.0), (17.5, -88.2, 55.0)]
    for i, (lat, lon, wind) in enumerate(coords):
        o = Observation(
            sid=sample_storm.sid, ts=base + dt.timedelta(hours=6 * i), step_index=i,
            lat=lat, lon=lon, geom=f"SRID=4326;POINT({lon} {lat})",
            wind_kt=wind, pressure_hpa=1010 - wind, category=-1, nature=None,
            storm_speed_kt=None, storm_dir_deg=None, dist2land_km=None,
            is_synoptic=True, is_observed=True,
        )
        db_session.add(o)
        rows.append(o)
    db_session.flush()
    return rows


@pytest.fixture
def sample_prediction(db_session, sample_storm, sample_model) -> Prediction:
    p = Prediction(
        sid=sample_storm.sid, task="track",
        origin_ts=dt.datetime(2010, 6, 26, 12, tzinfo=dt.timezone.utc),
        lead_hours=6, valid_ts=dt.datetime(2010, 6, 26, 18, tzinfo=dt.timezone.utc),
        model_id=sample_model.id,
        pred_lat=17.15, pred_lon=-87.15, pred_geom="SRID=4326;POINT(-87.15 17.15)",
        pred_wind_kt=None, pred_pressure_hpa=None, error_radius_km=30.0,
        true_lat=17.2, true_lon=-87.2, true_wind_kt=45.0,
        track_error_km=6.4, wind_error_kt=None,
    )
    db_session.add(p)
    db_session.flush()
    return p
