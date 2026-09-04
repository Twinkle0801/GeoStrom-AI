"""Migration reproducibility: the database must be creatable from Alembic
migrations alone, per the Phase 3 task's explicit validation requirement.

Runs `alembic upgrade head` as a real subprocess against a throwaway
database (created and dropped by this test), then asserts every expected
table and index exists -- proving the committed migration, not just the
ORM models, is sufficient to reproduce the schema.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import psycopg2
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
ADMIN_URL = "postgresql://geostrom:geostrom_local_dev_pw_9f3a@localhost:5434/geostrom"
THROWAWAY_DB = "geostrom_migration_test"
THROWAWAY_URL = f"postgresql+psycopg2://geostrom:geostrom_local_dev_pw_9f3a@localhost:5434/{THROWAWAY_DB}"


def _admin_connect():
    try:
        return psycopg2.connect(ADMIN_URL)
    except psycopg2.OperationalError:
        pytest.skip("Local Postgres not reachable at localhost:5434 -- skipping migration test")


@pytest.fixture
def throwaway_db():
    conn = _admin_connect()
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(f"DROP DATABASE IF EXISTS {THROWAWAY_DB}")
    cur.execute(f"CREATE DATABASE {THROWAWAY_DB}")
    conn.close()
    yield
    conn = _admin_connect()
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(f"DROP DATABASE IF EXISTS {THROWAWAY_DB}")
    conn.close()


def test_alembic_upgrade_head_from_scratch(throwaway_db):
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=str(BACKEND_ROOT), capture_output=True, text=True,
        env={"DATABASE_URL": THROWAWAY_URL, **_inherit_path_env()},
    )
    assert result.returncode == 0, f"alembic upgrade failed:\n{result.stdout}\n{result.stderr}"

    conn = psycopg2.connect(f"postgresql://geostrom:geostrom_local_dev_pw_9f3a@localhost:5434/{THROWAWAY_DB}")
    cur = conn.cursor()
    cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema='public'
        AND table_name IN ('model_versions','storms','observations','predictions','alembic_version')
        ORDER BY table_name
    """)
    tables = [r[0] for r in cur.fetchall()]
    assert tables == ["alembic_version", "model_versions", "observations", "predictions", "storms"]

    cur.execute("SELECT extname FROM pg_extension WHERE extname = 'postgis'")
    assert cur.fetchone() is not None, "PostGIS extension was not enabled by the migration"

    cur.execute("""
        SELECT indexname FROM pg_indexes
        WHERE tablename IN ('storms','observations','predictions')
        AND indexname LIKE '%geom%'
    """)
    geom_indexes = [r[0] for r in cur.fetchall()]
    assert len(geom_indexes) >= 3, "Expected GIST geometry indexes on storms/observations/predictions"
    conn.close()


def _inherit_path_env() -> dict:
    import os
    return {k: v for k, v in os.environ.items() if k != "DATABASE_URL"}
