"""Backend settings, from environment variables only.

Per docs/API_ARCHITECTURE.md §6.3 and the Phase 3 task instructions: no
secret or credential is ever hardcoded. `DATABASE_URL` is required; there is
no baked-in default pointing at a real database, only a documented example
in `.env.example`.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Required. e.g. postgresql+psycopg2://geostrom:***@localhost:5434/geostrom
    database_url: str

    api_v1_prefix: str = "/api/v1"
    allowed_origins: list[str] = ["http://localhost:3001", "http://localhost:3000"]

    # docs/API_ARCHITECTURE.md §2: default 50, max 500
    default_page_size: int = 50
    max_page_size: int = 500

    project_name: str = "GeoStrom AI"
    api_version: str = "0.4.0"  # Phase 9 Gemini explanation layer

    # Phase 9 -- Gemini explanation layer. Never a hardcoded default: no key
    # means Gemini is simply unconfigured and the API falls back to the
    # deterministic template (app/gemini/service.py), it never fails startup
    # and never fails a request. Per docs/API_ARCHITECTURE.md §8.1.
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.6-flash"
    gemini_timeout_seconds: float = 10.0
    gemini_max_output_tokens: int = 900
    gemini_temperature: float = 0.2
    gemini_max_retries: int = 1  # bounded; see app/gemini/service.py's retry-policy docstring


@lru_cache
def get_settings() -> Settings:
    return Settings()
