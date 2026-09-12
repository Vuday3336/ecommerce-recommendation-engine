"""Typed application settings.

Every value comes from the environment (or a `.env` file at the repository root).
Nothing in this module has a usable default for a secret: if a secret is missing
the application fails loudly at startup rather than silently running insecurely.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import PostgresDsn, RedisDsn, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]

Environment = Literal["development", "staging", "production", "test"]


class Settings(BaseSettings):
    """Application configuration resolved from environment variables."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Application ---------------------------------------------------------
    app_env: Environment = "development"
    app_name: str = "recommendation-platform"
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"

    # --- API -----------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_v1_prefix: str = "/api/v1"
    cors_allowed_origins: str = "http://localhost:5173"

    # --- Security ------------------------------------------------------------
    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 14

    # --- PostgreSQL ----------------------------------------------------------
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "recommendations"
    postgres_user: str = "recsys"
    postgres_password: str = ""
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_echo: bool = False

    # --- Rate limiting -------------------------------------------------------
    rate_limit_auth_per_minute: int = 10
    rate_limit_events_per_minute: int = 600
    rate_limit_recommendations_per_minute: int = 120

    # --- Redis ---------------------------------------------------------------
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    cache_ttl_home_seconds: int = 900
    cache_ttl_similar_seconds: int = 21_600
    cache_ttl_trending_seconds: int = 300

    # --- Recommendation engine -----------------------------------------------
    artifact_dir: Path = REPO_ROOT / "ml" / "artifacts"
    dataset_dir: Path = REPO_ROOT / "data" / "synthetic"
    recommendations_per_section: int = 12
    enable_ml_ranker: bool = True
    model_name: str = "recsys-ranker"
    model_version: str = "v1"

    # --- Synthetic data ------------------------------------------------------
    seed: int = 42
    n_users: int = 10_000
    n_products: int = 5_000
    n_events: int = 250_000
    simulation_days: int = 180

    @field_validator("log_level")
    @classmethod
    def _upper(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = value.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return upper

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url(self) -> str:
        """Synchronous SQLAlchemy URL (psycopg2 driver)."""
        return str(
            PostgresDsn.build(
                scheme="postgresql+psycopg2",
                username=self.postgres_user,
                password=self.postgres_password or None,
                host=self.postgres_host,
                port=self.postgres_port,
                path=self.postgres_db,
            )
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def redis_url(self) -> str:
        return str(
            RedisDsn.build(
                scheme="redis",
                password=self.redis_password or None,
                host=self.redis_host,
                port=self.redis_port,
                path=str(self.redis_db),
            )
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]

    def require_secrets(self) -> None:
        """Fail fast when running outside development without real secrets."""
        if self.app_env == "development":
            return
        missing = [
            name
            for name, value in (
                ("JWT_SECRET_KEY", self.jwt_secret_key),
                ("POSTGRES_PASSWORD", self.postgres_password),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                f"Missing required secrets for app_env={self.app_env}: {', '.join(missing)}"
            )


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor. Used as a FastAPI dependency from Phase 10."""
    return Settings()


settings = get_settings()
