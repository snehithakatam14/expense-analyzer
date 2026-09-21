"""
config.py — Type-safe environment variable management.

Uses pydantic-settings BaseSettings for automatic .env loading,
type coercion, and validation. Call get_settings() anywhere to
get the singleton config object.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables or a .env file.

    All fields are type-validated at startup; the app will fail fast if
    required vars are missing or malformed.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Silently ignore unrecognized env vars
    )

    # ─────────────────────────────────────────────────────────────
    # OpenAI
    # ─────────────────────────────────────────────────────────────
    openai_api_key: str = Field(
        ...,
        description="OpenAI secret API key (required).",
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model identifier for categorization.",
    )
    openai_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Per-request timeout for OpenAI API calls (seconds).",
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts for transient API failures.",
    )

    # ─────────────────────────────────────────────────────────────
    # Database
    # ─────────────────────────────────────────────────────────────
    database_url: str = Field(
        default="sqlite:///./expense_analyzer.db",
        description="SQLAlchemy-compatible database URL.",
    )

    # ─────────────────────────────────────────────────────────────
    # Flask
    # ─────────────────────────────────────────────────────────────
    flask_env: Literal["development", "production", "testing"] = Field(
        default="development",
        description="Flask runtime environment.",
    )
    flask_debug: bool = Field(
        default=False,
        description="Enable Flask debug mode (never True in production).",
    )
    flask_secret_key: str = Field(
        default="change-me-in-production-use-a-long-random-string",
        description="Flask session secret key — must be changed in production.",
    )

    # ─────────────────────────────────────────────────────────────
    # Parser / Categorizer
    # ─────────────────────────────────────────────────────────────
    categorizer_batch_size: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of transactions sent per OpenAI API call.",
    )
    anomaly_zscore_threshold: float = Field(
        default=2.0,
        ge=0.5,
        le=5.0,
        description="Z-score threshold above which a transaction is an anomaly.",
    )
    max_upload_size_mb: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum file upload size in megabytes.",
    )

    # ─────────────────────────────────────────────────────────────
    # Validators
    # ─────────────────────────────────────────────────────────────

    @field_validator("openai_api_key")
    @classmethod
    def api_key_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("OPENAI_API_KEY must not be empty.")
        return v.strip()

    @field_validator("flask_secret_key")
    @classmethod
    def warn_default_secret(cls, v: str) -> str:
        # We allow the default in dev/testing but the app can emit a warning.
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.

    Cached after first call — call ``get_settings.cache_clear()``
    in tests that need a fresh config.
    """
    return Settings()
