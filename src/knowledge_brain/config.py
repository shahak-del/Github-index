"""Configuration loading and validation.

Secrets are wrapped in ``pydantic.SecretStr`` so that logging, ``repr()``,
or accidental ``print()`` calls never leak their values. Use
``.get_secret_value()`` only at the point of use (building an API client).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REQUIRED_SECRET_FIELDS = ("dropbox_access_token", "qdrant_url", "qdrant_api_key")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Secrets / connection ---
    dropbox_access_token: SecretStr = Field(alias="DROPBOX_ACCESS_TOKEN")
    qdrant_url: SecretStr = Field(alias="QDRANT_URL")
    qdrant_api_key: SecretStr = Field(alias="QDRANT_API_KEY")

    # --- Index settings ---
    qdrant_collection: str = Field(default="dropbox_knowledge", alias="QDRANT_COLLECTION")
    source_name: str = Field(default="dropbox", alias="SOURCE_NAME")

    # --- Embeddings ---
    embedding_provider: str = Field(default="local", alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(
        default="intfloat/multilingual-e5-small", alias="EMBEDDING_MODEL"
    )
    embedding_api_key: SecretStr | None = Field(default=None, alias="EMBEDDING_API_KEY")

    # --- Chunking ---
    chunk_size_chars: int = Field(default=1800, alias="CHUNK_SIZE_CHARS")
    chunk_overlap_chars: int = Field(default=200, alias="CHUNK_OVERLAP_CHARS")

    # --- Batching / concurrency ---
    embedding_batch_size: int = Field(default=32, alias="EMBEDDING_BATCH_SIZE")
    qdrant_upsert_batch_size: int = Field(default=64, alias="QDRANT_UPSERT_BATCH_SIZE")
    download_concurrency: int = Field(default=4, alias="DOWNLOAD_CONCURRENCY")
    max_file_size_mb: int = Field(default=50, alias="MAX_FILE_SIZE_MB")

    # --- Local paths ---
    state_dir: Path = Field(default=Path("state"), alias="STATE_DIR")
    log_dir: Path = Field(default=Path("logs"), alias="LOG_DIR")
    dropbox_root_path: str = Field(default="", alias="DROPBOX_ROOT_PATH")

    @field_validator("embedding_provider")
    @classmethod
    def _lower_provider(cls, v: str) -> str:
        return v.strip().lower() or "local"

    @property
    def checkpoint_db_path(self) -> Path:
        return self.state_dir / "checkpoint.sqlite3"


def load_settings() -> Settings:
    """Load and validate settings from the environment / .env file.

    Never prints or logs secret values. Raises a clear, actionable error
    (naming only the missing *variable names*, never values) if required
    configuration is absent.
    """
    try:
        settings = Settings()  # type: ignore[call-arg]
    except Exception as exc:  # pydantic ValidationError
        missing = _missing_field_names(exc)
        if missing:
            raise ConfigError(
                "Missing required environment variable(s): "
                + ", ".join(missing)
                + ". Set them in .env (see .env.example)."
            ) from None
        raise ConfigError(f"Invalid configuration: {_safe_error(exc)}") from None

    settings.state_dir.mkdir(parents=True, exist_ok=True)
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    return settings


class ConfigError(RuntimeError):
    pass


def _missing_field_names(exc: Exception) -> list[str]:
    names = []
    for err in getattr(exc, "errors", lambda: [])():
        if err.get("type") == "missing":
            loc = err.get("loc", ())
            if loc:
                names.append(str(loc[0]).upper())
    return names


def _safe_error(exc: Exception) -> str:
    # Strip potential values pydantic includes in error messages by only
    # reporting field names / error types, never "input" values.
    parts = []
    for err in getattr(exc, "errors", lambda: [])():
        loc = ".".join(str(p) for p in err.get("loc", ()))
        parts.append(f"{loc}: {err.get('type')}")
    return "; ".join(parts) or str(type(exc).__name__)
