"""Centralized application configuration.

Every service reads configuration from this module instead of touching
``os.environ`` directly. Pydantic Settings loads values from environment
variables and from the project-level ``.env`` file, validates them, and gives
the rest of the code typed attributes.

Keeping configuration in one place also makes the backend easy to test: tests
can construct a ``Settings`` instance with different values without mutating
global process state.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> parents[2] is the backend directory and
# parents[3] is the repository root. Deriving paths from the file location
# means the application behaves the same no matter which directory it is
# started from.
BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent

# Qdrant collection names may only contain letters, digits, underscores and
# hyphens. We reuse the configured alias as the prefix of physical collections.
TEMP_COLLECTION_INFIX = "__ingest_"


class Settings(BaseSettings):
    """Typed representation of all runtime configuration.

    Environment variable names intentionally match the field names (for
    example ``GROQ_API_KEY`` -> ``groq_api_key``). Pydantic Settings performs
    that matching case-insensitively.
    """

    model_config = SettingsConfigDict(
        env_file=(PROJECT_DIR / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Groq -----------------------------------------------------------------
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    groq_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    groq_max_tokens: int = Field(default=1024, ge=128, le=8192)
    groq_timeout_seconds: float = Field(default=60.0, gt=0)

    # --- Qdrant Cloud ---------------------------------------------------------
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "askpolicy_documents"
    qdrant_timeout_seconds: int = Field(default=30, gt=0)

    # --- Application ----------------------------------------------------------
    app_env: str = "development"
    backend_host: str = "127.0.0.1"
    backend_port: int = Field(default=8000, ge=1, le=65535)
    frontend_origin: str = "http://localhost:5173"

    # --- Embeddings -----------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_dimension: int = Field(default=384, gt=0)

    # --- Chunking -------------------------------------------------------------
    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=200, ge=0)

    # --- Retrieval ------------------------------------------------------------
    retrieval_top_k: int = Field(default=5, ge=1, le=50)

    # --- Conversation memory --------------------------------------------------
    # The harness caps server-side memory at five question/answer exchanges.
    conversation_max_exchanges: int = Field(default=5, ge=1, le=20)

    # --- Paths ----------------------------------------------------------------
    documents_dir: Path = PROJECT_DIR / "documents"
    model_cache_dir: Path = BACKEND_DIR / ".cache" / "huggingface"

    @field_validator("qdrant_collection")
    @classmethod
    def _validate_collection(cls, value: str) -> str:
        """Reject empty collection names early so failures are easy to trace."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("QDRANT_COLLECTION must not be empty")
        return cleaned

    @model_validator(mode="after")
    def _validate_chunking(self) -> Settings:
        """Overlap must be smaller than the chunk size, otherwise splitting loops."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    @property
    def frontend_origins(self) -> list[str]:
        """Return allowed browser origins as a list.

        ``FRONTEND_ORIGIN`` accepts a comma-separated list so a developer can
        allow both ``localhost`` and ``127.0.0.1`` during local development.
        """
        origins = [origin.strip() for origin in self.frontend_origin.split(",")]
        return [origin for origin in origins if origin]

    @property
    def qdrant_is_configured(self) -> bool:
        return bool(self.qdrant_url.strip() and self.qdrant_api_key.strip())

    @property
    def groq_is_configured(self) -> bool:
        return bool(self.groq_api_key.strip() and self.groq_model.strip())

    def require_qdrant(self) -> None:
        """Raise a clear error when Qdrant credentials are missing.

        Validation happens when the relevant service is used rather than at
        import time, which keeps unit tests and the health endpoint usable
        without live credentials.
        """
        from app.core.errors import ConfigurationError

        if not self.qdrant_is_configured:
            raise ConfigurationError(
                "Qdrant is not configured. Set QDRANT_URL and QDRANT_API_KEY in .env."
            )

    def require_groq(self) -> None:
        from app.core.errors import ConfigurationError

        if not self.groq_is_configured:
            raise ConfigurationError(
                "Groq is not configured. Set GROQ_API_KEY and GROQ_MODEL in .env."
            )


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached ``Settings`` instance."""
    return Settings()


settings = get_settings()
