"""Local embedding generation with ``all-MiniLM-L6-v2``.

The same model embeds document chunks during ingestion and questions during
retrieval, so both share one semantic space. It runs on CPU and loads lazily.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

# Configures the HF cache before sentence-transformers reads it.
from app.core import model_cache  # noqa: F401  (imported for its side effect)
from app.core.config import BACKEND_DIR, settings
from app.core.errors import ConfigurationError
from app.core.logging import get_logger

if TYPE_CHECKING:
    from langchain_huggingface import HuggingFaceEmbeddings

logger = get_logger(__name__)

#: Batch size for encoding; 32 keeps CPU memory modest.
EMBEDDING_BATCH_SIZE = 32


class EmbeddingService:
    """Thread-safe wrapper around LangChain's ``HuggingFaceEmbeddings``."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device
        self._embeddings: HuggingFaceEmbeddings | None = None
        self._lock = threading.Lock()

    @property
    def cache_folder(self) -> str:
        """Project-managed directory where model weights are stored."""
        cache_root = settings.model_cache_dir
        if not cache_root.is_absolute():
            cache_root = BACKEND_DIR / cache_root
        return str(cache_root / "sentence_transformers")

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """Return the embedding client, loading it on first use."""
        if self._embeddings is None:
            with self._lock:
                if self._embeddings is None:
                    from langchain_huggingface import HuggingFaceEmbeddings

                    logger.info(
                        "Loading embedding model '%s' on %s (cache: %s)",
                        self.model_name,
                        self.device,
                        self.cache_folder,
                    )
                    self._embeddings = HuggingFaceEmbeddings(
                        model_name=self.model_name,
                        cache_folder=self.cache_folder,
                        model_kwargs={"device": self.device},
                        encode_kwargs={
                            "normalize_embeddings": True,
                            "batch_size": EMBEDDING_BATCH_SIZE,
                        },
                    )
                    logger.info("Embedding model loaded")
        return self._embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of chunks (used by ingestion)."""
        if not texts:
            return []
        return self.get_embeddings().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single user question (used by retrieval)."""
        cleaned = text.strip()
        if not cleaned:
            raise ConfigurationError("Cannot embed an empty question.")
        return self.get_embeddings().embed_query(cleaned)

    def verify_dimension(self, expected: int) -> int:
        """Embed a probe string and assert the model's vector size."""
        vector = self.embed_query("dimension check")
        actual = len(vector)
        if actual != expected:
            raise ConfigurationError(
                f"Embedding dimension mismatch: model '{self.model_name}' "
                f"returned {actual} values but the collection expects {expected}."
            )
        return actual


def get_embedding_service() -> EmbeddingService:
    """FastAPI dependency: process-wide embedding service."""
    return _embedding_service


_embedding_service = EmbeddingService()
