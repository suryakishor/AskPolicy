"""Local embedding generation with ``all-MiniLM-L6-v2``.

The same embedding model is used for document chunks during ingestion and for
user questions during retrieval. Both vectors must live in the same semantic
space for cosine similarity to be meaningful.

The model runs on CPU by default and is loaded lazily: starting the API should
not require downloading or loading a model, and tests that never touch
embeddings should stay fast.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

# Importing this module configures HF_HOME / SENTENCE_TRANSFORMERS_HOME before
# sentence-transformers reads them. Keep this import at the top.
from app.core import model_cache  # noqa: F401  (imported for its side effect)
from app.core.config import BACKEND_DIR, settings
from app.core.errors import ConfigurationError
from app.core.logging import get_logger

if TYPE_CHECKING:
    from langchain_huggingface import HuggingFaceEmbeddings

logger = get_logger(__name__)

#: Sentence Transformers' ``encode`` accepts a batch size; 32 keeps memory
#: usage modest on CPU while still giving decent throughput.
EMBEDDING_BATCH_SIZE = 32


class EmbeddingService:
    """Thin, thread-safe wrapper around LangChain's ``HuggingFaceEmbeddings``.

    Notes:
        ``all-MiniLM-L6-v2`` produces 384-dimensional vectors and has a maximum
        input length of 256 word pieces. The ingestion chunker works in
        characters (1000 by default); for normal English prose that is well
        below the model limit, but the model will silently truncate unusually
        dense text. See ``docs/ARCHITECTURE.md`` for the character-vs-token
        explanation.
    """

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
        """Return the initialized embedding client, loading it on first use.

        ``HuggingFaceEmbeddings`` is imported lazily so that importing the
        FastAPI app does not pull in PyTorch/Sentence Transformers. The model
        cache configuration at the top of this module still runs first.
        """
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
        """Run a tiny embedding and assert the model's vector size.

        Catching a dimension mismatch during ingestion is much cheaper than
        discovering it after creating a collection with the wrong size.
        """
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
