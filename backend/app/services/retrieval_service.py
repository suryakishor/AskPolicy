"""Dense vector retrieval from Qdrant via LangChain's Qdrant integration.

Retrieval embeds the question, asks Qdrant for the nearest chunks by cosine
similarity, and optionally filters by source file.
"""

from __future__ import annotations

import threading

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http import models as qdrant_models

from app.core.config import settings
from app.core.errors import CollectionNotReadyError, RetrievalError
from app.core.logging import get_logger
from app.schemas.documents import RetrievedChunk
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.qdrant_service import QdrantService, get_qdrant_service

logger = get_logger(__name__)


class RetrievalService:
    """Perform similarity search against the active Qdrant collection."""

    def __init__(
        self,
        qdrant_service: QdrantService | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self.qdrant = qdrant_service or get_qdrant_service()
        self.embeddings = embedding_service or get_embedding_service()
        self._store: QdrantVectorStore | None = None
        self._lock = threading.Lock()

    def _get_store(self) -> QdrantVectorStore:
        """Create the LangChain vector store on first use, bound to the alias."""
        if self._store is None:
            with self._lock:
                if self._store is None:
                    if not self.qdrant.collection_exists():
                        raise CollectionNotReadyError()
                    self._store = QdrantVectorStore(
                        client=self.qdrant.get_client(),
                        collection_name=self.qdrant.collection_name,
                        embedding=self.embeddings.get_embeddings(),
                        # Payload keys written by ingestion match LangChain's
                        # defaults: page_content + metadata.
                        content_payload_key="page_content",
                        metadata_payload_key="metadata",
                    )
        return self._store

    @staticmethod
    def _build_filter(source: str | None) -> qdrant_models.Filter | None:
        """Build a Qdrant filter on ``metadata.source`` for an optional document."""
        if not source:
            return None
        return qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="metadata.source",
                    match=qdrant_models.MatchValue(value=source),
                )
            ]
        )

    def retrieve(
        self,
        question: str,
        source: str | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Return the most relevant chunks for a question, ordered by similarity."""
        cleaned = question.strip()
        if not cleaned:
            return []

        k = top_k or settings.retrieval_top_k
        try:
            store = self._get_store()
            results: list[tuple[Document, float]] = store.similarity_search_with_score(
                cleaned,
                k=k,
                filter=self._build_filter(source),
            )
        except CollectionNotReadyError:
            raise
        except Exception as exc:
            logger.exception("Vector retrieval failed")
            raise RetrievalError() from exc

        chunks = [self._to_chunk(document, score) for document, score in results]
        logger.info(
            "Retrieved %d chunks (top_k=%d, source=%s)",
            len(chunks),
            k,
            source or "all",
        )
        return chunks

    @staticmethod
    def _to_chunk(document: Document, score: float) -> RetrievedChunk:
        """Map a LangChain document + score to the public API schema."""
        metadata = document.metadata or {}
        page_number = metadata.get("page_number")
        return RetrievedChunk(
            file_name=str(metadata.get("file_name") or "unknown"),
            source=str(metadata.get("source") or ""),
            file_type=str(metadata.get("file_type") or ""),
            page_number=int(page_number) if page_number is not None else None,
            chunk_index=int(metadata.get("chunk_index", 0)),
            content=document.page_content,
            score=float(score),
        )


def get_retrieval_service() -> RetrievalService:
    """FastAPI dependency: process-wide retrieval service."""
    return _retrieval_service


_retrieval_service = RetrievalService()
