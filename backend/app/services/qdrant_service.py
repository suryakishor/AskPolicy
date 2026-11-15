"""Qdrant Cloud access layer.

The API talks to Qdrant through a *collection alias* (``askpolicy_documents``
by default). Ingestion builds a brand-new physical collection and atomically
moves the alias to it, so the backend never needs a restart after re-indexing:
every request simply resolves the alias again.

This module owns connection handling and reads that are not part of the RAG
search path (health checks and the indexed-document list).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import settings
from app.core.errors import (
    CollectionNotReadyError,
    QdrantConnectionError,
)
from app.core.logging import get_logger

logger = get_logger(__name__)

#: Number of points fetched per scroll page when aggregating document counts.
SCROLL_PAGE_SIZE = 512


@dataclass
class IndexedDocument:
    """Aggregated information about one indexed source file."""

    source: str
    file_name: str
    file_type: str
    chunk_count: int = 0


@dataclass
class IndexSummary:
    """Result of reading the current index."""

    documents: list[IndexedDocument] = field(default_factory=list)
    total_chunks: int = 0
    collection_name: str = ""


class QdrantService:
    """Lazy Qdrant client wrapper shared by the API routes.

    A ``QdrantClient`` is safe to reuse across requests, but constructing it at
    import time would make the app fail to start when credentials are missing
    or the network is down. The client is therefore created on first use and
    guarded by a lock.
    """

    def __init__(self, client: QdrantClient | None = None) -> None:
        self._client = client
        self._lock = threading.Lock()
        self.collection_name = settings.qdrant_collection

    def get_client(self) -> QdrantClient:
        """Return the shared client, creating it if necessary."""
        if self._client is None:
            with self._lock:
                if self._client is None:
                    settings.require_qdrant()
                    logger.info("Connecting to Qdrant at %s", settings.qdrant_url)
                    self._client = QdrantClient(
                        url=settings.qdrant_url,
                        api_key=settings.qdrant_api_key,
                        timeout=settings.qdrant_timeout_seconds,
                    )
        return self._client

    def set_client(self, client: QdrantClient) -> None:
        """Inject a client (used by tests and the ingestion command)."""
        self._client = client

    def collection_exists(self) -> bool:
        """Whether the configured alias currently resolves to a collection."""
        try:
            return self.get_client().collection_exists(self.collection_name)
        except Exception as exc:
            logger.warning("Qdrant collection check failed: %s", exc)
            raise QdrantConnectionError() from exc

    def get_index_summary(self) -> IndexSummary:
        """Read actual document metadata from Qdrant to build ``GET /api/documents``.

        There is no Postgres or other metadata store in Version 1, so the vector
        payload is the single source of truth. We scroll through the collection
        (payload only, no vectors) and aggregate chunk counts per source file.
        """
        if not self.collection_exists():
            raise CollectionNotReadyError()

        client = self.get_client()
        documents: dict[str, IndexedDocument] = {}
        total_chunks = 0
        offset = None

        try:
            while True:
                points, offset = client.scroll(
                    collection_name=self.collection_name,
                    limit=SCROLL_PAGE_SIZE,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in points:
                    payload = point.payload or {}
                    metadata = payload.get("metadata") or {}
                    source = str(metadata.get("source") or metadata.get("file_name") or "")
                    if not source:
                        continue
                    file_name = str(metadata.get("file_name") or source.rsplit("/", 1)[-1])
                    file_type = str(
                        metadata.get("file_type") or file_name.rsplit(".", 1)[-1].lower()
                    )
                    entry = documents.get(source)
                    if entry is None:
                        entry = IndexedDocument(
                            source=source,
                            file_name=file_name,
                            file_type=file_type,
                        )
                        documents[source] = entry
                    entry.chunk_count += 1
                    total_chunks += 1

                if offset is None:
                    break
        except UnexpectedResponse as exc:
            logger.exception("Qdrant scroll failed")
            raise QdrantConnectionError() from exc
        except Exception as exc:
            logger.exception("Unexpected error while reading the Qdrant index")
            raise QdrantConnectionError() from exc

        ordered = sorted(documents.values(), key=lambda doc: doc.source.lower())
        return IndexSummary(
            documents=ordered,
            total_chunks=total_chunks,
            collection_name=self.collection_name,
        )

    def health(self) -> dict[str, object]:
        """Return non-sensitive connection details for ``GET /api/health``."""
        details: dict[str, object] = {
            "configured": settings.qdrant_is_configured,
            "collection": self.collection_name,
            "connected": False,
            "collection_ready": False,
        }
        if not settings.qdrant_is_configured:
            details["detail"] = "QDRANT_URL or QDRANT_API_KEY is missing"
            return details

        try:
            client = self.get_client()
            # A cheap API call proves credentials and network connectivity.
            client.get_collections()
            details["connected"] = True
            details["collection_ready"] = client.collection_exists(self.collection_name)
        except Exception as exc:
            logger.warning("Qdrant health check failed: %s", exc)
            details["detail"] = "Qdrant connection failed"
        return details


def get_qdrant_service() -> QdrantService:
    """Process-wide Qdrant service used by API dependencies."""
    return _qdrant_service


_qdrant_service = QdrantService()
