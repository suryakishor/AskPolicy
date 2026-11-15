"""Retrieval tests using an in-memory Qdrant instance and fake embeddings.

No Qdrant Cloud credentials, embedding model download, or network access is
required: the Qdrant client's local mode provides the same collection/filter
API used in production.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from app.core.errors import CollectionNotReadyError
from app.services.qdrant_service import QdrantService
from app.services.retrieval_service import RetrievalService
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

COLLECTION = "policy_test"


class KeywordEmbeddings(Embeddings):
    """Deterministic embeddings so tests can assert semantic ordering."""

    def __init__(self) -> None:
        self.vectors = {
            "leave": [1.0, 0.0, 0.0, 0.0],
            "annual": [1.0, 0.0, 0.0, 0.0],
            "travel": [0.0, 1.0, 0.0, 0.0],
            "flight": [0.0, 1.0, 0.0, 0.0],
        }

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        for keyword, vector in self.vectors.items():
            if keyword in lowered:
                return vector
        return [0.0, 0.0, 1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeEmbeddingService:
    """Duck-typed stand-in for EmbeddingService."""

    def __init__(self) -> None:
        self.embeddings = KeywordEmbeddings()

    def get_embeddings(self) -> Embeddings:
        return self.embeddings


def payload(content: str, source: str, page: int | None, chunk: int) -> dict:
    metadata: dict[str, object] = {
        "source": source,
        "file_name": source.rsplit("/", 1)[-1],
        "file_type": "pdf" if page is not None else "txt",
        "chunk_index": chunk,
    }
    if page is not None:
        metadata["page_number"] = page
    return {"page_content": content, "metadata": metadata}


@pytest.fixture
def qdrant() -> QdrantService:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=qdrant_models.VectorParams(
            size=4,
            distance=qdrant_models.Distance.COSINE,
        ),
    )
    client.upsert(
        collection_name=COLLECTION,
        wait=True,
        points=[
            qdrant_models.PointStruct(
                id=str(uuid4()),
                vector=[1.0, 0.0, 0.0, 0.0],
                payload=payload(
                    "Employees receive 20 days of annual leave.",
                    "hr/leave_policy.pdf",
                    2,
                    0,
                ),
            ),
            qdrant_models.PointStruct(
                id=str(uuid4()),
                vector=[0.85, 0.15, 0.0, 0.0],
                payload=payload(
                    "Leave requests must be approved by a manager.",
                    "hr/leave_policy.pdf",
                    3,
                    1,
                ),
            ),
            qdrant_models.PointStruct(
                id=str(uuid4()),
                vector=[0.0, 1.0, 0.0, 0.0],
                payload=payload(
                    "Economy flights are booked for approved travel.",
                    "travel_policy.txt",
                    None,
                    0,
                ),
            ),
        ],
    )
    service = QdrantService(client)
    service.collection_name = COLLECTION
    return service


def test_similarity_search_returns_relevant_chunks(qdrant: QdrantService) -> None:
    retrieval = RetrievalService(qdrant, FakeEmbeddingService())

    chunks = retrieval.retrieve("What is the annual leave allowance?", top_k=2)

    assert len(chunks) == 2
    assert all("leave" in chunk.content.lower() for chunk in chunks)
    assert chunks[0].file_name == "leave_policy.pdf"
    assert chunks[0].page_number == 2
    assert chunks[0].source == "hr/leave_policy.pdf"
    assert chunks[0].score >= chunks[1].score


def test_metadata_filter_limits_search_to_one_document(
    qdrant: QdrantService,
) -> None:
    retrieval = RetrievalService(qdrant, FakeEmbeddingService())

    chunks = retrieval.retrieve("annual leave", source="travel_policy.txt", top_k=5)

    assert len(chunks) == 1
    assert chunks[0].source == "travel_policy.txt"
    # TXT files must not receive a fabricated page number.
    assert chunks[0].page_number is None
    assert chunks[0].file_name == "travel_policy.txt"


def test_filter_with_no_matching_document_returns_empty(
    qdrant: QdrantService,
) -> None:
    retrieval = RetrievalService(qdrant, FakeEmbeddingService())

    assert retrieval.retrieve("leave", source="missing.pdf", top_k=5) == []


def test_missing_collection_raises_domain_error(qdrant: QdrantService) -> None:
    qdrant.collection_name = "does_not_exist"
    retrieval = RetrievalService(qdrant, FakeEmbeddingService())

    with pytest.raises(CollectionNotReadyError):
        retrieval.retrieve("leave")


def test_empty_collection_returns_empty_list() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="empty_collection",
        vectors_config=qdrant_models.VectorParams(
            size=4,
            distance=qdrant_models.Distance.COSINE,
        ),
    )
    qdrant = QdrantService(client)
    qdrant.collection_name = "empty_collection"
    retrieval = RetrievalService(qdrant, FakeEmbeddingService())

    assert retrieval.retrieve("leave", top_k=5) == []


def test_index_summary_aggregates_real_payloads(qdrant: QdrantService) -> None:
    summary = qdrant.get_index_summary()

    assert summary.total_chunks == 3
    assert summary.collection_name == COLLECTION
    documents = {document.source: document for document in summary.documents}
    assert set(documents) == {"hr/leave_policy.pdf", "travel_policy.txt"}
    assert documents["hr/leave_policy.pdf"].chunk_count == 2
    assert documents["hr/leave_policy.pdf"].file_name == "leave_policy.pdf"
    assert documents["hr/leave_policy.pdf"].file_type == "pdf"
    assert documents["travel_policy.txt"].chunk_count == 1
