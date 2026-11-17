"""Vector index deletion tests against an in-memory Qdrant instance."""

from __future__ import annotations

from pathlib import Path

from app.core.config import TEMP_COLLECTION_INFIX
from ingestion.delete_index import DeletionStats, delete_index, format_summary
from ingestion.ingest import run_ingestion
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

ALIAS = "askpolicy_delete_test"


class FakeEmbeddingService:
    """Deterministic 384-dimensional embeddings without loading a model."""

    dimension = 384

    def verify_dimension(self, expected: int) -> int:
        assert expected == self.dimension
        return self.dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * (self.dimension - 1) for _ in texts]


def write_document(root: Path, name: str, content: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def alias_target(client: QdrantClient) -> str | None:
    for alias in client.get_aliases().aliases:
        if alias.alias_name == ALIAS:
            return alias.collection_name
    return None


def seed_index(client: QdrantClient, tmp_path: Path) -> str:
    """Ingest one document and return the active physical collection name."""
    root = tmp_path / "documents"
    root.mkdir()
    write_document(root, "leave.txt", "Employees receive 20 days of annual leave. " * 40)
    run_ingestion(
        root, client=client, embedding_service=FakeEmbeddingService(), collection_alias=ALIAS
    )
    target = alias_target(client)
    assert target is not None
    return target


def test_delete_removes_alias_and_physical_collection(tmp_path: Path) -> None:
    client = QdrantClient(":memory:")
    physical = seed_index(client, tmp_path)

    stats = delete_index(client, alias=ALIAS)

    assert stats.deleted_alias is True
    assert stats.deleted_collection == physical
    assert alias_target(client) is None
    assert client.collection_exists(physical) is False
    assert client.get_collections().collections == []


def test_delete_removes_leftover_temp_collections(tmp_path: Path) -> None:
    client = QdrantClient(":memory:")
    seed_index(client, tmp_path)

    orphan = f"{ALIAS}{TEMP_COLLECTION_INFIX}deadbeef"
    client.create_collection(
        collection_name=orphan,
        vectors_config=qdrant_models.VectorParams(size=384, distance=qdrant_models.Distance.COSINE),
    )

    stats = delete_index(client, alias=ALIAS)

    assert orphan in stats.deleted_temp_collections
    assert client.collection_exists(orphan) is False
    assert client.get_collections().collections == []


def test_delete_without_alias_is_safe() -> None:
    client = QdrantClient(":memory:")

    stats = delete_index(client, alias=ALIAS)

    assert stats.deleted_alias is False
    assert stats.deleted_collection is None
    assert "nothing to delete" in format_summary(stats)


def test_format_summary_reports_deleted_collection() -> None:
    stats = DeletionStats(
        collection=ALIAS,
        deleted_alias=True,
        deleted_collection="askpolicy_delete_test__ingest_abc123",
    )

    summary = format_summary(stats)

    assert f"Collection alias: {ALIAS}" in summary
    assert "Deleted alias and collection: askpolicy_delete_test__ingest_abc123" in summary
    assert "Deletion completed." in summary
