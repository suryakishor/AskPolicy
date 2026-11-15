"""End-to-end ingestion tests against an in-memory Qdrant instance.

Embeddings are faked, but collection creation, uploads, validation, alias
switching, cleanup, and payload metadata all use the real Qdrant client.
"""

from __future__ import annotations

from pathlib import Path

import ingestion.ingest as ingest_module
import pytest
from app.core.errors import IngestionError, NoDocumentsFoundError
from ingestion.ingest import format_summary, run_ingestion
from qdrant_client import QdrantClient

ALIAS = "askpolicy_test"


class FakeEmbeddingService:
    """Deterministic 384-dimensional embeddings without loading a model."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.embedded_texts: list[str] = []

    def verify_dimension(self, expected: int) -> int:
        assert expected == self.dimension
        return self.dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded_texts.extend(texts)
        return [[1.0] + [0.0] * (self.dimension - 1) for _ in texts]


def write_documents(root: Path, files: dict[str, str]) -> None:
    for relative_path, content in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def alias_target(client: QdrantClient) -> str | None:
    for alias in client.get_aliases().aliases:
        if alias.alias_name == ALIAS:
            return alias.collection_name
    return None


def collection_count(client: QdrantClient) -> int:
    return len(client.get_collections().collections)


def payload_sources(client: QdrantClient, collection: str) -> set[str]:
    records, _ = client.scroll(
        collection_name=collection,
        limit=100,
        with_payload=True,
        with_vectors=False,
    )
    sources = set()
    for record in records:
        metadata = (record.payload or {}).get("metadata", {})
        sources.add(metadata.get("source"))
    return sources


def test_ingestion_creates_alias_and_stores_real_metadata(tmp_path: Path) -> None:
    root = tmp_path / "documents"
    root.mkdir()
    write_documents(
        root,
        {
            "leave.txt": "Employees receive 20 days of annual leave. " * 100,
            "nested/travel.txt": "Book economy flights for approved travel. " * 100,
            "empty.txt": "    ",
            "ignored.md": "not a supported policy file",
        },
    )
    client = QdrantClient(":memory:")
    embedder = FakeEmbeddingService()

    stats = run_ingestion(root, client=client, embedding_service=embedder, collection_alias=ALIAS)

    assert stats.discovered == 3
    assert stats.processed == 2
    assert stats.skipped == 1
    assert stats.chunks > 2
    assert stats.replaced_previous is False
    assert stats.collection == ALIAS
    assert len(embedder.embedded_texts) == stats.chunks

    target = alias_target(client)
    assert target is not None
    assert client.count(collection_name=ALIAS, exact=True).count == stats.chunks
    assert payload_sources(client, ALIAS) == {"leave.txt", "nested/travel.txt"}

    records, _ = client.scroll(
        collection_name=ALIAS,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    payload = records[0].payload or {}
    metadata = payload["metadata"]
    assert payload["page_content"]
    assert metadata["file_type"] == "txt"
    assert "chunk_index" in metadata
    # TXT documents do not have pages, so no page_number key is invented.
    assert "page_number" not in metadata
    assert collection_count(client) == 1

    summary = format_summary(stats)
    assert f"Documents discovered: {stats.discovered}" in summary
    assert f"Total chunks generated: {stats.chunks}" in summary
    assert "Qdrant Cloud" in summary


def test_second_ingestion_replaces_previous_collection(tmp_path: Path) -> None:
    client = QdrantClient(":memory:")
    embedder = FakeEmbeddingService()

    first_root = tmp_path / "first"
    first_root.mkdir()
    write_documents(first_root, {"leave.txt": "Old leave policy content. " * 30})
    run_ingestion(first_root, client=client, embedding_service=embedder, collection_alias=ALIAS)
    first_target = alias_target(client)

    second_root = tmp_path / "second"
    second_root.mkdir()
    write_documents(
        second_root,
        {"reimbursement.txt": "New reimbursement process details. " * 30},
    )
    stats = run_ingestion(
        second_root, client=client, embedding_service=embedder, collection_alias=ALIAS
    )

    second_target = alias_target(client)
    assert stats.replaced_previous is True
    assert second_target is not None
    assert second_target != first_target
    assert client.collection_exists(first_target) is False
    assert collection_count(client) == 1
    assert payload_sources(client, ALIAS) == {"reimbursement.txt"}


def test_no_documents_keeps_existing_index(tmp_path: Path) -> None:
    client = QdrantClient(":memory:")
    embedder = FakeEmbeddingService()

    valid_root = tmp_path / "valid"
    valid_root.mkdir()
    write_documents(valid_root, {"leave.txt": "Leave policy. " * 20})
    run_ingestion(valid_root, client=client, embedding_service=embedder, collection_alias=ALIAS)
    previous_target = alias_target(client)
    previous_count = client.count(collection_name=ALIAS, exact=True).count

    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    write_documents(empty_root, {"empty.txt": "   ", "notes.md": "unsupported"})

    with pytest.raises(NoDocumentsFoundError):
        run_ingestion(empty_root, client=client, embedding_service=embedder, collection_alias=ALIAS)

    assert alias_target(client) == previous_target
    assert client.count(collection_name=ALIAS, exact=True).count == previous_count


def test_failed_upload_keeps_previous_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = QdrantClient(":memory:")
    embedder = FakeEmbeddingService()

    first_root = tmp_path / "first"
    first_root.mkdir()
    write_documents(first_root, {"leave.txt": "Stable leave policy. " * 20})
    run_ingestion(first_root, client=client, embedding_service=embedder, collection_alias=ALIAS)
    previous_target = alias_target(client)
    previous_count = client.count(collection_name=ALIAS, exact=True).count

    def fail_upload(*args, **kwargs) -> None:
        raise RuntimeError("simulated upload failure")

    monkeypatch.setattr(ingest_module, "_upsert_chunks", fail_upload)

    second_root = tmp_path / "second"
    second_root.mkdir()
    write_documents(second_root, {"travel.txt": "New travel policy. " * 20})

    with pytest.raises(IngestionError):
        run_ingestion(
            second_root, client=client, embedding_service=embedder, collection_alias=ALIAS
        )

    # Alias still serves the old data and the temp collection was discarded.
    assert alias_target(client) == previous_target
    assert client.count(collection_name=ALIAS, exact=True).count == previous_count
    assert collection_count(client) == 1


def test_alias_switch_failure_restores_previous_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = QdrantClient(":memory:")
    embedder = FakeEmbeddingService()

    first_root = tmp_path / "first"
    first_root.mkdir()
    write_documents(first_root, {"leave.txt": "Stable leave policy. " * 20})
    run_ingestion(first_root, client=client, embedding_service=embedder, collection_alias=ALIAS)
    previous_target = alias_target(client)

    def fail_switch(*args, **kwargs):
        raise RuntimeError("simulated alias failure")

    monkeypatch.setattr(ingest_module, "switch_alias", fail_switch)

    second_root = tmp_path / "second"
    second_root.mkdir()
    write_documents(second_root, {"travel.txt": "New travel policy. " * 20})

    with pytest.raises(IngestionError):
        run_ingestion(
            second_root, client=client, embedding_service=embedder, collection_alias=ALIAS
        )

    assert alias_target(client) == previous_target
    assert client.collection_exists(previous_target) is True
    assert collection_count(client) == 1
