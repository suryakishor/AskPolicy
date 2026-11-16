"""AskPolicy document ingestion CLI.

Run from the ``backend`` directory:

    uv run python -m ingestion.ingest

Ingestion builds a temporary collection, validates it, moves the alias in one
request, and only then removes the old one. On failure the previous index stays active.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from app.core.config import TEMP_COLLECTION_INFIX, settings
from app.core.errors import (
    IngestionError,
    NoDocumentsFoundError,
    QdrantConnectionError,
)
from app.core.logging import configure_logging, get_logger
from app.core.model_cache import configure_model_cache
from app.services.embedding_service import EmbeddingService
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from ingestion.chunker import build_splitter, chunk_documents
from ingestion.loaders import discover_documents, load_document

logger = get_logger(__name__)

#: Points uploaded per Qdrant request.
UPLOAD_BATCH_SIZE = 64


@dataclass
class IngestionStats:
    """Facts collected during a successful ingestion run."""

    discovered: int
    processed: int
    skipped: int
    pdf_pages: int
    chunks: int
    embedding_model: str
    collection: str
    replaced_previous: bool


# --------------------------------------------------------------------------- #
# Qdrant collection lifecycle
# --------------------------------------------------------------------------- #
def create_qdrant_client() -> QdrantClient:
    """Create a client from validated environment settings."""
    settings.require_qdrant()
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=settings.qdrant_timeout_seconds,
    )


def _temporary_collection_name(alias: str) -> str:
    """Generate a unique physical collection name for this ingestion run."""
    return f"{alias}{TEMP_COLLECTION_INFIX}{uuid4().hex[:12]}"


def _create_collection(client: QdrantClient, name: str, dimension: int) -> None:
    """Create a cosine-distance collection sized for the embedding model."""
    client.create_collection(
        collection_name=name,
        vectors_config=qdrant_models.VectorParams(
            size=dimension,
            distance=qdrant_models.Distance.COSINE,
        ),
    )


def _upsert_chunks(
    client: QdrantClient,
    collection_name: str,
    chunks: list[Document],
    vectors: list[list[float]],
    batch_size: int = UPLOAD_BATCH_SIZE,
) -> None:
    """Upload chunks and vectors in batches using LangChain's payload shape."""
    if len(chunks) != len(vectors):
        raise IngestionError("Internal error: chunk and vector counts differ.")

    total = len(chunks)
    for start in range(0, total, batch_size):
        batch_chunks = chunks[start : start + batch_size]
        batch_vectors = vectors[start : start + batch_size]
        points = [
            qdrant_models.PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload={
                    "page_content": chunk.page_content,
                    "metadata": chunk.metadata,
                },
            )
            for chunk, vector in zip(batch_chunks, batch_vectors, strict=True)
        ]
        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )
        logger.info("Uploaded %d/%d chunks", min(start + batch_size, total), total)


def _validate_collection(
    client: QdrantClient,
    collection_name: str,
    expected_count: int,
    sample_vector: list[float],
) -> None:
    """Prove the upload is complete and searchable before switching aliases."""
    actual = client.count(collection_name=collection_name, exact=True).count
    if actual != expected_count:
        raise IngestionError(
            f"Validation failed: expected {expected_count} points, found {actual}."
        )

    # A real query confirms the points are searchable, not just stored.
    probe = client.query_points(
        collection_name=collection_name,
        query=sample_vector,
        limit=1,
        with_payload=False,
    )
    if not probe.points:
        raise IngestionError("Validation failed: the new collection returned no results.")


def _alias_targets(client: QdrantClient) -> dict[str, str]:
    """Return a mapping of alias name -> physical collection name."""
    return {alias.alias_name: alias.collection_name for alias in client.get_aliases().aliases}


def switch_alias(
    client: QdrantClient,
    alias: str,
    new_collection: str,
) -> str | None:
    """Move ``alias`` to ``new_collection`` and return its previous target."""
    old_target = _alias_targets(client).get(alias)

    operations: list[qdrant_models.AliasOperations] = []
    if old_target:
        operations.append(
            qdrant_models.DeleteAliasOperation(
                delete_alias=qdrant_models.DeleteAlias(alias_name=alias)
            )
        )
    operations.append(
        qdrant_models.CreateAliasOperation(
            create_alias=qdrant_models.CreateAlias(
                collection_name=new_collection,
                alias_name=alias,
            )
        )
    )

    try:
        client.update_collection_aliases(change_aliases_operations=operations)
    except Exception:
        # The delete may have applied even though the create failed; restore it.
        if old_target:
            try:
                client.update_collection_aliases(
                    change_aliases_operations=[
                        qdrant_models.CreateAliasOperation(
                            create_alias=qdrant_models.CreateAlias(
                                collection_name=old_target,
                                alias_name=alias,
                            )
                        )
                    ]
                )
                logger.warning("Restored alias '%s' to '%s'", alias, old_target)
            except Exception:
                logger.exception(
                    "Could not restore alias '%s'; manual attention required",
                    alias,
                )
        raise

    logger.info("Alias '%s' now points to '%s'", alias, new_collection)
    return old_target


def cleanup_old_collections(
    client: QdrantClient,
    alias: str,
    active_collection: str,
    previous_collection: str | None,
) -> None:
    """Delete the previous collection and any orphaned ingestion collections."""
    protected = {active_collection}
    protected.update(_alias_targets(client).values())

    try:
        all_collections = [item.name for item in client.get_collections().collections]
    except Exception as exc:
        logger.warning("Could not list collections during cleanup: %s", exc)
        return

    for name in all_collections:
        if name in protected:
            continue
        is_previous = previous_collection is not None and name == previous_collection
        is_orphan_temp = name.startswith(f"{alias}{TEMP_COLLECTION_INFIX}")
        if not (is_previous or is_orphan_temp):
            continue
        try:
            client.delete_collection(collection_name=name)
            logger.info("Deleted old collection '%s'", name)
        except Exception as exc:
            # Cleanup failure must not invalidate a successful re-index.
            logger.warning("Could not delete collection '%s': %s", name, exc)


def replace_index(
    client: QdrantClient,
    alias: str,
    chunks: list[Document],
    vectors: list[list[float]],
    dimension: int,
) -> bool:
    """Build, validate, and activate a new collection safely."""
    new_collection = _temporary_collection_name(alias)

    try:
        _create_collection(client, new_collection, dimension)
        _upsert_chunks(client, new_collection, chunks, vectors)
        _validate_collection(
            client,
            new_collection,
            expected_count=len(chunks),
            sample_vector=vectors[0],
        )
    except Exception as exc:
        # The old collection and alias were never touched.
        try:
            client.delete_collection(collection_name=new_collection)
        except Exception:
            logger.warning("Could not remove failed temp collection '%s'", new_collection)
        if isinstance(exc, IngestionError):
            raise
        raise IngestionError(f"Failed to build the new index: {exc}") from exc

    try:
        previous = switch_alias(client, alias, new_collection)
    except Exception as exc:
        try:
            client.delete_collection(collection_name=new_collection)
        except Exception:
            logger.warning("Could not remove temp collection '%s'", new_collection)
        raise IngestionError(
            "Failed to activate the new index. The previous index is still active."
        ) from exc

    cleanup_old_collections(client, alias, new_collection, previous)
    return previous is not None


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run_ingestion(
    documents_root: Path | None = None,
    client: QdrantClient | None = None,
    embedding_service: EmbeddingService | None = None,
    collection_alias: str | None = None,
) -> IngestionStats:
    """Execute the full ingestion pipeline and return run statistics."""
    root = (documents_root or settings.documents_dir).resolve()
    alias = collection_alias or settings.qdrant_collection

    discovered = discover_documents(root)
    logger.info("Discovered %d supported documents in %s", len(discovered), root)
    if not discovered:
        raise NoDocumentsFoundError()

    # --- 1. Load -----------------------------------------------------------
    loaded_documents: list[Document] = []
    processed = 0
    skipped = 0
    pdf_pages = 0

    for path in discovered:
        result = load_document(path, root)
        pdf_pages += result.pdf_pages
        if result.empty or not result.documents:
            skipped += 1
            logger.warning("Skipped document with no extractable text: %s", path.name)
            continue
        processed += 1
        loaded_documents.extend(result.documents)

    if not loaded_documents:
        raise NoDocumentsFoundError()

    # --- 2. Chunk ----------------------------------------------------------
    chunks = chunk_documents(loaded_documents, build_splitter())
    if not chunks:
        raise NoDocumentsFoundError()

    # --- 3. Embed ----------------------------------------------------------
    embedder = embedding_service or EmbeddingService()
    configure_model_cache()
    dimension = embedder.verify_dimension(settings.embedding_dimension)
    try:
        vectors = embedder.embed_documents([chunk.page_content for chunk in chunks])
    except Exception as exc:
        raise IngestionError(f"Embedding generation failed: {exc}") from exc
    if len(vectors) != len(chunks):
        raise IngestionError("Embedding count does not match chunk count.")
    logger.info("Generated %d embeddings (%d dimensions)", len(vectors), dimension)

    # --- 4. Upload, validate, swap ----------------------------------------
    qdrant = client or create_qdrant_client()
    try:
        qdrant.get_collections()
    except Exception as exc:
        raise QdrantConnectionError("Could not connect to Qdrant.") from exc

    replaced = replace_index(qdrant, alias, chunks, vectors, dimension)

    return IngestionStats(
        discovered=len(discovered),
        processed=processed,
        skipped=skipped,
        pdf_pages=pdf_pages,
        chunks=len(chunks),
        embedding_model=settings.embedding_model.rsplit("/", 1)[-1],
        collection=alias,
        replaced_previous=replaced,
    )


def format_summary(stats: IngestionStats) -> str:
    """Render the human-readable ingestion report."""
    if stats.replaced_previous:
        activation = "Previous collection replaced successfully."
    else:
        activation = "New collection created and activated successfully."

    return "\n".join(
        [
            "ASKPOLICY DOCUMENT INGESTION",
            "",
            f"Documents discovered: {stats.discovered}",
            f"Documents processed: {stats.processed}",
            f"Documents skipped (no extractable text): {stats.skipped}",
            f"Total PDF pages: {stats.pdf_pages}",
            f"Total chunks generated: {stats.chunks}",
            f"Embedding model: {stats.embedding_model}",
            "Vector database: Qdrant Cloud",
            "",
            activation,
            f"Collection alias: {stats.collection}",
            "",
            "Ingestion completed.",
        ]
    )


def main() -> int:
    """CLI entry point. Returns a process exit code."""
    configure_logging()
    print("ASKPOLICY DOCUMENT INGESTION\n")
    try:
        stats = run_ingestion()
    except NoDocumentsFoundError as exc:
        logger.warning("%s", exc.public_message)
        print(f"No ingestion performed: {exc.public_message}")
        return 1
    except IngestionError as exc:
        logger.error("Ingestion failed: %s", exc)
        print(f"\nIngestion failed: {exc.public_message}")
        return 1
    except Exception:
        logger.exception("Unexpected ingestion failure")
        print(
            "\nIngestion failed unexpectedly. The previous index was left unchanged. "
            "See the log above for details."
        )
        return 1

    print("\n" + format_summary(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
