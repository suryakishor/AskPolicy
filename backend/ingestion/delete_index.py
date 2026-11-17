"""AskPolicy vector index deletion CLI.

Run from the ``backend`` directory:

    uv run python -m ingestion.delete_index

Deletes the configured collection alias, the physical collection it points to,
and any leftover temporary ingestion collections. This is destructive: the
index is removed from Qdrant and must be rebuilt with ingestion afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.config import TEMP_COLLECTION_INFIX, settings
from app.core.errors import AskPolicyError
from app.core.logging import configure_logging, get_logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

logger = get_logger(__name__)


@dataclass
class DeletionStats:
    """Facts collected during a delete run."""

    collection: str
    deleted_alias: bool
    deleted_collection: str | None
    deleted_temp_collections: list[str] = field(default_factory=list)


def create_qdrant_client() -> QdrantClient:
    """Create a client from validated environment settings."""
    settings.require_qdrant()
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=settings.qdrant_timeout_seconds,
    )


def _alias_targets(client: QdrantClient) -> dict[str, str]:
    """Return a mapping of alias name -> physical collection name."""
    return {alias.alias_name: alias.collection_name for alias in client.get_aliases().aliases}


def delete_index(client: QdrantClient, alias: str | None = None) -> DeletionStats:
    """Delete the alias, its physical collection, and leftover temp collections."""
    target_alias = alias or settings.qdrant_collection
    physical = _alias_targets(client).get(target_alias)

    deleted_alias = False
    if physical:
        # Drop the alias first so the collection is not left dangling.
        client.update_collection_aliases(
            change_aliases_operations=[
                qdrant_models.DeleteAliasOperation(
                    delete_alias=qdrant_models.DeleteAlias(alias_name=target_alias)
                )
            ]
        )
        deleted_alias = True
        client.delete_collection(collection_name=physical)
        logger.info("Deleted alias '%s' and collection '%s'", target_alias, physical)
    else:
        logger.info("Alias '%s' was not present; nothing to delete", target_alias)

    # Remove temporary collections left behind by failed ingestion runs.
    deleted_temp: list[str] = []
    prefix = f"{target_alias}{TEMP_COLLECTION_INFIX}"
    for item in client.get_collections().collections:
        if item.name.startswith(prefix):
            client.delete_collection(collection_name=item.name)
            deleted_temp.append(item.name)
            logger.info("Deleted leftover temp collection '%s'", item.name)

    return DeletionStats(
        collection=target_alias,
        deleted_alias=deleted_alias,
        deleted_collection=physical,
        deleted_temp_collections=deleted_temp,
    )


def format_summary(stats: DeletionStats) -> str:
    """Render the human-readable deletion report."""
    if stats.deleted_collection:
        status = f"Deleted alias and collection: {stats.deleted_collection}"
    else:
        status = "No active alias was found; nothing to delete."

    lines = [
        "ASKPOLICY VECTOR INDEX DELETION",
        "",
        f"Collection alias: {stats.collection}",
        "Vector database: Qdrant Cloud",
        "",
        status,
    ]
    if stats.deleted_temp_collections:
        lines.append(
            f"Deleted leftover temporary collections: {len(stats.deleted_temp_collections)}"
        )
    lines.extend(["", "Deletion completed."])
    return "\n".join(lines)


def main() -> int:
    """CLI entry point. Returns a process exit code."""
    configure_logging()
    print("ASKPOLICY VECTOR INDEX DELETION\n")

    try:
        client = create_qdrant_client()
        client.get_collections()
    except AskPolicyError as exc:
        logger.error("Deletion failed: %s", exc)
        print(f"\nDeletion failed: {exc.public_message}")
        return 1
    except Exception:
        logger.exception("Could not connect to Qdrant")
        print("\nDeletion failed: could not connect to Qdrant. See the log above.")
        return 1

    try:
        stats = delete_index(client)
    except Exception:
        logger.exception("Unexpected deletion failure")
        print("\nDeletion failed unexpectedly. See the log above for details.")
        return 1

    print("\n" + format_summary(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
