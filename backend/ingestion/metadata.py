"""Metadata helpers for the ingestion pipeline.

Chunk metadata is stored in the Qdrant payload and drives source filtering,
document listing, and citations. Page numbers are only added for PDFs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


def relative_source(path: Path, documents_root: Path) -> str:
    """Return the path relative to ``documents/`` using forward slashes."""
    return path.resolve().relative_to(documents_root.resolve()).as_posix()


def base_metadata(path: Path, documents_root: Path) -> dict[str, Any]:
    """Build the metadata shared by all chunks of a document."""
    return {
        "source": relative_source(path, documents_root),
        "file_name": path.name,
        "file_type": path.suffix.lower().lstrip("."),
    }


def chunk_metadata(
    base: dict[str, Any],
    chunk_index: int,
    page_number: int | None = None,
) -> dict[str, Any]:
    """Create metadata for one chunk; ``page_number`` is only set for PDFs."""
    metadata = dict(base)
    if page_number is not None:
        metadata["page_number"] = page_number
    metadata["chunk_index"] = chunk_index
    return metadata
