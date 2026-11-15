"""Metadata helpers for the ingestion pipeline.

Chunk metadata is stored inside the Qdrant payload and is later used for
source filtering, document listing, and citations in the UI. Values are always
derived from the real file path; page numbers are only added when the source
format actually provides them (PDF).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


def relative_source(path: Path, documents_root: Path) -> str:
    """Return the document path relative to ``documents/`` using ``/``.

    Relative paths keep nested folders meaningful (for example
    ``hr/leave_policy.pdf``) while remaining stable across operating systems.
    """
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
    """Create metadata for one chunk.

    ``page_number`` is omitted for DOCX/TXT files because those formats do not
    expose physical pages. Writing ``None`` into the payload would still be an
    invented page, so the key is simply not present.
    """
    metadata = dict(base)
    if page_number is not None:
        metadata["page_number"] = page_number
    metadata["chunk_index"] = chunk_index
    return metadata
