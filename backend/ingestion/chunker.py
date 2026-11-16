"""Recursive character chunking with metadata and overlap.

Chunks are measured in characters (model-independent and cheap) and overlap so
text that straddles a boundary stays visible in at least one chunk.
"""

from __future__ import annotations

from app.core.config import settings
from app.core.logging import get_logger
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.metadata import chunk_metadata

logger = get_logger(__name__)

# Prefer paragraph breaks, then lines, then spaces, then inside words.
DEFAULT_SEPARATORS = ["\n\n", "\n", " ", ""]


def build_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """Create a stateless splitter using the configured size and overlap."""
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    if overlap >= size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=DEFAULT_SEPARATORS,
        keep_separator=True,
    )


def chunk_documents(
    documents: list[Document],
    splitter: RecursiveCharacterTextSplitter | None = None,
) -> list[Document]:
    """Split loaded documents into non-empty chunks with per-file indexes."""
    active_splitter = splitter or build_splitter()
    counters: dict[str, int] = {}
    chunks: list[Document] = []

    for document in documents:
        source = str(document.metadata.get("source", ""))
        pieces = active_splitter.split_text(document.page_content)
        for piece in pieces:
            text = piece.strip()
            # Skip whitespace-only pieces; they carry no meaning.
            if not text:
                continue

            next_index = counters.get(source, 0)
            metadata = dict(document.metadata)
            page_number = metadata.pop("page_number", None)
            chunks.append(
                Document(
                    page_content=text,
                    metadata=chunk_metadata(metadata, next_index, page_number),
                )
            )
            counters[source] = next_index + 1

    logger.info("Generated %d chunks from %d loaded units", len(chunks), len(documents))
    return chunks
