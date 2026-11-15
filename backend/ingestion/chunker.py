"""Recursive character chunking with metadata and overlap.

Why characters instead of tokens?
    Character counts are model-independent and cheap, while token counts require
    the embedding model's tokenizer and a different splitter API. The chosen
    model (``all-MiniLM-L6-v2``) accepts at most 256 word pieces. A 1000
    character chunk of normal English is roughly 200-250 tokens, so it fits in
    practice; unusually dense text can still be truncated by the model, which is
    a known limitation documented in the architecture notes.

Why overlap?
    Consecutive chunks share 200 characters so a sentence or definition that
    straddles a boundary is still fully visible in at least one chunk. Too much
    overlap inflates storage and duplicate retrieval, so it stays well below the
    chunk size.
"""

from __future__ import annotations

from app.core.config import settings
from app.core.logging import get_logger
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.metadata import chunk_metadata

logger = get_logger(__name__)

# Default hierarchy: prefer paragraph breaks, then lines, then spaces, and only
# split inside a word as a last resort. This preserves readable boundaries.
DEFAULT_SEPARATORS = ["\n\n", "\n", " ", ""]


def build_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """Create the configured splitter.

    The splitter is stateless across calls; it can be reused for every page or
    file in an ingestion run.
    """
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
    """Split page/file documents into chunks with a per-file chunk index.

    Args:
        documents: Output of the loaders. PDF pages arrive in page order.
        splitter: Optional pre-built splitter (useful in tests).

    Returns:
        Non-empty chunks. ``chunk_index`` increases across the whole source file,
        while ``page_number`` is preserved for PDFs.
    """
    active_splitter = splitter or build_splitter()
    counters: dict[str, int] = {}
    chunks: list[Document] = []

    for document in documents:
        source = str(document.metadata.get("source", ""))
        pieces = active_splitter.split_text(document.page_content)
        for piece in pieces:
            text = piece.strip()
            # Recursive splitting can produce whitespace-only pieces; they carry
            # no meaning and would waste an embedding call.
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
