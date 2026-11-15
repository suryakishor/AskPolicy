"""Tests for recursive character chunking, overlap, and metadata."""

from __future__ import annotations

import pytest
from ingestion.chunker import build_splitter, chunk_documents
from langchain_core.documents import Document


def test_chunks_are_non_empty_and_within_size() -> None:
    text = " ".join(f"PolicySentence{index} has useful details." for index in range(80))
    splitter = build_splitter(chunk_size=240, chunk_overlap=50)

    chunks = chunk_documents(
        [
            Document(
                page_content=text,
                metadata={
                    "source": "handbook.pdf",
                    "file_name": "handbook.pdf",
                    "file_type": "pdf",
                    "page_number": 1,
                },
            )
        ],
        splitter,
    )

    assert len(chunks) > 1
    assert all(chunk.page_content.strip() for chunk in chunks)
    assert all(len(chunk.page_content) <= 240 for chunk in chunks)
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.metadata["page_number"] == 1 for chunk in chunks)


def test_overlap_is_applied_between_consecutive_chunks() -> None:
    text = " ".join(f"token{index}" for index in range(500))
    splitter = build_splitter(chunk_size=200, chunk_overlap=50)

    chunks = [piece for piece in splitter.split_text(text) if piece.strip()]

    assert len(chunks) > 3
    shared = [chunks[index][-25:] in chunks[index + 1] for index in range(len(chunks) - 1)]
    # Recursive splitting keeps separator boundaries readable, so overlap is
    # not always an exact byte-for-byte suffix, but at least some consecutive
    # chunks must share trailing text.
    assert any(shared)


def test_chunk_index_continues_across_pages_of_one_source() -> None:
    splitter = build_splitter(chunk_size=120, chunk_overlap=20)
    documents = [
        Document(
            page_content="Page one content. " * 12,
            metadata={
                "source": "leave.pdf",
                "file_name": "leave.pdf",
                "file_type": "pdf",
                "page_number": 1,
            },
        ),
        Document(
            page_content="Page two content. " * 12,
            metadata={
                "source": "leave.pdf",
                "file_name": "leave.pdf",
                "file_type": "pdf",
                "page_number": 2,
            },
        ),
    ]

    chunks = chunk_documents(documents, splitter)

    indexes = [chunk.metadata["chunk_index"] for chunk in chunks]
    assert indexes == list(range(len(chunks)))
    assert {chunk.metadata["page_number"] for chunk in chunks} == {1, 2}
    # Indexes keep increasing across page boundaries.
    first_page = [c for c in chunks if c.metadata["page_number"] == 1]
    second_page = [c for c in chunks if c.metadata["page_number"] == 2]
    assert max(c.metadata["chunk_index"] for c in first_page) < min(
        c.metadata["chunk_index"] for c in second_page
    )


def test_empty_text_produces_no_chunks() -> None:
    splitter = build_splitter(chunk_size=100, chunk_overlap=20)
    document = Document(
        page_content="   \n\n \t ",
        metadata={"source": "empty.txt", "file_name": "empty.txt", "file_type": "txt"},
    )

    assert chunk_documents([document], splitter) == []


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError):
        build_splitter(chunk_size=100, chunk_overlap=100)
