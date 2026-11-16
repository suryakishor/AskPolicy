"""Pydantic schemas for indexed documents and retrieval results."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """One chunk returned by similarity search, with its stored text."""

    file_name: str
    source: str = ""
    file_type: str = ""
    page_number: int | None = None
    chunk_index: int = 0
    content: str
    score: float = Field(description="Cosine similarity score (higher is closer).")


class DocumentInfo(BaseModel):
    """A source file currently present in the vector index."""

    file_name: str
    file_type: str
    source: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    """Response body for ``GET /api/documents``."""

    documents: list[DocumentInfo]
    total_documents: int
    total_chunks: int
    collection: str
