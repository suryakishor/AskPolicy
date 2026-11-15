"""Endpoint that lists documents currently stored in Qdrant."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.schemas.documents import DocumentInfo, DocumentListResponse
from app.services.qdrant_service import QdrantService, get_qdrant_service

router = APIRouter(tags=["documents"])


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(
    qdrant: Annotated[QdrantService, Depends(get_qdrant_service)],
) -> DocumentListResponse:
    """Return the real indexed documents and their chunk counts.

    There is no separate metadata database: the payloads in Qdrant are read and
    aggregated on each request. If ingestion has not run, the route raises a
    ``CollectionNotReadyError`` which the app maps to HTTP 503.
    """
    summary = qdrant.get_index_summary()
    documents = [
        DocumentInfo(
            file_name=doc.file_name,
            file_type=doc.file_type,
            source=doc.source,
            chunk_count=doc.chunk_count,
        )
        for doc in summary.documents
    ]
    return DocumentListResponse(
        documents=documents,
        total_documents=len(documents),
        total_chunks=summary.total_chunks,
        collection=summary.collection_name,
    )
