"""Health and readiness endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app import __version__
from app.core.config import settings
from app.schemas.health import HealthResponse
from app.services.qdrant_service import QdrantService, get_qdrant_service

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(
    qdrant: Annotated[QdrantService, Depends(get_qdrant_service)],
) -> HealthResponse:
    """Report API status plus Qdrant connectivity.

    The endpoint is intentionally cheap: it does not call Groq or load the
    embedding model. ``degraded`` means the API is running but at least one
    dependency is not ready (missing credentials or no indexed collection).
    """
    qdrant_status = qdrant.health()
    ready = (
        bool(qdrant_status.get("connected"))
        and bool(qdrant_status.get("collection_ready"))
        and settings.groq_is_configured
    )
    return HealthResponse(
        status="ok" if ready else "degraded",
        app_env=settings.app_env,
        version=__version__,
        embedding_model=settings.embedding_model,
        llm_model=settings.groq_model,
        qdrant=qdrant_status,
        documents_indexed=bool(qdrant_status.get("collection_ready")),
    )
