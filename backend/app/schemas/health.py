"""Response schema for the health endpoint."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class DependencyStatus(BaseModel):
    """Status of a backing service."""

    configured: bool
    connected: bool = False
    detail: str | None = None


class HealthResponse(BaseModel):
    """Public health information (never contains credentials)."""

    status: Literal["ok", "degraded"]
    app_env: str
    embedding_model: str
    llm_model: str
    qdrant: dict
    documents_indexed: bool
