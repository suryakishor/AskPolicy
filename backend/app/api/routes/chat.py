"""Question-answering endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ResetRequest,
    ResetResponse,
)
from app.services.rag_service import RagService, get_rag_service

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    rag: Annotated[RagService, Depends(get_rag_service)],
) -> ChatResponse:
    """Answer a question using retrieved policy documents."""
    result = rag.chat(
        question=request.question,
        thread_id=request.thread_id,
        source=request.source,
        history=request.history,
    )
    return ChatResponse(
        answer=result.answer,
        sources=result.sources,
        thread_id=result.thread_id,
    )


@router.post("/chat/reset", response_model=ResetResponse)
def reset_chat(
    request: ResetRequest,
    rag: Annotated[RagService, Depends(get_rag_service)],
) -> ResetResponse:
    """Clear the in-memory conversation state for a thread."""
    cleared = rag.reset(request.thread_id)
    return ResetResponse(cleared=cleared, thread_id=request.thread_id)
