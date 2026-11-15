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
    """Answer a question using retrieved policy documents.

    Request validation (non-empty question, length limits) happens in Pydantic.
    Expected service failures propagate to the application-level exception
    handlers, which return safe messages and appropriate status codes.
    """
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
    """Clear the in-memory conversation state for a thread.

    The frontend calls this when the user starts a new chat or clears the
    current one. If the thread does not exist the operation is still a no-op
    success, which keeps the client logic simple.
    """
    cleared = rag.reset(request.thread_id)
    return ResetResponse(cleared=cleared, thread_id=request.thread_id)
