"""Pydantic request/response schemas for the chat API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.schemas.documents import RetrievedChunk


class ChatMessage(BaseModel):
    """A client-supplied historical message, used only to seed a new thread."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)


class ChatRequest(BaseModel):
    """Body for ``POST /api/chat``."""

    question: str = Field(min_length=1, max_length=4_000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)
    source: str | None = Field(
        default=None,
        max_length=500,
        description="Optional relative document path to restrict retrieval to.",
    )
    thread_id: str | None = Field(
        default=None,
        max_length=128,
        description="Conversation/thread identifier. Omit to start a new thread.",
    )

    @field_validator("question")
    @classmethod
    def _question_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Question must not be empty")
        return cleaned

    @field_validator("source", "thread_id")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class SourceReference(RetrievedChunk):
    """A retrieved chunk cited in an answer."""


class ChatResponse(BaseModel):
    """Body returned by ``POST /api/chat``."""

    answer: str
    sources: list[SourceReference]
    thread_id: str


class ResetRequest(BaseModel):
    """Body for ``POST /api/chat/reset``."""

    thread_id: str = Field(min_length=1, max_length=128)


class ResetResponse(BaseModel):
    """Confirmation that a conversation's in-memory state was cleared."""

    cleared: bool
    thread_id: str
