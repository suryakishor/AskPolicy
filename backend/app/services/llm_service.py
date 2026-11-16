"""Groq chat-completion integration for answer generation."""

from __future__ import annotations

import threading

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.core.config import settings
from app.core.errors import ConfigurationError, LLMError
from app.core.logging import get_logger

logger = get_logger(__name__)


class LLMService:
    """Lazy wrapper around ``ChatGroq``; the client is created on first use."""

    def __init__(self) -> None:
        self._llm: ChatGroq | None = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return settings.groq_model

    def _get_llm(self) -> ChatGroq:
        if self._llm is None:
            with self._lock:
                if self._llm is None:
                    settings.require_groq()
                    self._llm = ChatGroq(
                        api_key=settings.groq_api_key,
                        model=settings.groq_model,
                        temperature=settings.groq_temperature,
                        max_tokens=settings.groq_max_tokens,
                        timeout=settings.groq_timeout_seconds,
                        max_retries=2,
                    )
        return self._llm

    def generate_answer(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Generate one grounded answer from the system and user prompts."""
        try:
            response = self._get_llm().invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
        except ConfigurationError:
            raise
        except Exception as exc:
            # Log the provider error but never forward it to the browser.
            logger.exception("Groq completion failed")
            raise LLMError() from exc

        content = response.content
        if isinstance(content, list):
            # Flatten structured content blocks into plain text.
            parts = [str(block.get("text", "")) for block in content if isinstance(block, dict)]
            content = "".join(parts)
        answer = str(content or "").strip()
        if not answer:
            logger.error("Groq returned an empty answer")
            raise LLMError("The answer service returned an empty response.")
        return answer


def get_llm_service() -> LLMService:
    """Process-wide LLM service used by API dependencies."""
    return _llm_service


_llm_service = LLMService()
