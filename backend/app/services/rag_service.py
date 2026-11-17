"""RAG orchestration with short-term conversation memory.

It retrieves chunks from Qdrant, builds a grounded prompt, calls Groq, and stores
each exchange in a per-thread LangGraph memory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from app.core.config import settings
from app.core.errors import CollectionNotReadyError
from app.core.logging import get_logger
from app.core.prompts import (
    NOT_FOUND_MESSAGE,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from app.schemas.chat import ChatMessage, SourceReference
from app.schemas.documents import RetrievedChunk
from app.services.llm_service import LLMService, get_llm_service
from app.services.retrieval_service import (
    RetrievalService,
    get_retrieval_service,
)

logger = get_logger(__name__)

#: Matches the bracketed excerpt numbers requested by the system prompt.
CITATION_PATTERN = re.compile(r"\[(\d+)\]")

#: Non-standard citation artifacts some models emit, e.g. "【3†L5-L6】".
CITATION_ARTIFACT_PATTERN = re.compile(r"【\s*(\d+)[^】]*】")
#: Any remaining non-standard marker that cannot be mapped to a citation.
CITATION_LEFTOVER_PATTERN = re.compile(r"【[^】]*】")


@dataclass
class ChatResult:
    """Internal result of one chat turn."""

    answer: str
    sources: list[SourceReference] = field(default_factory=list)
    thread_id: str = ""


class ChatState(MessagesState):
    """Conversation messages plus the latest turn's sources."""

    sources: list[dict]


class RagService:
    """Coordinates retrieval, prompting, generation, and thread memory."""

    def __init__(
        self,
        retrieval_service: RetrievalService | None = None,
        llm_service: LLMService | None = None,
    ) -> None:
        self.retrieval = retrieval_service or get_retrieval_service()
        self.llm = llm_service or get_llm_service()
        self.max_exchanges = settings.conversation_max_exchanges
        self._graph = self._build_graph()

    # ------------------------------------------------------------------ graph
    def _build_graph(self):
        """Compile the one-node memory graph backed by an in-memory checkpointer."""
        builder = StateGraph(ChatState)
        builder.add_node("answer", self._answer_node)
        builder.add_edge(START, "answer")
        builder.add_edge("answer", END)
        return builder.compile(checkpointer=InMemorySaver())

    def _answer_node(self, state: ChatState, config: RunnableConfig) -> dict:
        """Retrieve evidence, generate an answer, and update thread state."""
        messages: list[BaseMessage] = list(state.get("messages", []))
        if not messages:
            # Defensive: the graph is always invoked with a human message.
            return {"messages": [], "sources": []}

        question = self._message_text(messages[-1]).strip()
        history = self._history_pairs(messages[:-1])
        source_filter = (config.get("configurable") or {}).get("source")

        chunks = self.retrieval.retrieve(question, source=source_filter)

        # No retrieved evidence means there is nothing to ground an answer in.
        # Skipping the LLM here removes any opportunity to hallucinate.
        if not chunks:
            answer = NOT_FOUND_MESSAGE
            sources: list[SourceReference] = []
        else:
            user_prompt = build_user_prompt(question, chunks, history)
            answer = self._normalize_citations(self.llm.generate_answer(SYSTEM_PROMPT, user_prompt))
            if answer.strip() == NOT_FOUND_MESSAGE:
                # Unsupported claims should not be accompanied by unrelated
                # citations that could look like evidence.
                sources = []
            else:
                sources = self._select_sources(answer, chunks)

        # Keep the checkpoint small: retain at most N full exchanges
        # (question + answer), where N defaults to five.
        max_messages = self.max_exchanges * 2
        new_total = len(messages) + 1
        removals = []
        if new_total > max_messages:
            remove_count = new_total - max_messages
            removals = [
                RemoveMessage(id=message.id) for message in messages[:remove_count] if message.id
            ]

        return {
            "messages": [*removals, AIMessage(content=answer)],
            "sources": [source.model_dump() for source in sources],
        }

    @staticmethod
    def _message_text(message: BaseMessage) -> str:
        """Normalize LangChain message content to plain text."""
        content = message.content
        if isinstance(content, list):
            return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        return str(content or "")

    def _history_pairs(self, messages: list[BaseMessage]) -> list[tuple[str, str]]:
        """Convert recent messages to (role, text) pairs for prompt context."""
        recent = messages[-self.max_exchanges * 2 :]
        pairs: list[tuple[str, str]] = []
        for message in recent:
            if isinstance(message, HumanMessage):
                pairs.append(("user", self._message_text(message)))
            elif isinstance(message, AIMessage):
                pairs.append(("assistant", self._message_text(message)))
        return pairs

    @staticmethod
    def _normalize_citations(answer: str) -> str:
        """Rewrite non-standard citation markers to the bracketed [n] format.

        Models sometimes emit artifacts such as "【3†L5-L6】"; keep the excerpt
        number and drop the rest, then remove any markers that cannot be mapped.
        """
        answer = CITATION_ARTIFACT_PATTERN.sub(lambda match: f"[{match.group(1)}]", answer)
        return CITATION_LEFTOVER_PATTERN.sub("", answer)

    @staticmethod
    def _to_source(chunk: RetrievedChunk) -> SourceReference:
        """Map a retrieval result to the public source schema."""
        return SourceReference(**chunk.model_dump())

    @classmethod
    def _select_sources(
        cls,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> list[SourceReference]:
        """Return only the excerpts the answer actually cited.

        A greeting or other non-policy answer contains no citations, so it
        returns no sources.
        """
        cited: list[int] = []
        for match in CITATION_PATTERN.finditer(answer):
            index = int(match.group(1)) - 1
            if 0 <= index < len(chunks) and index not in cited:
                cited.append(index)

        if not cited:
            logger.debug("Answer contained no valid citations; returning no sources")
            return []

        return [cls._to_source(chunks[index]) for index in sorted(cited)]

    # --------------------------------------------------------------- public API
    def chat(
        self,
        question: str,
        thread_id: str | None = None,
        source: str | None = None,
        history: list[ChatMessage] | None = None,
    ) -> ChatResult:
        """Run one question through the RAG pipeline.

        ``history`` only seeds a brand-new thread; server-side memory is
        authoritative afterwards.
        """
        resolved_thread_id = (thread_id or "").strip() or uuid4().hex
        config: RunnableConfig = {
            "configurable": {
                "thread_id": resolved_thread_id,
                "source": source,
            }
        }

        try:
            self._seed_history(resolved_thread_id, config, history or [])
            final_state = self._graph.invoke(
                {"messages": [HumanMessage(content=question.strip())]},
                config=config,
            )
        except CollectionNotReadyError:
            raise
        except Exception:
            logger.exception("RAG request failed (thread_id=%s)", resolved_thread_id)
            raise

        # The answer node appends an AIMessage; the last message in the
        # checkpoint is therefore the answer for this turn.
        conversation = final_state.get("messages") or []
        answer = self._message_text(conversation[-1]).strip() if conversation else ""

        sources = [SourceReference(**item) for item in (final_state.get("sources") or [])]
        return ChatResult(
            answer=answer,
            sources=sources,
            thread_id=resolved_thread_id,
        )

    def reset(self, thread_id: str) -> bool:
        """Delete all in-memory state for a conversation thread."""
        cleaned = thread_id.strip()
        if not cleaned:
            return False
        checkpointer = self._graph.checkpointer
        checkpointer.delete_thread(cleaned)
        logger.info("Cleared conversation memory (thread_id=%s)", cleaned)
        return True

    def _seed_history(
        self,
        thread_id: str,
        config: RunnableConfig,
        history: list[ChatMessage],
    ) -> None:
        """Seed a brand-new thread from optional client history, once."""
        if not history:
            return

        snapshot = self._graph.get_state(config)
        existing = (snapshot.values or {}).get("messages") if snapshot else None
        if existing:
            return

        recent = history[-self.max_exchanges * 2 :]
        messages: list[BaseMessage] = [
            HumanMessage(content=item.content)
            if item.role == "user"
            else AIMessage(content=item.content)
            for item in recent
        ]
        self._graph.update_state(config, {"messages": messages})
        logger.info("Seeded %d history messages (thread_id=%s)", len(messages), thread_id)


def get_rag_service() -> RagService:
    """FastAPI dependency: process-wide RAG service."""
    return _rag_service


_rag_service = RagService()
