"""RAG pipeline and LangGraph conversation-memory tests with faked services."""

from __future__ import annotations

from app.core.prompts import NOT_FOUND_MESSAGE
from app.schemas.chat import ChatMessage
from app.schemas.documents import RetrievedChunk
from app.services.rag_service import RagService

THREAD = "test-thread"


class FakeRetrieval:
    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self.chunks = chunks or []
        self.calls: list[tuple[str, str | None]] = []

    def retrieve(
        self,
        question: str,
        source: str | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append((question, source))
        return self.chunks


class FakeLLM:
    def __init__(self, answer: str = "Employees receive 20 days of leave [1].") -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    def generate_answer(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.answer


def leave_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        file_name="leave_policy.pdf",
        source="hr/leave_policy.pdf",
        file_type="pdf",
        page_number=2,
        chunk_index=4,
        content="Employees receive 20 days of annual leave each year.",
        score=0.91,
    )


def thread_messages(rag: RagService, thread_id: str = THREAD) -> list:
    snapshot = rag._graph.get_state({"configurable": {"thread_id": thread_id}})
    return list((snapshot.values or {}).get("messages", []))


def test_answer_uses_retrieved_context_and_maps_sources() -> None:
    retrieval = FakeRetrieval([leave_chunk()])
    llm = FakeLLM()
    rag = RagService(retrieval, llm)

    result = rag.chat("What is the leave policy?", thread_id=THREAD)

    assert result.answer.startswith("Employees receive 20 days")
    assert result.thread_id == THREAD
    assert len(result.sources) == 1
    assert result.sources[0].file_name == "leave_policy.pdf"
    assert result.sources[0].page_number == 2
    assert result.sources[0].chunk_index == 4

    # The prompt contains the retrieved excerpt and the anti-hallucination rules.
    system_prompt, user_prompt = llm.calls[0]
    assert "Never invent or guess policies" in system_prompt
    assert "Employees receive 20 days" in user_prompt
    assert "What is the leave policy?" in user_prompt


def test_source_filter_is_forwarded_to_retrieval() -> None:
    retrieval = FakeRetrieval([leave_chunk()])
    rag = RagService(retrieval, FakeLLM())

    rag.chat("Leave policy?", thread_id=THREAD, source="hr/leave_policy.pdf")

    assert retrieval.calls == [("Leave policy?", "hr/leave_policy.pdf")]


def test_no_retrieved_chunks_returns_required_not_found_message() -> None:
    retrieval = FakeRetrieval([])
    llm = FakeLLM()
    rag = RagService(retrieval, llm)

    result = rag.chat("What is the sabbatical policy?", thread_id=THREAD)

    assert result.answer == NOT_FOUND_MESSAGE
    assert result.sources == []
    # The LLM is skipped entirely when there is no evidence to ground an answer.
    assert llm.calls == []
    assert len(thread_messages(rag)) == 2


def test_model_not_found_answer_does_not_return_misleading_sources() -> None:
    retrieval = FakeRetrieval([leave_chunk()])
    llm = FakeLLM(answer=NOT_FOUND_MESSAGE)
    rag = RagService(retrieval, llm)

    result = rag.chat("What is the sabbatical policy?", thread_id=THREAD)

    assert result.answer == NOT_FOUND_MESSAGE
    assert result.sources == []


def test_only_cited_retrieved_chunks_are_returned_as_sources() -> None:
    travel_chunk = RetrievedChunk(
        file_name="travel_policy.txt",
        source="hr/travel_policy.txt",
        file_type="txt",
        page_number=None,
        chunk_index=0,
        content="Hotel reimbursement is capped at 180 USD per night.",
        score=0.72,
    )
    retrieval = FakeRetrieval([leave_chunk(), travel_chunk])
    llm = FakeLLM(answer="The hotel cap is 180 USD per night [2].")
    rag = RagService(retrieval, llm)

    result = rag.chat("What is the hotel reimbursement cap?", thread_id=THREAD)

    assert [source.file_name for source in result.sources] == ["travel_policy.txt"]
    assert result.sources[0].content == travel_chunk.content


def test_answer_without_citations_returns_no_sources() -> None:
    retrieval = FakeRetrieval([leave_chunk()])
    llm = FakeLLM(answer="Employees receive 20 days of annual leave.")
    rag = RagService(retrieval, llm)

    result = rag.chat("What is the leave policy?", thread_id=THREAD)

    assert result.sources == []


def test_non_standard_citation_markers_are_normalized() -> None:
    travel_chunk = RetrievedChunk(
        file_name="travel_policy.txt",
        source="hr/travel_policy.txt",
        file_type="txt",
        page_number=None,
        chunk_index=0,
        content="Hotel reimbursement is capped at 180 USD per night.",
        score=0.72,
    )
    retrieval = FakeRetrieval([leave_chunk(), travel_chunk])
    llm = FakeLLM(answer="The hotel cap is 180 USD per night【2†L5-L6】.")
    rag = RagService(retrieval, llm)

    result = rag.chat("What is the hotel reimbursement cap?", thread_id=THREAD)

    # The artifact is rewritten to a standard marker and still resolves a source.
    assert "【" not in result.answer
    assert "[2]" in result.answer
    assert [source.file_name for source in result.sources] == ["travel_policy.txt"]


def test_unrecognized_citation_markers_are_stripped() -> None:
    retrieval = FakeRetrieval([leave_chunk()])
    llm = FakeLLM(answer="Employees receive 20 days of annual leave【notes】.")
    rag = RagService(retrieval, llm)

    result = rag.chat("What is the leave policy?", thread_id=THREAD)

    assert "【" not in result.answer
    assert result.sources == []


def test_greeting_gets_conversational_answer_without_sources() -> None:
    retrieval = FakeRetrieval([leave_chunk()])
    llm = FakeLLM(answer="Hello! Ask me anything about our company policies.")
    rag = RagService(retrieval, llm)

    result = rag.chat("hello", thread_id=THREAD)

    assert result.answer.startswith("Hello!")
    assert result.sources == []
    # The system prompt instructs the model how to handle greetings and small talk.
    system_prompt, _ = llm.calls[0]
    assert "greeting" in system_prompt.lower()


def test_thread_memory_keeps_only_last_five_exchanges() -> None:
    retrieval = FakeRetrieval([leave_chunk()])
    llm = FakeLLM()
    rag = RagService(retrieval, llm)

    for number in range(1, 8):
        rag.chat(f"Question {number}", thread_id=THREAD)

    messages = thread_messages(rag)
    assert len(messages) == 10  # five questions + five answers

    questions = [message.content for message in messages if message.type == "human"]
    assert questions == [f"Question {number}" for number in range(3, 8)]

    # The latest prompt sees recent follow-up context but not evicted turns.
    latest_prompt = llm.calls[-1][1]
    assert "Question 6" in latest_prompt
    assert "Question 1" not in latest_prompt


def test_threads_are_isolated() -> None:
    rag = RagService(FakeRetrieval([leave_chunk()]), FakeLLM())

    rag.chat("First thread question", thread_id="thread-a")
    rag.chat("Second thread question", thread_id="thread-b")

    assert thread_messages(rag, "thread-a")[0].content == "First thread question"
    assert thread_messages(rag, "thread-b")[0].content == "Second thread question"


def test_reset_clears_thread_state() -> None:
    rag = RagService(FakeRetrieval([leave_chunk()]), FakeLLM())
    rag.chat("What is the leave policy?", thread_id=THREAD)
    assert len(thread_messages(rag)) == 2

    assert rag.reset(THREAD) is True
    assert thread_messages(rag) == []


def test_new_thread_can_be_seeded_from_client_history() -> None:
    retrieval = FakeRetrieval([leave_chunk()])
    llm = FakeLLM()
    rag = RagService(retrieval, llm)

    rag.chat(
        "And how many days carry over?",
        thread_id="seeded-thread",
        history=[
            ChatMessage(role="user", content="What is the leave policy?"),
            ChatMessage(role="assistant", content="You receive 20 days [1]."),
        ],
    )

    prompt = llm.calls[0][1]
    assert "What is the leave policy?" in prompt
    assert "You receive 20 days [1]." in prompt
    assert "not verified policy evidence" in prompt


def test_missing_thread_id_generates_one() -> None:
    rag = RagService(FakeRetrieval([]), FakeLLM())

    result = rag.chat("Any question?")

    assert result.thread_id
    assert len(result.thread_id) >= 16
