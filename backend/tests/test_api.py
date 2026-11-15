"""API contract tests with external services replaced by fakes."""

from __future__ import annotations

import pytest
from app.core.config import settings
from app.core.errors import CollectionNotReadyError, LLMError
from app.main import app
from app.schemas.chat import SourceReference
from app.services.qdrant_service import (
    IndexedDocument,
    IndexSummary,
    get_qdrant_service,
)
from app.services.rag_service import ChatResult, get_rag_service
from fastapi.testclient import TestClient

client = TestClient(app)


class StubQdrant:
    def __init__(
        self,
        health: dict | None = None,
        summary: IndexSummary | None = None,
        error: Exception | None = None,
    ) -> None:
        self._health = health or {
            "configured": True,
            "connected": True,
            "collection_ready": True,
            "collection": "askpolicy_documents",
        }
        self._summary = summary
        self._error = error

    def health(self) -> dict:
        return self._health

    def get_index_summary(self) -> IndexSummary:
        if self._error:
            raise self._error
        assert self._summary is not None
        return self._summary


class StubRag:
    def __init__(
        self,
        answer: str = "You receive 20 days of leave [1].",
        sources: list[SourceReference] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.answer = answer
        self.sources = sources or []
        self.error = error
        self.calls: list[dict] = []
        self.reset_calls: list[str] = []

    def chat(self, question, thread_id=None, source=None, history=None) -> ChatResult:
        if self.error:
            raise self.error
        self.calls.append(
            {
                "question": question,
                "thread_id": thread_id,
                "source": source,
                "history": history,
            }
        )
        return ChatResult(
            answer=self.answer,
            sources=self.sources,
            thread_id=thread_id or "generated-thread",
        )

    def reset(self, thread_id: str) -> bool:
        self.reset_calls.append(thread_id)
        return True


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def use_qdrant(stub: StubQdrant) -> None:
    app.dependency_overrides[get_qdrant_service] = lambda: stub


def use_rag(stub: StubRag) -> None:
    app.dependency_overrides[get_rag_service] = lambda: stub


def source_reference() -> SourceReference:
    return SourceReference(
        file_name="leave_policy.pdf",
        source="hr/leave_policy.pdf",
        file_type="pdf",
        page_number=2,
        chunk_index=4,
        content="Employees receive 20 days of annual leave.",
        score=0.91,
    )


def test_health_reports_degraded_when_qdrant_not_ready() -> None:
    use_qdrant(
        StubQdrant(
            health={
                "configured": True,
                "connected": True,
                "collection_ready": False,
                "collection": "askpolicy_documents",
                "detail": "collection missing",
            }
        )
    )

    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["documents_indexed"] is False
    assert body["embedding_model"] == settings.embedding_model
    # Credentials are never included.
    assert "api_key" not in response.text.lower()


def test_health_ok_when_services_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "groq_api_key", "test-key")
    monkeypatch.setattr(settings, "groq_model", "test-model")
    use_qdrant(StubQdrant())

    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["llm_model"] == "test-model"


def test_documents_returns_indexed_metadata() -> None:
    summary = IndexSummary(
        documents=[
            IndexedDocument(
                source="hr/leave_policy.pdf",
                file_name="leave_policy.pdf",
                file_type="pdf",
                chunk_count=12,
            ),
            IndexedDocument(
                source="travel.txt",
                file_name="travel.txt",
                file_type="txt",
                chunk_count=3,
            ),
        ],
        total_chunks=15,
        collection_name="askpolicy_documents",
    )
    use_qdrant(StubQdrant(summary=summary))

    response = client.get("/api/documents")

    assert response.status_code == 200
    body = response.json()
    assert body["total_documents"] == 2
    assert body["total_chunks"] == 15
    assert body["documents"][0]["chunk_count"] == 12


def test_documents_returns_503_before_ingestion() -> None:
    use_qdrant(StubQdrant(error=CollectionNotReadyError()))

    response = client.get("/api/documents")

    assert response.status_code == 503
    assert "No policy documents are indexed" in response.json()["detail"]


def test_chat_returns_answer_sources_and_thread_id() -> None:
    stub = StubRag(answer="Twenty days [1].", sources=[source_reference()])
    use_rag(stub)

    response = client.post(
        "/api/chat",
        json={
            "question": "What is the leave policy?",
            "history": [{"role": "user", "content": "Hello"}],
            "source": "hr/leave_policy.pdf",
            "thread_id": "thread-123",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Twenty days [1]."
    assert body["thread_id"] == "thread-123"
    assert body["sources"][0]["file_name"] == "leave_policy.pdf"
    assert body["sources"][0]["page_number"] == 2
    assert stub.calls[0]["source"] == "hr/leave_policy.pdf"
    assert stub.calls[0]["history"][0].content == "Hello"


def test_chat_generates_thread_id_when_absent() -> None:
    stub = StubRag(answer="Answer", sources=[])
    use_rag(stub)

    response = client.post("/api/chat", json={"question": "Any policy?"})

    assert response.status_code == 200
    assert response.json()["thread_id"] == "generated-thread"


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_chat_rejects_empty_questions(question: str) -> None:
    use_rag(StubRag())

    response = client.post("/api/chat", json={"question": question})

    assert response.status_code == 422


def test_chat_rejects_missing_question() -> None:
    use_rag(StubRag())

    response = client.post("/api/chat", json={})

    assert response.status_code == 422


def test_missing_collection_maps_to_503() -> None:
    use_rag(StubRag(error=CollectionNotReadyError()))

    response = client.post("/api/chat", json={"question": "What is the leave policy?"})

    assert response.status_code == 503
    assert "No policy documents are indexed" in response.json()["detail"]


def test_llm_failure_maps_to_502_without_internal_details() -> None:
    use_rag(StubRag(error=LLMError("upstream secret detail")))

    response = client.post("/api/chat", json={"question": "What is the leave policy?"})

    assert response.status_code == 502
    assert response.json()["detail"] == "upstream secret detail"
    # The response is a controlled domain message, not a stack trace/provider body.


def test_reset_clears_thread() -> None:
    stub = StubRag()
    use_rag(stub)

    response = client.post("/api/chat/reset", json={"thread_id": "thread-123"})

    assert response.status_code == 200
    assert response.json() == {"cleared": True, "thread_id": "thread-123"}
    assert stub.reset_calls == ["thread-123"]


def test_reset_rejects_blank_thread_id() -> None:
    use_rag(StubRag())

    response = client.post("/api/chat/reset", json={"thread_id": ""})

    assert response.status_code == 422


def test_cors_allows_configured_frontend_origin() -> None:
    if not settings.frontend_origins:
        pytest.skip("No frontend origin configured")
    origin = settings.frontend_origins[0]

    response = client.options(
        "/api/chat",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
