"""Domain exceptions shared by the ingestion process and the API.

The API maps these exceptions to friendly HTTP responses. Internal details
(timeouts, stack traces, provider payloads) are logged server-side and never
returned to the browser.
"""

from __future__ import annotations


class AskPolicyError(Exception):
    """Base class for expected, user-actionable application errors."""

    #: Safe message that may be shown to an API consumer.
    public_message = "The request could not be completed."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.public_message)
        if message:
            self.public_message = message


class ConfigurationError(AskPolicyError):
    """A required setting or credential is missing or invalid."""

    public_message = "The service is not configured correctly."


class QdrantConnectionError(AskPolicyError):
    """Qdrant could not be reached or rejected the request."""

    public_message = "The policy index is currently unavailable. Please try again later."


class CollectionNotReadyError(AskPolicyError):
    """No indexed collection exists yet (ingestion has not been run)."""

    public_message = (
        "No policy documents are indexed yet. Add documents to the documents/ "
        "folder and run the ingestion command."
    )


class RetrievalError(AskPolicyError):
    """Searching the vector store failed."""

    public_message = "The policy search could not be completed. Please try again."


class LLMError(AskPolicyError):
    """The Groq chat completion failed."""

    public_message = "The answer service is temporarily unavailable. Please try again."


class IngestionError(AskPolicyError):
    """Document ingestion failed before the new index could be activated."""

    public_message = "Document ingestion failed. The previous index was kept."


class NoDocumentsFoundError(IngestionError):
    """There was nothing valid to index, so the old collection was preserved."""

    public_message = (
        "No supported documents with extractable text were found. "
        "The existing index was left unchanged."
    )
