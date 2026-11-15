"""AskPolicy FastAPI application.

Run from the ``backend`` directory:

    uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api import api_router
from app.core.config import settings
from app.core.errors import (
    AskPolicyError,
    CollectionNotReadyError,
    ConfigurationError,
    IngestionError,
    LLMError,
    QdrantConnectionError,
    RetrievalError,
)
from app.core.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Log configuration state without crashing on missing optional values."""
    logger.info("Starting AskPolicy API v%s (%s)", __version__, settings.app_env)
    if not settings.qdrant_is_configured:
        logger.warning("Qdrant credentials are missing; document endpoints will be degraded")
    if not settings.groq_is_configured:
        logger.warning("Groq credentials are missing; chat requests will fail")
    yield
    logger.info("AskPolicy API stopped")


app = FastAPI(
    title="AskPolicy API",
    description="Grounded question answering over company policy documents.",
    version=__version__,
    lifespan=lifespan,
)

# Only the React development origins listed in .env may call the API.
# Credentials are not used in Version 1, so allow_credentials stays False.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.frontend_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log method, path, status, and duration (never headers or bodies)."""
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "%s %s -> %d (%.0f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


app.include_router(api_router)


def _status_for(exc: AskPolicyError) -> int:
    """Map domain errors to meaningful HTTP status codes."""
    if isinstance(exc, (ConfigurationError, CollectionNotReadyError, QdrantConnectionError)):
        return 503
    if isinstance(exc, RetrievalError):
        return 503
    if isinstance(exc, LLMError):
        return 502
    if isinstance(exc, IngestionError):
        return 500
    return 500


@app.exception_handler(AskPolicyError)
async def handle_askpolicy_error(request: Request, exc: AskPolicyError) -> JSONResponse:
    """Return safe messages for expected failures.

    Internal exception text is logged here but the response only contains a
    user-actionable message.
    """
    status_code = _status_for(exc)
    logger.warning(
        "Request failed (%s %s): %s",
        request.method,
        request.url.path,
        exc,
    )
    return JSONResponse(
        status_code=status_code,
        content={"detail": exc.public_message},
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Log unexpected exceptions and hide implementation details from clients."""
    logger.exception(
        "Unhandled error during %s %s",
        request.method,
        request.url.path,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."},
    )
