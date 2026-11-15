"""Route modules for the AskPolicy API."""

from fastapi import APIRouter

from app.api.routes import chat, documents, health

api_router = APIRouter(prefix="/api")
api_router.include_router(health.router)
api_router.include_router(documents.router)
api_router.include_router(chat.router)

__all__ = ["api_router"]
