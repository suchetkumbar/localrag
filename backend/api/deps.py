"""
Dependency injection for FastAPI route handlers.
All services are singletons created at startup and injected via Depends().
"""

from __future__ import annotations

from typing import Annotated, AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from config.config import Settings, get_settings
from database.models import get_session
from backend.services.chat_service import ChatService
from backend.services.ingestion_service import IngestionService
from backend.services.session_service import SessionService
from backend.services.vector_store_service import VectorStoreService

# These are populated during application startup (see main.py)
_vector_store: VectorStoreService | None = None
_ingestion_svc: IngestionService | None = None
_chat_svc: ChatService | None = None
_session_svc: SessionService = SessionService()


def set_services(
    vector_store: VectorStoreService,
    ingestion_svc: IngestionService,
    chat_svc: ChatService,
) -> None:
    global _vector_store, _ingestion_svc, _chat_svc
    _vector_store = vector_store
    _ingestion_svc = ingestion_svc
    _chat_svc = chat_svc


# ---------------------------------------------------------------------------
# FastAPI Depends functions
# ---------------------------------------------------------------------------

def get_app_settings() -> Settings:
    return get_settings()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async for session in get_session():
        yield session


def get_vector_store() -> VectorStoreService:
    if _vector_store is None:
        raise RuntimeError("VectorStoreService not initialized")
    return _vector_store


def get_ingestion_service() -> IngestionService:
    if _ingestion_svc is None:
        raise RuntimeError("IngestionService not initialized")
    return _ingestion_svc


def get_chat_service() -> ChatService:
    if _chat_svc is None:
        raise RuntimeError("ChatService not initialized")
    return _chat_svc


def get_session_service() -> SessionService:
    return _session_svc


# Annotated shorthands
SettingsDep = Annotated[Settings, Depends(get_app_settings)]
DBSession = Annotated[AsyncSession, Depends(get_db_session)]
VectorStoreDep = Annotated[VectorStoreService, Depends(get_vector_store)]
IngestionDep = Annotated[IngestionService, Depends(get_ingestion_service)]
ChatDep = Annotated[ChatService, Depends(get_chat_service)]
SessionSvcDep = Annotated[SessionService, Depends(get_session_service)]
