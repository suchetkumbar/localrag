"""
LocalRAG FastAPI application.
Run with: uvicorn backend.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import deps
from backend.api.chat import router as chat_router
from backend.api.documents import router as documents_router
from backend.api.health import router as health_router
from backend.api.sessions import router as sessions_router
from backend.middleware.middleware import APIKeyMiddleware, LoggingMiddleware
from backend.services.chat_service import ChatService
from backend.services.ingestion_service import IngestionService
from backend.services.logging_service import setup_logging
from backend.services.vector_store_service import VectorStoreService
from backend.services.watcher_service import FolderWatcherService
from config.config import ensure_directories, get_settings
from database.models import create_tables, init_engine

logger = structlog.get_logger(__name__)

settings = get_settings()
ensure_directories(settings)
setup_logging(settings.log_level, settings.log_file_path)

_watcher: FolderWatcherService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    global _watcher

    logger.info("app_starting", version="1.0.0", env=settings.app_env)

    # Initialize database
    init_engine(str(settings.sqlite_path))
    await create_tables()
    logger.info("database_ready")

    # Initialize vector store
    vector_store = VectorStoreService(settings)
    vector_store.initialize()

    # Initialize services
    ingestion_svc = IngestionService(settings, vector_store)
    chat_svc = ChatService(settings, vector_store)

    # Register with DI
    deps.set_services(vector_store, ingestion_svc, chat_svc)

    # Start folder watcher
    from database.models import _session_factory

    _watcher = FolderWatcherService(settings, ingestion_svc, _session_factory)
    await _watcher.start()

    logger.info("app_ready", host=settings.server.host, port=settings.server.port)

    yield

    # Shutdown
    logger.info("app_stopping")
    if _watcher:
        await _watcher.stop()
    from database.models import close_engine

    await close_engine()
    logger.info("app_stopped")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title="LocalRAG",
        description="100% local RAG API — powered by Ollama + ChromaDB + FastAPI",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom middleware (order: outermost first)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(
        APIKeyMiddleware,
        api_key=settings.api_key,
        header_name=settings.security.api_key_header,
        enabled=settings.api_key_enabled,
    )

    # Routers
    app.include_router(health_router)
    app.include_router(sessions_router)
    app.include_router(chat_router)
    app.include_router(documents_router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.reload,
        workers=settings.server.workers,
        log_config=None,  # We handle logging ourselves
    )
