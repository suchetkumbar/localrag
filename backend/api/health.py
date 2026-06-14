"""
Health check and Prometheus metrics endpoints.
GET /health   - liveness + readiness probe
GET /metrics  - Prometheus text exposition
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from sqlalchemy import text

from backend.api.deps import DBSession, SettingsDep, VectorStoreDep
from backend.models.schemas import HealthResponse

router = APIRouter(tags=["observability"])


@router.get("/health", response_model=HealthResponse)
async def health(
    db: DBSession,
    settings: SettingsDep,
    vector_store: VectorStoreDep,
) -> HealthResponse:
    """Liveness and readiness probe."""

    # Check Ollama
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{settings.ollama.base_url}/api/tags")
            ollama_ok = r.status_code == 200
    except Exception:
        pass

    # Check SQLite
    sqlite_ok = False
    try:
        await db.execute(text("SELECT 1"))
        sqlite_ok = True
    except Exception:
        pass

    # Check Chroma
    chroma_chunks = 0
    try:
        stats = await vector_store.get_collection_stats()
        chroma_chunks = stats["chunk_count"]
    except Exception:
        pass

    overall = "ok" if (ollama_ok and sqlite_ok) else "degraded"

    return HealthResponse(
        status=overall,
        version="1.0.0",
        ollama_connected=ollama_ok,
        chroma_chunks=chroma_chunks,
        sqlite_connected=sqlite_ok,
    )


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> PlainTextResponse:
    """Prometheus metrics endpoint."""
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )
