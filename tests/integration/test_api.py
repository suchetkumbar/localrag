"""
Integration tests for FastAPI API endpoints.
Uses httpx AsyncClient with the full app (mocked services).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from backend.api import deps
from backend.main import create_app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_vector_store():
    vs = AsyncMock()
    vs.get_collection_stats = AsyncMock(return_value={
        "collection_name": "documents",
        "chunk_count": 42,
        "embed_model": "nomic-embed-text",
        "persist_dir": "./data/chroma",
    })
    vs.similarity_search = AsyncMock(return_value=[])
    return vs


@pytest.fixture
def mock_ingestion():
    svc = AsyncMock()
    return svc


@pytest.fixture
def mock_chat():
    svc = AsyncMock()
    from backend.services.chat_service import ChatSource
    svc.chat = AsyncMock(return_value=(
        "This is the answer.",
        [ChatSource(filename="test.pdf", chunk_index=0, page=1, score=0.9, snippet="snippet")],
    ))
    return svc


@pytest.fixture
def mock_session_svc():
    svc = AsyncMock()
    from datetime import datetime

    from database.models import Session as SessionModel

    session = MagicMock(spec=SessionModel)
    session.id = "test-session-id"
    session.title = "Test Chat"
    session.created_at = datetime(2024, 1, 1, 12, 0, 0)
    session.updated_at = datetime(2024, 1, 1, 12, 0, 0)
    session.messages = []

    svc.create_session = AsyncMock(return_value=session)
    svc.get_session = AsyncMock(return_value=session)
    svc.list_sessions = AsyncMock(return_value=[session])
    svc.delete_session = AsyncMock(return_value=True)
    svc.rename_session = AsyncMock(return_value=session)
    svc.get_messages = AsyncMock(return_value=[])
    return svc


@pytest.fixture
async def client(mock_vector_store, mock_ingestion, mock_chat, mock_session_svc):
    app = create_app()
    deps.set_services(mock_vector_store, mock_ingestion, mock_chat)

    # Override session service
    app.dependency_overrides[deps.get_session_service] = lambda: mock_session_svc
    app.dependency_overrides[deps.get_vector_store] = lambda: mock_vector_store
    app.dependency_overrides[deps.get_ingestion_service] = lambda: mock_ingestion
    app.dependency_overrides[deps.get_chat_service] = lambda: mock_chat

    # Override DB session
    async def noop_db():
        db = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = []
        result.scalar_one_or_none.return_value = None
        db.execute = AsyncMock(return_value=result)
        yield db

    app.dependency_overrides[deps.get_db_session] = noop_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_endpoint(client):
    with patch("backend.api.health.httpx.AsyncClient") as mock_httpx:
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_httpx.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_resp)

        resp = await client.get("/health")

    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "version" in data
    assert "ollama_connected" in data


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_session(client, mock_session_svc):
    resp = await client.post("/api/sessions", json={"title": "My Chat"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["id"] == "test-session-id"
    assert data["title"] == "Test Chat"


@pytest.mark.asyncio
async def test_list_sessions(client):
    resp = await client.get("/api/sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 1


@pytest.mark.asyncio
async def test_get_session(client):
    resp = await client.get("/api/sessions/test-session-id")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "test-session-id"
    assert "messages" in data


@pytest.mark.asyncio
async def test_get_session_not_found(client, mock_session_svc):
    mock_session_svc.get_session = AsyncMock(return_value=None)
    resp = await client.get("/api/sessions/nonexistent-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_session(client):
    resp = await client.delete("/api/sessions/test-session-id")
    assert resp.status_code == 204


@pytest.mark.asyncio
async def test_rename_session(client):
    resp = await client.patch("/api/sessions/test-session-id", json={"title": "Renamed"})
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_endpoint(client):
    resp = await client.post(
        "/api/sessions/test-session-id/chat",
        json={"message": "What is in my documents?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "This is the answer."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["filename"] == "test.pdf"
    assert data["session_id"] == "test-session-id"


@pytest.mark.asyncio
async def test_chat_session_not_found(client, mock_session_svc):
    mock_session_svc.get_session = AsyncMock(return_value=None)
    resp = await client.post(
        "/api/sessions/bad-session/chat",
        json={"message": "hello"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_chat_empty_message_rejected(client):
    resp = await client.post(
        "/api/sessions/test-session-id/chat",
        json={"message": ""},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_document_stats(client):
    resp = await client.get("/api/documents/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["chunk_count"] == 42
    assert data["collection_name"] == "documents"


@pytest.mark.asyncio
async def test_list_documents_empty(client):
    resp = await client.get("/api/documents")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_upload_unsupported_type(client, tmp_path):
    content = b"some content"
    resp = await client.post(
        "/api/documents/upload",
        files={"file": ("test.csv", content, "text/csv")},
    )
    assert resp.status_code == 422
