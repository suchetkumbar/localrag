"""
Tests for MCP server tool implementations.
Validates tool schemas, inputs, outputs, and error handling.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_doc(
    doc_id="doc-123",
    filename="test.pdf",
    file_type="pdf",
    status="ready",
    chunk_count=5,
):
    from datetime import datetime

    from database.models import Document as DocumentModel

    doc = MagicMock(spec=DocumentModel)
    doc.id = doc_id
    doc.filename = filename
    doc.file_type = file_type
    doc.file_size = 1024
    doc.chunk_count = chunk_count
    doc.status = status
    doc.source = "upload"
    doc.error_message = None
    doc.chroma_ids = json.dumps([f"c{i}" for i in range(chunk_count)])
    doc.created_at = datetime(2024, 1, 1)
    doc.updated_at = datetime(2024, 1, 1)
    return doc


# ---------------------------------------------------------------------------
# search_documents tool
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_search_documents_returns_results():
    from langchain_core.documents import Document as LCDocument

    from mcp_server.server import _tool_search_documents

    mock_result = [
        (
            LCDocument(
                page_content="Relevant content here.",
                metadata={"filename": "report.pdf", "chunk_index": 2, "page": 1},
            ),
            0.87,
        )
    ]

    with patch("mcp_server.server._vector_store") as mock_vs:
        mock_vs.similarity_search = AsyncMock(return_value=mock_result)
        result = await _tool_search_documents({"query": "what is the report about?"})

    assert not result.isError
    assert "report.pdf" in result.content[0].text
    assert "0.8700" in result.content[0].text


@pytest.mark.asyncio
async def test_tool_search_documents_no_results():
    from mcp_server.server import _tool_search_documents

    with patch("mcp_server.server._vector_store") as mock_vs:
        mock_vs.similarity_search = AsyncMock(return_value=[])
        result = await _tool_search_documents({"query": "something obscure"})

    assert not result.isError
    assert "No relevant documents" in result.content[0].text


# ---------------------------------------------------------------------------
# ingest_document tool
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_ingest_document_file_not_found():
    from mcp_server.server import _tool_ingest_document

    result = await _tool_ingest_document({"file_path": "/nonexistent/path/file.pdf"})

    assert result.isError
    assert "not found" in result.content[0].text.lower()


@pytest.mark.asyncio
async def test_tool_ingest_document_unsupported_type(tmp_path):
    from mcp_server.server import _tool_ingest_document

    bad_file = tmp_path / "test.xyz"
    bad_file.write_text("content")

    with patch("mcp_server.server._ingestion_svc") as mock_svc:
        mock_svc.is_supported = MagicMock(return_value=False)
        result = await _tool_ingest_document({"file_path": str(bad_file)})

    assert result.isError
    assert "Unsupported" in result.content[0].text


@pytest.mark.asyncio
async def test_tool_ingest_document_success(tmp_path):
    from mcp_server.server import _tool_ingest_document

    good_file = tmp_path / "report.pdf"
    good_file.write_bytes(b"fake pdf")

    mock_doc = make_doc()

    with patch("mcp_server.server._ingestion_svc") as mock_svc, \
         patch("mcp_server.server.get_session") as mock_get_session:

        mock_svc.is_supported = MagicMock(return_value=True)
        mock_svc.ingest_file = AsyncMock(return_value=mock_doc)

        # Mock async context manager
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=AsyncMock())
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_get_session.return_value = mock_ctx

        result = await _tool_ingest_document({"file_path": str(good_file)})

    assert not result.isError
    parsed = json.loads(result.content[0].text)
    assert parsed["id"] == "doc-123"
    assert parsed["status"] == "ready"


# ---------------------------------------------------------------------------
# delete_document tool
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_delete_document_not_found():
    from mcp_server.server import _tool_delete_document

    with patch("mcp_server.server.get_session") as mock_get_session:
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = result_mock

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_db)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_get_session.return_value = mock_ctx

        result = await _tool_delete_document({"document_id": "nonexistent"})

    assert result.isError
    assert "not found" in result.content[0].text.lower()


# ---------------------------------------------------------------------------
# get_collection_stats tool
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_get_collection_stats():
    from mcp_server.server import _tool_get_collection_stats

    expected = {
        "collection_name": "documents",
        "chunk_count": 100,
        "embed_model": "nomic-embed-text",
        "persist_dir": "./data/chroma",
    }

    with patch("mcp_server.server._vector_store") as mock_vs:
        mock_vs.get_collection_stats = AsyncMock(return_value=expected)
        result = await _tool_get_collection_stats({})

    assert not result.isError
    parsed = json.loads(result.content[0].text)
    assert parsed["chunk_count"] == 100
    assert parsed["embed_model"] == "nomic-embed-text"


# ---------------------------------------------------------------------------
# ask_question tool
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_ask_question_success():
    from backend.services.chat_service import ChatSource
    from mcp_server.server import _tool_ask_question

    mock_sources = [
        ChatSource(filename="doc.pdf", chunk_index=0, page=1, score=0.9, snippet="snippet text")
    ]

    with patch("mcp_server.server._chat_svc") as mock_chat, \
         patch("mcp_server.server._session_svc") as mock_sess, \
         patch("mcp_server.server.get_session") as mock_get_session:

        mock_chat.chat = AsyncMock(return_value=("The answer is 42.", mock_sources))
        mock_sess.get_session = AsyncMock(return_value=MagicMock())

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=AsyncMock())
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_get_session.return_value = mock_ctx

        result = await _tool_ask_question({
            "session_id": "sess-1",
            "question": "What is the answer?",
        })

    assert not result.isError
    parsed = json.loads(result.content[0].text)
    assert parsed["answer"] == "The answer is 42."
    assert len(parsed["sources"]) == 1
    assert parsed["sources"][0]["filename"] == "doc.pdf"
