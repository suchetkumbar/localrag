"""
Unit tests for ChatService.
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.services.chat_service import ChatService, ChatSource
from config.config import Settings, OllamaConfig, RetrievalConfig, ChatConfig


@pytest.fixture
def settings():
    s = MagicMock(spec=Settings)
    s.ollama = OllamaConfig()
    s.retrieval = RetrievalConfig(top_k=5, score_threshold=0.3)
    s.chat = ChatConfig(max_history_turns=5, max_context_tokens=3000)
    return s


@pytest.fixture
def mock_vector_store():
    vs = AsyncMock()
    vs.similarity_search = AsyncMock(return_value=[])
    return vs


@pytest.fixture
def chat_svc(settings, mock_vector_store):
    svc = ChatService.__new__(ChatService)
    svc.settings = settings
    svc.vector_store = mock_vector_store
    svc._llm = AsyncMock()
    return svc


# ---------------------------------------------------------------------------

def test_build_sources_deduplicates():
    from langchain.schema import Document as LCDocument

    svc = MagicMock()
    svc._build_sources = ChatService._build_sources.__get__(svc, ChatService)

    doc1 = LCDocument(
        page_content="content1",
        metadata={"filename": "a.pdf", "chunk_index": 0, "page": 1},
    )
    doc2 = LCDocument(
        page_content="content2",
        metadata={"filename": "a.pdf", "chunk_index": 0, "page": 1},
    )  # duplicate
    doc3 = LCDocument(
        page_content="content3",
        metadata={"filename": "b.pdf", "chunk_index": 1, "page": 2},
    )

    sources = svc._build_sources([(doc1, 0.9), (doc2, 0.85), (doc3, 0.7)])
    assert len(sources) == 2
    filenames = [s.filename for s in sources]
    assert "a.pdf" in filenames
    assert "b.pdf" in filenames


def test_chat_source_to_dict():
    source = ChatSource(
        filename="test.pdf",
        chunk_index=2,
        page=5,
        score=0.87654,
        snippet="This is a long snippet " * 20,
    )
    d = source.to_dict()
    assert d["filename"] == "test.pdf"
    assert d["chunk_index"] == 2
    assert d["page"] == 5
    assert d["score"] == round(0.87654, 4)
    assert len(d["snippet"]) <= 200


def test_build_messages_with_context(settings):
    from langchain.schema import Document as LCDocument

    svc = ChatService.__new__(ChatService)
    svc.settings = settings

    doc = LCDocument(
        page_content="Important context here.",
        metadata={"filename": "report.pdf", "chunk_index": 0},
    )

    messages = svc._build_messages([], "What is the report about?", [(doc, 0.9)])

    # Should have system + human
    assert len(messages) == 2
    # Last message should contain the context
    assert "Important context here." in messages[-1].content
    assert "[Source: report.pdf, chunk 0]" in messages[-1].content


def test_build_messages_no_context(settings):
    svc = ChatService.__new__(ChatService)
    svc.settings = settings

    messages = svc._build_messages([], "What is the answer?", [])
    assert len(messages) == 2
    assert "No relevant documents found" in messages[-1].content


@pytest.mark.asyncio
async def test_chat_persists_messages(chat_svc, mock_vector_store):
    from langchain.schema import Document as LCDocument

    mock_vector_store.similarity_search.return_value = [
        (
            LCDocument(page_content="answer content", metadata={"filename": "f.pdf", "chunk_index": 0, "page": 1}),
            0.9,
        )
    ]

    mock_llm_response = MagicMock()
    mock_llm_response.content = "This is the answer."
    chat_svc._llm.ainvoke = AsyncMock(return_value=mock_llm_response)

    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    mock_db.flush = AsyncMock()

    with patch.object(chat_svc, "_load_history", return_value=[]):
        with patch.object(chat_svc, "_persist_messages", new_callable=AsyncMock) as mock_persist:
            answer, sources = await chat_svc.chat("session-1", "What is this?", mock_db)

    assert answer == "This is the answer."
    assert len(sources) == 1
    assert sources[0].filename == "f.pdf"
    mock_persist.assert_awaited_once()
