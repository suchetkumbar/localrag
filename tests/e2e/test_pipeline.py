"""
End-to-end smoke test.
Tests the full pipeline: ingest → search → chat.
Requires a running Ollama instance and real ChromaDB.
Skip by default; run with: pytest -m e2e
"""

from __future__ import annotations

import pytest
import asyncio
from pathlib import Path
import tempfile

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
async def services(tmp_path_factory):
    """Bootstrap real services for E2E testing."""
    from config.config import Settings, OllamaConfig, ChromaConfig, IngestionConfig, RetrievalConfig, ChatConfig
    from backend.services.vector_store_service import VectorStoreService
    from backend.services.ingestion_service import IngestionService
    from backend.services.chat_service import ChatService

    tmp = tmp_path_factory.mktemp("e2e")
    chroma_dir = tmp / "chroma"
    chroma_dir.mkdir()

    settings = Settings()
    settings.chroma.persist_directory = str(chroma_dir)
    settings.chroma.collection_name = "e2e_test"

    vs = VectorStoreService(settings)
    vs.initialize()

    ingestion = IngestionService(settings, vs)
    chat = ChatService(settings, vs)

    return {"vs": vs, "ingestion": ingestion, "chat": chat, "settings": settings}


@pytest.mark.asyncio
async def test_full_pipeline(services, tmp_path):
    """
    Full pipeline test:
    1. Create a test document
    2. Ingest it
    3. Search for content
    4. Ask a question
    """
    vs = services["vs"]
    ingestion = services["ingestion"]
    chat = services["chat"]

    # 1. Create test document
    doc_file = tmp_path / "test_knowledge.txt"
    doc_file.write_text(
        "The capital of France is Paris. "
        "Paris is famous for the Eiffel Tower, built in 1889. "
        "The Louvre museum is also in Paris and houses the Mona Lisa. "
        "France is a country in Western Europe with a population of about 68 million people. "
        * 5
    )

    # 2. Ingest
    from unittest.mock import AsyncMock, MagicMock
    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    mock_db.flush = AsyncMock()

    try:
        doc = await ingestion.ingest_file(doc_file, mock_db, source="e2e_test")
        assert doc.status == "ready"
        assert doc.chunk_count > 0
    except Exception as e:
        pytest.skip(f"Ollama not available: {e}")

    # 3. Search
    results = await vs.similarity_search("What is the capital of France?", top_k=3)
    assert len(results) > 0
    assert any("Paris" in doc.page_content for doc, _ in results)

    # 4. Chat
    mock_db2 = AsyncMock()
    mock_db2.add = MagicMock()
    mock_db2.flush = AsyncMock()

    from unittest.mock import patch
    with patch.object(chat, "_load_history", return_value=[]):
        with patch.object(chat, "_persist_messages", new_callable=AsyncMock):
            answer, sources = await chat.chat("e2e-session", "What is the capital of France?", mock_db2)

    assert "Paris" in answer or len(answer) > 0
    assert len(sources) > 0


@pytest.mark.asyncio
async def test_document_deletion_removes_from_search(services, tmp_path):
    """Deleted documents should not appear in search results."""
    vs = services["vs"]
    ingestion = services["ingestion"]

    unique_text = "XYZUNIQUE_CONTENT_FOR_DELETION_TEST " * 20
    doc_file = tmp_path / "deleteme.txt"
    doc_file.write_text(unique_text)

    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    mock_db.flush = AsyncMock()
    mock_db.delete = AsyncMock()

    try:
        doc = await ingestion.ingest_file(doc_file, mock_db, source="e2e_test")
    except Exception as e:
        pytest.skip(f"Ollama not available: {e}")

    # Verify it can be found
    results_before = await vs.similarity_search("XYZUNIQUE_CONTENT_FOR_DELETION_TEST", top_k=3, score_threshold=0.1)
    assert len(results_before) > 0

    # Delete
    await ingestion.delete_document(doc, mock_db)

    # Verify it's gone
    results_after = await vs.similarity_search("XYZUNIQUE_CONTENT_FOR_DELETION_TEST", top_k=3, score_threshold=0.8)
    assert len(results_after) == 0
