"""
Unit tests for VectorStoreService.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from langchain_core.documents import Document as LCDocument

from backend.services.vector_store_service import VectorStoreService
from config.config import ChromaConfig, OllamaConfig, Settings


@pytest.fixture
def settings():
    s = MagicMock(spec=Settings)
    s.ollama = OllamaConfig(
        base_url="http://localhost:11434",
        embed_model="nomic-embed-text",
    )
    s.chroma = ChromaConfig(
        persist_directory="./data/chroma",
        collection_name="documents",
    )
    s.chroma_path = MagicMock()
    s.chroma_path.__str__ = lambda self: "./data/chroma"
    return s


@pytest.fixture
def mock_chroma():
    chroma = MagicMock()
    collection = MagicMock()
    collection.count.return_value = 55
    collection.delete = MagicMock()
    chroma._collection = collection
    chroma.add_documents = MagicMock(return_value=["id1", "id2"])
    chroma.similarity_search_with_relevance_scores = MagicMock(
        return_value=[
            (LCDocument(page_content="Result 1", metadata={"filename": "a.pdf"}), 0.95),
            (LCDocument(page_content="Result 2", metadata={"filename": "b.pdf"}), 0.25),
        ]
    )
    return chroma


@pytest.fixture
def vs_service(settings, mock_chroma):
    svc = VectorStoreService(settings)
    svc._store = mock_chroma
    svc._embeddings = MagicMock()
    return svc


# ---------------------------------------------------------------------------

def test_store_raises_if_not_initialized():
    settings = MagicMock()
    svc = VectorStoreService(settings)
    with pytest.raises(RuntimeError, match="not initialized"):
        _ = svc.store


@pytest.mark.asyncio
async def test_add_documents(vs_service, mock_chroma):
    docs = [
        LCDocument(page_content="test content", metadata={"filename": "test.pdf"})
    ]
    ids = await vs_service.add_documents(docs)
    assert ids == ["id1", "id2"]
    mock_chroma.add_documents.assert_called_once_with(docs)


@pytest.mark.asyncio
async def test_delete_documents(vs_service, mock_chroma):
    await vs_service.delete_documents(["id1", "id2"])
    mock_chroma._collection.delete.assert_called_once_with(ids=["id1", "id2"])


@pytest.mark.asyncio
async def test_similarity_search_filters_by_threshold(vs_service):
    # Threshold 0.3 should filter out the 0.25 result
    results = await vs_service.similarity_search("query", top_k=5, score_threshold=0.3)
    assert len(results) == 1
    assert results[0][1] == 0.95


@pytest.mark.asyncio
async def test_similarity_search_returns_all_above_threshold(vs_service):
    results = await vs_service.similarity_search("query", top_k=5, score_threshold=0.2)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_get_collection_stats(vs_service, settings):
    stats = await vs_service.get_collection_stats()
    assert stats["chunk_count"] == 55
    assert stats["collection_name"] == "documents"
    assert stats["embed_model"] == "nomic-embed-text"
