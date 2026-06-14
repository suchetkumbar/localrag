"""
Unit tests for IngestionService.
Uses mocks to avoid real filesystem/ChromaDB calls.
"""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

from backend.services.ingestion_service import IngestionService, IngestionError
from config.config import Settings, IngestionConfig, OllamaConfig, ChromaConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def settings():
    s = MagicMock(spec=Settings)
    s.ingestion = IngestionConfig(
        chunk_size=512,
        chunk_overlap=64,
        supported_extensions=[".pdf", ".docx", ".md", ".txt"],
    )
    s.ollama = OllamaConfig()
    s.chroma = ChromaConfig()
    return s


@pytest.fixture
def mock_vector_store():
    vs = AsyncMock()
    vs.add_documents = AsyncMock(return_value=["id1", "id2", "id3"])
    vs.delete_documents = AsyncMock()
    return vs


@pytest.fixture
def ingestion_svc(settings, mock_vector_store):
    return IngestionService(settings, mock_vector_store)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_is_supported_pdf(ingestion_svc):
    assert ingestion_svc.is_supported(Path("test.pdf")) is True


def test_is_supported_docx(ingestion_svc):
    assert ingestion_svc.is_supported(Path("test.docx")) is True


def test_is_supported_markdown(ingestion_svc):
    assert ingestion_svc.is_supported(Path("test.md")) is True


def test_is_supported_txt(ingestion_svc):
    assert ingestion_svc.is_supported(Path("test.txt")) is True


def test_is_not_supported_csv(ingestion_svc):
    assert ingestion_svc.is_supported(Path("test.csv")) is False


def test_is_not_supported_exe(ingestion_svc):
    assert ingestion_svc.is_supported(Path("malware.exe")) is False


def test_unsupported_extension_raises(ingestion_svc):
    """_load_file should raise IngestionError for unsupported types."""
    with pytest.raises(IngestionError, match="Unsupported"):
        ingestion_svc._load_file(Path("test.xyz"))


def test_chunk_documents_adds_metadata(ingestion_svc):
    from langchain.schema import Document as LCDocument

    doc = LCDocument(page_content="Hello world " * 100, metadata={"source": "test.txt"})
    chunks = ingestion_svc._chunk_documents([doc], doc_id="abc-123", filename="test.txt")

    assert len(chunks) > 0
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["doc_id"] == "abc-123"
        assert chunk.metadata["filename"] == "test.txt"
        assert chunk.metadata["chunk_index"] == i
        assert chunk.metadata["chunk_total"] == len(chunks)


def test_chunk_documents_empty_raises(ingestion_svc):
    """Empty document list should produce zero chunks (not crash)."""
    chunks = ingestion_svc._chunk_documents([], doc_id="x", filename="x.txt")
    assert chunks == []


@pytest.mark.asyncio
async def test_ingest_file_success(ingestion_svc, mock_vector_store, tmp_path):
    """Full ingest pipeline with a real temp file."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test document. " * 50)

    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    mock_db.flush = AsyncMock()

    with patch.object(
        ingestion_svc, "_load_file",
        return_value=[
            MagicMock(page_content="chunk content", metadata={"source": "test.txt"})
        ]
    ):
        with patch.object(
            ingestion_svc, "_chunk_documents",
            return_value=[
                MagicMock(page_content="chunk", metadata={"doc_id": "x", "filename": "test.txt", "chunk_index": 0, "chunk_total": 1, "page": 0})
            ],
        ):
            doc = await ingestion_svc.ingest_file(test_file, mock_db, source="upload")

    assert doc.status == "ready"
    assert doc.chunk_count == 1
    mock_vector_store.add_documents.assert_awaited_once()


@pytest.mark.asyncio
async def test_ingest_file_marks_error_on_failure(ingestion_svc, tmp_path):
    """When loading fails, document status should be 'error'."""
    test_file = tmp_path / "bad.pdf"
    test_file.write_bytes(b"not a real pdf")

    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    mock_db.flush = AsyncMock()

    with patch.object(ingestion_svc, "_load_file", side_effect=Exception("parse error")):
        with pytest.raises(IngestionError):
            await ingestion_svc.ingest_file(test_file, mock_db, source="upload")

    # The document added to DB should have status="error"
    added_doc = mock_db.add.call_args_list[0][0][0]
    assert added_doc.status == "error"
    assert "parse error" in added_doc.error_message


@pytest.mark.asyncio
async def test_delete_document(ingestion_svc, mock_vector_store):
    import json
    from database.models import Document as DocumentModel

    doc = MagicMock(spec=DocumentModel)
    doc.id = "doc-id-1"
    doc.filename = "test.pdf"
    doc.chroma_ids = json.dumps(["c1", "c2", "c3"])

    mock_db = AsyncMock()
    mock_db.delete = AsyncMock()
    mock_db.flush = AsyncMock()

    await ingestion_svc.delete_document(doc, mock_db)

    mock_vector_store.delete_documents.assert_awaited_once_with(["c1", "c2", "c3"])
    mock_db.delete.assert_awaited_once_with(doc)
