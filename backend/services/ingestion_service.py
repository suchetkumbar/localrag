"""
Document ingestion service.
Handles parsing, chunking, embedding, and storing documents.
Supports PDF, DOCX, Markdown, and plain text.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import List, Tuple

import structlog
from langchain.schema import Document as LCDocument
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.ext.asyncio import AsyncSession

from config.config import Settings
from database.models import Document as DocumentModel
from monitoring.metrics import (
    DOCUMENTS_INGESTED_TOTAL,
    DOCUMENTS_INGESTION_ERRORS_TOTAL,
    DOCUMENTS_TOTAL,
    CHUNKS_TOTAL,
    INGESTION_DURATION_SECONDS,
)

logger = structlog.get_logger(__name__)


class IngestionError(Exception):
    pass


class IngestionService:
    """Parses, chunks, embeds, and stores documents."""

    def __init__(self, settings: Settings, vector_store_service):
        self.settings = settings
        self.vector_store = vector_store_service
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.ingestion.chunk_size,
            chunk_overlap=settings.ingestion.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def is_supported(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.settings.ingestion.supported_extensions

    async def ingest_file(
        self,
        file_path: Path,
        db_session: AsyncSession,
        source: str = "upload",
    ) -> DocumentModel:
        """
        Full ingestion pipeline for a single file.
        Returns the DocumentModel record.
        Raises IngestionError on failure.
        """
        start = time.time()
        file_type = file_path.suffix.lower().lstrip(".")

        log = logger.bind(filename=file_path.name, file_type=file_type, source=source)
        log.info("ingestion_started")

        # Create DB record (status=processing)
        doc_record = DocumentModel(
            id=str(uuid.uuid4()),
            filename=file_path.name,
            file_type=file_type,
            file_size=file_path.stat().st_size,
            source=source,
            status="processing",
        )
        db_session.add(doc_record)
        await db_session.flush()

        try:
            # 1. Parse
            raw_docs = self._load_file(file_path)
            log.info("file_parsed", page_count=len(raw_docs))

            # 2. Chunk
            chunks = self._chunk_documents(raw_docs, doc_record.id, file_path.name)
            log.info("file_chunked", chunk_count=len(chunks))

            # 3. Embed + store in ChromaDB
            chroma_ids = await self.vector_store.add_documents(chunks)
            log.info("embeddings_stored", chroma_id_count=len(chroma_ids))

            # 4. Update DB record
            doc_record.chunk_count = len(chunks)
            doc_record.chroma_ids = json.dumps(chroma_ids)
            doc_record.status = "ready"
            await db_session.flush()

            # Metrics
            duration = time.time() - start
            DOCUMENTS_INGESTED_TOTAL.labels(source=source, file_type=file_type).inc()
            INGESTION_DURATION_SECONDS.labels(file_type=file_type).observe(duration)
            CHUNKS_TOTAL.inc(len(chunks))

            log.info("ingestion_complete", duration_seconds=round(duration, 2))
            return doc_record

        except Exception as exc:
            doc_record.status = "error"
            doc_record.error_message = str(exc)
            await db_session.flush()

            DOCUMENTS_INGESTION_ERRORS_TOTAL.labels(
                file_type=file_type, error_type=type(exc).__name__
            ).inc()

            log.error("ingestion_failed", error=str(exc), exc_info=True)
            raise IngestionError(f"Failed to ingest {file_path.name}: {exc}") from exc

    def _load_file(self, file_path: Path) -> List[LCDocument]:
        """Load a file using the appropriate LangChain loader."""
        suffix = file_path.suffix.lower()
        loaders = {
            ".pdf": lambda: PyPDFLoader(str(file_path)),
            ".docx": lambda: Docx2txtLoader(str(file_path)),
            ".md": lambda: UnstructuredMarkdownLoader(str(file_path)),
            ".txt": lambda: TextLoader(str(file_path), encoding="utf-8"),
        }
        loader_factory = loaders.get(suffix)
        if loader_factory is None:
            raise IngestionError(f"Unsupported file type: {suffix}")

        loader = loader_factory()
        docs = loader.load()

        if not docs:
            raise IngestionError(f"No content extracted from {file_path.name}")

        return docs

    def _chunk_documents(
        self,
        docs: List[LCDocument],
        doc_id: str,
        filename: str,
    ) -> List[LCDocument]:
        """Split documents into chunks and enrich metadata."""
        chunks = self.splitter.split_documents(docs)

        for i, chunk in enumerate(chunks):
            chunk.metadata.update(
                {
                    "doc_id": doc_id,
                    "filename": filename,
                    "chunk_index": i,
                    "chunk_total": len(chunks),
                }
            )
            # Normalise page number
            if "page" not in chunk.metadata:
                chunk.metadata["page"] = 0

        return chunks

    async def delete_document(
        self,
        doc_record: DocumentModel,
        db_session: AsyncSession,
    ) -> None:
        """Remove a document and all its chunks from ChromaDB and SQLite."""
        if doc_record.chroma_ids:
            ids = json.loads(doc_record.chroma_ids)
            await self.vector_store.delete_documents(ids)
            CHUNKS_TOTAL.dec(len(ids))

        await db_session.delete(doc_record)
        await db_session.flush()
        DOCUMENTS_TOTAL.dec()
        logger.info("document_deleted", doc_id=doc_record.id, filename=doc_record.filename)
