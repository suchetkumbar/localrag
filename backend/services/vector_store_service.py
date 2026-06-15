"""
Vector store service wrapping ChromaDB + Ollama embeddings.
All embedding calls go to the local Ollama instance.
"""

from __future__ import annotations

import asyncio
from functools import partial

import structlog
from langchain_chroma import Chroma
from langchain_core.documents import Document as LCDocument
from langchain_ollama import OllamaEmbeddings

from config.config import Settings

logger = structlog.get_logger(__name__)


class VectorStoreService:
    """
    Manages the ChromaDB collection.
    Embedding is done via Ollama (nomic-embed-text model).
    All add/search calls are run in a thread pool to avoid
    blocking the async event loop.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._store: Chroma | None = None
        self._embeddings: OllamaEmbeddings | None = None

    def initialize(self) -> None:
        """Synchronous initialization (call from startup)."""
        cfg = self.settings
        logger.info(
            "vector_store_init",
            model=cfg.ollama.embed_model,
            persist_dir=str(cfg.chroma_path),
        )
        self._embeddings = OllamaEmbeddings(
            model=cfg.ollama.embed_model,
            base_url=cfg.ollama.base_url,
        )
        self._store = Chroma(
            collection_name=cfg.chroma.collection_name,
            embedding_function=self._embeddings,
            persist_directory=str(cfg.chroma_path),
        )
        logger.info("vector_store_ready")

    @property
    def store(self) -> Chroma:
        if self._store is None:
            raise RuntimeError("VectorStoreService not initialized.")
        return self._store

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def add_documents(self, docs: list[LCDocument]) -> list[str]:
        """Add documents and return their Chroma IDs."""
        loop = asyncio.get_event_loop()
        ids = await loop.run_in_executor(None, self.store.add_documents, docs)
        logger.debug("chunks_added", count=len(ids))
        return ids

    async def delete_documents(self, ids: list[str]) -> None:
        """Delete documents by Chroma IDs."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, partial(self.store._collection.delete, ids=ids)
        )
        logger.debug("chunks_deleted", count=len(ids))

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.3,
    ) -> list[tuple[LCDocument, float]]:
        """
        Return (document, score) pairs.
        Score is cosine similarity (higher = more similar).
        """
        loop = asyncio.get_event_loop()
        results: list[tuple[LCDocument, float]] = await loop.run_in_executor(
            None,
            partial(
                self.store.similarity_search_with_relevance_scores,
                query,
                k=top_k,
            ),
        )
        filtered = [(doc, score) for doc, score in results if score >= score_threshold]
        logger.debug(
            "similarity_search",
            query_preview=query[:60],
            total=len(results),
            filtered=len(filtered),
        )
        return filtered

    async def get_collection_stats(self) -> dict:
        """Return basic stats about the collection."""
        loop = asyncio.get_event_loop()
        count = await loop.run_in_executor(
            None, lambda: self.store._collection.count()
        )
        return {
            "collection_name": self.settings.chroma.collection_name,
            "chunk_count": count,
            "embed_model": self.settings.ollama.embed_model,
            "persist_dir": str(self.settings.chroma_path),
        }
