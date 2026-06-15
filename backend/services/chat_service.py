"""
RAG chat service.
Retrieves relevant chunks from ChromaDB, builds a prompt with
conversation history and citations, then calls Ollama for generation.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

import structlog
from langchain_core.documents import Document as LCDocument
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_ollama import ChatOllama
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.services.vector_store_service import VectorStoreService
from config.config import Settings
from database.models import Message
from monitoring.metrics import (
    LLM_DURATION_SECONDS,
    QUERIES_TOTAL,
    QUERY_DURATION_SECONDS,
    QUERY_ERRORS_TOTAL,
    RETRIEVAL_CHUNKS_RETURNED,
)

logger = structlog.get_logger(__name__)


class ChatSource:
    """Represents a citation source in a response."""

    def __init__(self, filename: str, chunk_index: int, page: int, score: float, snippet: str):
        self.filename = filename
        self.chunk_index = chunk_index
        self.page = page
        self.score = score
        self.snippet = snippet

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "chunk_index": self.chunk_index,
            "page": self.page,
            "score": round(self.score, 4),
            "snippet": self.snippet[:200],
        }


class ChatService:
    """Orchestrates retrieval + LLM generation with multi-turn memory."""

    def __init__(self, settings: Settings, vector_store: VectorStoreService):
        self.settings = settings
        self.vector_store = vector_store
        self._llm = ChatOllama(
            model=settings.ollama.chat_model,
            base_url=settings.ollama.base_url,
            timeout=settings.ollama.timeout,
            temperature=0.1,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chat(
        self,
        session_id: str,
        user_message: str,
        db_session: AsyncSession,
    ) -> tuple[str, list[ChatSource]]:
        """
        Process a user message and return (answer, sources).
        Persists both user message and assistant response to SQLite.
        """
        start = time.time()
        log = logger.bind(session_id=session_id)
        log.info("chat_request", message_preview=user_message[:80])

        try:
            # 1. Retrieve relevant chunks
            t_retrieve = time.time()
            chunks_with_scores = await self.vector_store.similarity_search(
                user_message,
                top_k=self.settings.retrieval.top_k,
                score_threshold=self.settings.retrieval.score_threshold,
            )
            RETRIEVAL_CHUNKS_RETURNED.observe(len(chunks_with_scores))
            log.debug("retrieval_done", chunks=len(chunks_with_scores), duration=round(time.time() - t_retrieve, 2))

            # 2. Build sources list
            sources = self._build_sources(chunks_with_scores)

            # 3. Load conversation history
            history = await self._load_history(session_id, db_session)

            # 4. Build LangChain messages
            messages = self._build_messages(history, user_message, chunks_with_scores)

            # 5. Call LLM
            t_llm = time.time()
            response = await self._llm.ainvoke(messages)
            answer = response.content
            llm_duration = time.time() - t_llm
            LLM_DURATION_SECONDS.observe(llm_duration)
            log.debug("llm_response", duration=round(llm_duration, 2), answer_len=len(answer))

            # 6. Persist messages
            await self._persist_messages(session_id, user_message, answer, sources, db_session)

            # Metrics
            total_duration = time.time() - start
            QUERIES_TOTAL.labels(session_id=session_id).inc()
            QUERY_DURATION_SECONDS.observe(total_duration)

            log.info("chat_complete", duration_seconds=round(total_duration, 2), source_count=len(sources))
            return answer, sources

        except Exception as exc:
            QUERY_ERRORS_TOTAL.labels(error_type=type(exc).__name__).inc()
            log.error("chat_failed", error=str(exc), exc_info=True)
            raise

    async def stream_chat(
        self,
        session_id: str,
        user_message: str,
        db_session: AsyncSession,
    ) -> AsyncIterator[str]:
        """
        Stream tokens from the LLM.
        Yields token strings. Final yield is a JSON sources block.
        """
        chunks_with_scores = await self.vector_store.similarity_search(
            user_message,
            top_k=self.settings.retrieval.top_k,
            score_threshold=self.settings.retrieval.score_threshold,
        )
        sources = self._build_sources(chunks_with_scores)
        history = await self._load_history(session_id, db_session)
        messages = self._build_messages(history, user_message, chunks_with_scores)

        full_answer = []
        async for chunk in self._llm.astream(messages):
            token = chunk.content
            full_answer.append(token)
            yield token

        answer = "".join(full_answer)
        await self._persist_messages(session_id, user_message, answer, sources, db_session)

        # Signal sources at the end
        yield "\n\n__SOURCES__:" + json.dumps([s.to_dict() for s in sources])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_sources(
        self, chunks_with_scores: list[tuple[LCDocument, float]]
    ) -> list[ChatSource]:
        seen = set()
        sources = []
        for doc, score in chunks_with_scores:
            m = doc.metadata
            key = (m.get("filename", ""), m.get("chunk_index", 0))
            if key not in seen:
                seen.add(key)
                sources.append(
                    ChatSource(
                        filename=m.get("filename", "unknown"),
                        chunk_index=m.get("chunk_index", 0),
                        page=m.get("page", 0),
                        score=score,
                        snippet=doc.page_content[:300],
                    )
                )
        return sources

    def _build_messages(
        self,
        history: list[Message],
        user_message: str,
        chunks_with_scores: list[tuple[LCDocument, float]],
    ) -> list:
        cfg = self.settings.chat
        messages = [SystemMessage(content=cfg.system_prompt)]

        # Conversation history (limited to max_history_turns pairs)
        turns = history[-(cfg.max_history_turns * 2):]
        for msg in turns:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            else:
                messages.append(AIMessage(content=msg.content))

        # Build context block
        if chunks_with_scores:
            context_parts = []
            token_budget = cfg.max_context_tokens
            for doc, score in chunks_with_scores:
                m = doc.metadata
                citation = f"[Source: {m.get('filename', 'unknown')}, chunk {m.get('chunk_index', 0)}]"
                entry = f"{citation}\n{doc.page_content}"
                if len(entry) > token_budget:
                    break
                context_parts.append(entry)
                token_budget -= len(entry)

            context = "\n\n---\n\n".join(context_parts)
            augmented_query = (
                f"Context from documents:\n\n{context}\n\n"
                f"Question: {user_message}"
            )
        else:
            augmented_query = (
                f"No relevant documents found for this question.\n\n"
                f"Question: {user_message}"
            )

        messages.append(HumanMessage(content=augmented_query))
        return messages

    async def _load_history(
        self, session_id: str, db_session: AsyncSession
    ) -> list[Message]:
        result = await db_session.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at)
        )
        return list(result.scalars().all())

    async def _persist_messages(
        self,
        session_id: str,
        user_message: str,
        answer: str,
        sources: list[ChatSource],
        db_session: AsyncSession,
    ) -> None:
        db_session.add(Message(session_id=session_id, role="user", content=user_message))
        db_session.add(
            Message(
                session_id=session_id,
                role="assistant",
                content=answer,
                sources=json.dumps([s.to_dict() for s in sources]),
            )
        )
        await db_session.flush()
