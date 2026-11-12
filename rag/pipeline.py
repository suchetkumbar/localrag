"""The RAG pipeline: chunk -> index -> retrieve -> (rerank) -> generate.

Unlike ``backend/services/chat_service.py`` this pipeline is stateless (no chat history,
no database writes), so answering the same question with the same config always does
the same work. That is a precondition for meaningful evaluation.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from rag.interfaces import Chunker, Generator, Reranker, Retriever
from rag.prompts import PromptTemplate, format_context
from rag.rerankers import rerank
from rag.types import Chunk, Document, Generation, ScoredChunk


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    ranking: list[ScoredChunk]  # full final ranking, depth candidate_k; what metrics score
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Answer:
    question: str
    ranking: list[ScoredChunk]
    context: list[ScoredChunk]  # chunks shown to the LLM; citation [n] is context[n-1]
    generation: Generation
    timings_ms: dict[str, float]


class RAGPipeline:
    def __init__(
        self,
        chunker: Chunker,
        retriever: Retriever,
        generator: Generator,
        prompt: PromptTemplate,
        *,
        candidate_k: int = 50,
        top_k: int = 5,
        max_context_chars: int = 8000,
        reranker: Reranker | None = None,
    ) -> None:
        if top_k > candidate_k:
            raise ValueError("top_k must be <= candidate_k")
        self.chunker = chunker
        self.retriever = retriever
        self.generator = generator
        self.prompt = prompt
        self.candidate_k = candidate_k
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.reranker = reranker
        self.chunks: list[Chunk] = []

    def index(self, docs: Sequence[Document]) -> list[Chunk]:
        self.chunks = [c for doc in docs for c in self.chunker.chunk(doc)]
        self.retriever.index(self.chunks)
        return self.chunks

    def retrieve(self, question: str) -> RetrievalResult:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        ranking = self.retriever.search(question, self.candidate_k)
        timings["retrieve_ms"] = (time.perf_counter() - t0) * 1000
        if self.reranker is not None:
            t0 = time.perf_counter()
            ranking = rerank(self.reranker, question, ranking)
            timings["rerank_ms"] = (time.perf_counter() - t0) * 1000
        return RetrievalResult(ranking=ranking, timings_ms=timings)

    def answer(self, question: str) -> Answer:
        retrieval = self.retrieve(question)
        context_text, included = format_context(
            retrieval.ranking[: self.top_k], self.max_context_chars
        )
        user = self.prompt.render_user(
            question=question, context=context_text or "(no passages retrieved)"
        )
        t0 = time.perf_counter()
        generation = self.generator.generate(self.prompt.system, user)
        timings = dict(retrieval.timings_ms)
        # Wall time of this call. For cached generations the original model latency is
        # in generation.latency_ms; the eval report uses that one.
        timings["generate_ms"] = (time.perf_counter() - t0) * 1000
        return Answer(
            question=question,
            ranking=retrieval.ranking,
            context=included,
            generation=generation,
            timings_ms=timings,
        )
