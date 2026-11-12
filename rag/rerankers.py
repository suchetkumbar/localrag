"""Cross-encoder reranking.

A bi-encoder (the dense retriever) embeds query and chunk *separately*, which is fast
but loses query-chunk interaction. A cross-encoder reads ``(query, chunk)`` together
and outputs one relevance score: far more accurate, but O(candidates) model calls per
query, so it only re-orders the first stage's top ``candidate_k``.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from rag.cache import JSON_CODEC, DiskCache
from rag.interfaces import Reranker
from rag.types import ScoredChunk


class CrossEncoderReranker:
    """bge-reranker-base (or another model) via fastembed's ONNX runtime.

    fastembed runs on onnxruntime, avoiding a multi-gigabyte PyTorch install. The model
    is downloaded and loaded lazily on first use, so building a pipeline is cheap.
    """

    def __init__(self, model: str = "BAAI/bge-reranker-base", batch_size: int = 16) -> None:
        self.model = model
        self.batch_size = batch_size
        self._encoder: Any = None

    @property
    def identity(self) -> Mapping[str, Any]:
        return {"provider": "fastembed", "model": self.model}

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        if self._encoder is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._encoder = TextCrossEncoder(model_name=self.model)
        return [
            float(s) for s in self._encoder.rerank(query, list(texts), batch_size=self.batch_size)
        ]

    def close(self) -> None:
        """Free the model (about 1 GB of RAM); it is reloaded on the next ``score``."""
        self._encoder = None


class CachedReranker:
    """Caches the score list for one (query, candidate texts) call."""

    def __init__(self, inner: Reranker, cache: DiskCache) -> None:
        self.inner = inner
        self.cache = cache

    @property
    def identity(self) -> Mapping[str, Any]:
        return self.inner.identity

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        value, _ = self.cache.get_or_compute(
            "rerank",
            {"reranker": dict(self.inner.identity), "query": query, "texts": list(texts)},
            lambda: self.inner.score(query, texts),
            JSON_CODEC,
        )
        return [float(s) for s in value]

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if callable(close):
            close()


def rerank(reranker: Reranker, query: str, candidates: Sequence[ScoredChunk]) -> list[ScoredChunk]:
    """Re-order all candidates by reranker score (ties broken by chunk_id)."""
    scores = reranker.score(query, [c.chunk.indexed_text for c in candidates])
    rescored = [ScoredChunk(c.chunk, s) for c, s in zip(candidates, scores, strict=True)]
    return sorted(rescored, key=lambda r: (-r.score, r.chunk.chunk_id))
