import os
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from rag.cache import DiskCache
from rag.rerankers import CachedReranker, CrossEncoderReranker, rerank
from rag.types import ScoredChunk
from tests.helpers import make_chunks


class _LengthReranker:
    """Scores by text length, so the expected order is obvious."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def identity(self) -> Mapping[str, Any]:
        return {"provider": "test"}

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        self.calls += 1
        return [float(len(t)) for t in texts]


def test_rerank_reorders_all_candidates_by_new_score() -> None:
    a, b, c = make_chunks(["aa", "aaaa", "a"])
    candidates = [ScoredChunk(a, 0.9), ScoredChunk(b, 0.5), ScoredChunk(c, 0.1)]
    out = rerank(_LengthReranker(), "q", candidates)
    assert [r.chunk for r in out] == [b, a, c]
    assert [r.score for r in out] == [4.0, 2.0, 1.0]


def test_rerank_empty() -> None:
    assert rerank(_LengthReranker(), "q", []) == []


def test_cached_reranker(cache: DiskCache) -> None:
    inner = _LengthReranker()
    r = CachedReranker(inner, cache)
    assert r.score("q", ["ab", "c"]) == r.score("q", ["ab", "c"]) == [2.0, 1.0]
    assert inner.calls == 1
    r.score("other query", ["ab", "c"])
    assert inner.calls == 2


def test_cross_encoder_is_lazy() -> None:
    r = CrossEncoderReranker()
    assert r._encoder is None  # no model download just by constructing it
    assert r.score("q", []) == []


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("RAG_RUN_MODEL_TESTS") != "1", reason="downloads a ~280 MB model"
)
def test_real_cross_encoder_prefers_relevant_text() -> None:
    r = CrossEncoderReranker("BAAI/bge-reranker-base")
    relevant, irrelevant = r.score(
        "How do I declare a path parameter in FastAPI?",
        [
            "Declare path parameters with Python format syntax: @app.get('/items/{item_id}').",
            "Bananas are a good source of potassium.",
        ],
    )
    assert relevant > irrelevant
