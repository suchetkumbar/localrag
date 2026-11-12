import math
from collections.abc import Sequence

import pytest

from rag.embedders import FakeEmbedder
from rag.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    reciprocal_rank_fusion,
    tokenize,
)
from rag.types import Chunk, ScoredChunk
from tests.helpers import make_chunks


def test_tokenize_keeps_identifiers() -> None:
    assert tokenize("Use response_model=Item, OK?") == ["use", "response_model", "item", "ok"]


class TestBM25:
    @pytest.fixture
    def bm25(self) -> tuple[BM25Retriever, list[Chunk]]:
        chunks = make_chunks(["cat sat", "cat cat dog", "bird"])
        r = BM25Retriever(k1=1.5, b=0.75)
        r.index(chunks)
        return r, chunks

    def test_scores_match_hand_computation(self, bm25: tuple[BM25Retriever, list[Chunk]]) -> None:
        r, _ = bm25
        # N=3, df(cat)=2, avgdl=(2+3+1)/3=2
        idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))  # ln(1.6)
        # chunk 0: tf=1, |d|=2 -> norm = 1 - 0.75 + 0.75 * 2/2 = 1.0
        s0 = idf * 1 * 2.5 / (1 + 1.5 * 1.0)
        # chunk 1: tf=2, |d|=3 -> norm = 0.25 + 0.75 * 3/2 = 1.375
        s1 = idf * 2 * 2.5 / (2 + 1.5 * 1.375)
        scores = r.scores("cat")
        assert scores.keys() == {0, 1}
        assert scores[0] == pytest.approx(s0)
        assert scores[1] == pytest.approx(s1)

    def test_ranking_and_k(self, bm25: tuple[BM25Retriever, list[Chunk]]) -> None:
        r, chunks = bm25
        results = r.search("cat", k=1)
        assert [x.chunk for x in results] == [chunks[1]]  # higher tf wins

    def test_duplicate_query_terms_count_once(
        self, bm25: tuple[BM25Retriever, list[Chunk]]
    ) -> None:
        r, _ = bm25
        assert r.scores("cat cat cat") == r.scores("cat")

    def test_rare_terms_weigh_more(self, bm25: tuple[BM25Retriever, list[Chunk]]) -> None:
        r, chunks = bm25
        # "dog" (df=1) outweighs the common "cat" (df=2): chunk 1 wins over chunk 0.
        assert r.search("dog cat", k=3)[0].chunk == chunks[1]

    def test_no_match_returns_empty(self, bm25: tuple[BM25Retriever, list[Chunk]]) -> None:
        r, _ = bm25
        assert r.search("unicorn", k=5) == []

    def test_ties_broken_by_chunk_id(self) -> None:
        chunks = make_chunks(["same text", "same text"])
        r = BM25Retriever()
        r.index(chunks)
        ids = [x.chunk.chunk_id for x in r.search("same", k=2)]
        assert ids == sorted(ids)


class TestRRF:
    def test_exact_scores_and_order(self) -> None:
        x, y, z, w = make_chunks(["x", "y", "z", "w"])
        a = [ScoredChunk(x, 9.0), ScoredChunk(y, 5.0), ScoredChunk(z, 1.0)]
        b = [ScoredChunk(y, 0.9), ScoredChunk(w, 0.8)]
        fused = reciprocal_rank_fusion([a, b], k=60)
        assert [f.chunk for f in fused] == [y, x, w, z]
        expected = [1 / 61 + 1 / 62, 1 / 61, 1 / 62, 1 / 63]
        assert [f.score for f in fused] == pytest.approx(expected)

    def test_raw_scores_are_ignored(self) -> None:
        x, y = make_chunks(["x", "y"])
        big = reciprocal_rank_fusion([[ScoredChunk(x, 1e9), ScoredChunk(y, -1e9)]])
        small = reciprocal_rank_fusion([[ScoredChunk(x, 0.2), ScoredChunk(y, 0.1)]])
        assert [s.score for s in big] == [s.score for s in small]


class TestDense:
    def test_nearest_neighbour_and_k(self) -> None:
        chunks = make_chunks(
            ["oauth2 scopes and tokens", "path parameters in routes", "background tasks run later"]
        )
        r = DenseRetriever(FakeEmbedder(dim=128))
        r.index(chunks)
        results = r.search("how do oauth2 scopes work", k=2)
        assert len(results) == 2
        assert results[0].chunk == chunks[0]
        assert results[0].score >= results[1].score

    def test_reindex_replaces_previous_index(self) -> None:
        r = DenseRetriever(FakeEmbedder())
        r.index(make_chunks(["old content"], doc_id="old.md"))
        r.index(make_chunks(["new content"], doc_id="new.md"))
        assert [x.chunk.doc_id for x in r.search("content", k=5)] == ["new.md"]

    def test_empty_index(self) -> None:
        r = DenseRetriever(FakeEmbedder())
        r.index([])
        assert r.search("anything", k=3) == []


class _StubRetriever:
    def __init__(self, ranking: list[Chunk]) -> None:
        self.ranking = ranking
        self.indexed: list[Chunk] = []
        self.requested_k: int | None = None

    def index(self, chunks: Sequence[Chunk]) -> None:
        self.indexed = list(chunks)

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        self.requested_k = k
        return [ScoredChunk(c, 1.0) for c in self.ranking[:k]]


def test_hybrid_fuses_and_truncates() -> None:
    a, b, c = make_chunks(["a", "b", "c"])
    s1, s2 = _StubRetriever([a, b]), _StubRetriever([b, c])
    hybrid = HybridRetriever([s1, s2], rrf_k=60)
    hybrid.index([a, b, c])
    assert s1.indexed == s2.indexed == [a, b, c]
    results = hybrid.search("q", k=2)
    assert [r.chunk for r in results] == [b, a]  # b appears in both lists
    assert s1.requested_k == 2
