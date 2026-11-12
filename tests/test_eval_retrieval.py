import math

import pytest

from rag.eval.retrieval import (
    covered_spans,
    covers,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    retrieval_metrics,
)
from rag.golden.schema import Evidence
from rag.types import Chunk


def chunk(start: int, end: int, doc: str = "d.md") -> Chunk:
    return Chunk(f"{doc}#{start}-{end}", doc, "x" * (end - start), start, end)


def span(start: int, end: int, doc: str = "d.md") -> Evidence:
    return Evidence(doc_id=doc, start=start, end=end, quote="x" * (end - start))


class TestCovers:
    def test_chunk_containing_span(self) -> None:
        assert covers(chunk(0, 1000), span(100, 200))

    def test_partial_overlap_threshold(self) -> None:
        ev = span(100, 200)  # 100 chars
        assert covers(chunk(150, 1000), ev)  # 50 of 100 span chars: exactly 0.5
        assert not covers(chunk(151, 1000), ev)  # 49 chars, and 49/849 of the chunk
        assert covers(chunk(151, 1000), ev, min_fraction=0.4)

    def test_small_chunk_inside_long_evidence(self) -> None:
        # Evidence longer than the chunk: covered because the chunk is mostly evidence.
        assert covers(chunk(500, 600), span(0, 2000))

    def test_other_document_or_no_overlap(self) -> None:
        assert not covers(chunk(0, 1000, doc="other.md"), span(100, 200))
        assert not covers(chunk(200, 300), span(100, 200))  # touching, not overlapping


# covered[i] = span indices covered by the chunk at rank i+1
COVERED = [frozenset(), frozenset({0}), frozenset({0}), frozenset(), frozenset({1})]


def test_recall_at_k() -> None:
    assert recall_at_k(COVERED, 2, 1) == 0.0
    assert recall_at_k(COVERED, 2, 2) == 0.5
    assert recall_at_k(COVERED, 2, 4) == 0.5  # rank 3 covers span 0 again: no gain
    assert recall_at_k(COVERED, 2, 5) == 1.0
    assert recall_at_k(COVERED, 2, 100) == 1.0  # k beyond the ranking


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(COVERED) == 0.5
    assert reciprocal_rank([frozenset({0})]) == 1.0
    assert reciprocal_rank([frozenset(), frozenset()]) == 0.0


def test_ndcg_matches_hand_computation() -> None:
    # Gains: rank 2 (span 0, new), rank 3 (span 0 again: 0), rank 5 (span 1, new).
    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(COVERED, 2, 10) == pytest.approx(dcg / idcg)
    assert ndcg_at_k([frozenset({0}), frozenset({1})], 2, 10) == pytest.approx(1.0)
    assert ndcg_at_k([frozenset()] * 3, 2, 10) == 0.0


def test_ndcg_with_more_spans_than_k() -> None:
    covered = [frozenset({i}) for i in range(3)]
    assert ndcg_at_k(covered, 5, 3) == pytest.approx(1.0)  # IDCG capped at k


def test_metrics_undefined_without_evidence() -> None:
    with pytest.raises(ValueError):
        recall_at_k([], 0, 5)
    with pytest.raises(ValueError):
        ndcg_at_k([], 0, 10)


def test_retrieval_metrics_end_to_end() -> None:
    evidence = [span(100, 200), span(5000, 5100)]
    ranking = [chunk(2000, 3000), chunk(0, 1000), chunk(4900, 5900)]
    covered = covered_spans(ranking, evidence)
    assert covered == [frozenset(), frozenset({0}), frozenset({1})]
    m = retrieval_metrics(ranking, evidence)
    assert m["recall@1"] == 0.0
    assert m["recall@3"] == 1.0
    assert m["mrr"] == 0.5
    assert set(m) == {"recall@1", "recall@3", "recall@5", "recall@10", "mrr", "ndcg@10"}
