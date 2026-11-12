"""Retrieval metrics computed from evidence *spans*, so they work for any chunker.

Relevance
---------
A retrieved chunk *covers* an evidence span when they are in the same document and

    overlap >= min_fraction * len(span)     (the chunk contains most of the evidence), or
    overlap >= min_fraction * len(chunk)    (the chunk lies mostly inside the evidence).

The second condition matters when evidence is longer than a chunk: a small chunk that
sits entirely inside a long evidence passage is relevant even though it holds only part
of it. ``min_fraction`` defaults to 0.5.

Metrics (per question; the run reports the mean over answerable questions)
--------------------------------------------------------------------------
Let a question have n evidence spans and let ``covered[i]`` be the set of spans covered
by the chunk at rank i+1.

* recall@k  = |union of covered[0..k-1]| / n
  The fraction of the evidence found in the top k. A multi-hop question with two spans
  scores 0.5 if only one is retrieved.
* MRR       = 1 / (rank of the first chunk covering any span), or 0 if none does.
* nDCG@k    = DCG@k / IDCG@k with binary, *novelty-based* gains: a chunk earns gain 1
  only if it covers a span not covered by a higher-ranked chunk. So a second chunk
  covering the same evidence (for example an overlapping window) earns nothing.
      DCG@k  = sum over ranks r <= k of gain_r / log2(r + 1)
      IDCG@k = sum over r = 1..min(n, k) of 1 / log2(r + 1)   (every span found, in order)
"""

import math
from collections.abc import Sequence

from rag.golden.schema import Evidence
from rag.types import Chunk


def overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def covers(chunk: Chunk, span: Evidence, min_fraction: float = 0.5) -> bool:
    if chunk.doc_id != span.doc_id:
        return False
    ov = overlap(chunk.start, chunk.end, span.start, span.end)
    if ov == 0:
        return False
    return ov >= min_fraction * (span.end - span.start) or ov >= min_fraction * (
        chunk.end - chunk.start
    )


def covered_spans(
    ranking: Sequence[Chunk], evidence: Sequence[Evidence], min_fraction: float = 0.5
) -> list[frozenset[int]]:
    """For each ranked chunk, the indices of the evidence spans it covers."""
    return [
        frozenset(i for i, span in enumerate(evidence) if covers(chunk, span, min_fraction))
        for chunk in ranking
    ]


def recall_at_k(covered: Sequence[frozenset[int]], n_spans: int, k: int) -> float:
    if n_spans == 0:
        raise ValueError("recall is undefined without evidence")
    found: set[int] = set()
    for spans in covered[:k]:
        found |= spans
    return len(found) / n_spans


def reciprocal_rank(covered: Sequence[frozenset[int]]) -> float:
    for rank, spans in enumerate(covered, start=1):
        if spans:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(covered: Sequence[frozenset[int]], n_spans: int, k: int) -> float:
    if n_spans == 0:
        raise ValueError("nDCG is undefined without evidence")
    seen: set[int] = set()
    dcg = 0.0
    for rank, spans in enumerate(covered[:k], start=1):
        new = spans - seen
        if new:
            dcg += 1.0 / math.log2(rank + 1)
            seen |= new
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(n_spans, k) + 1))
    return dcg / idcg


def retrieval_metrics(
    ranking: Sequence[Chunk],
    evidence: Sequence[Evidence],
    ks: Sequence[int] = (1, 3, 5, 10),
    ndcg_k: int = 10,
    min_fraction: float = 0.5,
) -> dict[str, float]:
    covered = covered_spans(ranking, evidence, min_fraction)
    n = len(evidence)
    metrics = {f"recall@{k}": recall_at_k(covered, n, k) for k in ks}
    metrics["mrr"] = reciprocal_rank(covered)
    metrics[f"ndcg@{ndcg_k}"] = ndcg_at_k(covered, n, ndcg_k)
    return metrics
