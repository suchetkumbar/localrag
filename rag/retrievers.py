"""Retrievers: BM25 (lexical), dense (Qdrant), and hybrid (reciprocal rank fusion)."""

import math
import re
import uuid
from collections import Counter, defaultdict
from collections.abc import Sequence

from rag.interfaces import Embedder, Retriever
from rag.types import Chunk, ScoredChunk

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    r"""Lowercase ``\w+`` tokens. ``\w`` keeps underscores, so identifiers such as
    ``response_model`` stay one token, which matters for API documentation."""
    return _TOKEN.findall(text.lower())


def _ranked(scores: dict[int, float], chunks: Sequence[Chunk], k: int) -> list[ScoredChunk]:
    """Top-k by score; ties broken by chunk_id so rankings are fully deterministic."""
    order = sorted(scores, key=lambda i: (-scores[i], chunks[i].chunk_id))
    return [ScoredChunk(chunks[i], scores[i]) for i in order[:k]]


class BM25Retriever:
    """Okapi BM25, implemented directly so every number can be explained.

        score(q, d) = sum over unique query terms t of
            idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))

        idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

    The "1 +" inside the log (the Lucene variant) keeps idf positive even for terms in
    more than half the documents; the classic formula goes negative there and would
    *penalise* documents for matching a query word.

    k1 controls term-frequency saturation (the 10th "oauth" adds much less than the 1st);
    b controls length normalisation (b=1: full, b=0: none).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._doc_len: list[int] = []
        self._avgdl = 0.0
        self._idf: dict[str, float] = {}

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._doc_len = []
        for i, chunk in enumerate(self._chunks):
            tokens = tokenize(chunk.indexed_text)
            self._doc_len.append(len(tokens))
            for term, tf in Counter(tokens).items():
                postings[term].append((i, tf))
        self._postings = dict(postings)
        n = len(self._chunks)
        self._avgdl = sum(self._doc_len) / n if n else 0.0
        self._idf = {
            term: math.log(1 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
            for term, plist in self._postings.items()
        }

    def scores(self, query: str) -> dict[int, float]:
        """Score for every chunk containing at least one query term."""
        acc: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            plist = self._postings.get(term)
            if not plist:
                continue
            idf = self._idf[term]
            for i, tf in plist:
                norm = 1 - self.b + self.b * self._doc_len[i] / self._avgdl
                acc[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return dict(acc)

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        return _ranked(self.scores(query), self._chunks, k)


# A fixed namespace so a chunk_id always maps to the same Qdrant point UUID.
_POINT_NAMESPACE = uuid.UUID("6f1c1d1e-2b9a-4c55-9a52-6b1f0f6d3c11")


class DenseRetriever:
    """Cosine similarity search in Qdrant.

    Qdrant runs *embedded* (``location=":memory:"`` or a directory path): no server
    process, so tests and CI need nothing extra. The same client code works against a
    Qdrant server by passing a URL instead.
    """

    def __init__(
        self, embedder: Embedder, location: str = ":memory:", collection: str = "chunks"
    ) -> None:
        from qdrant_client import QdrantClient

        self.embedder = embedder
        self.collection = collection
        if location.startswith(("http://", "https://")):
            self._client = QdrantClient(url=location)
        elif location == ":memory:":
            self._client = QdrantClient(location=":memory:")
        else:
            self._client = QdrantClient(path=location)
        self._chunks: dict[str, Chunk] = {}

    def index(self, chunks: Sequence[Chunk]) -> None:
        from qdrant_client.models import Distance, PointStruct, VectorParams

        self._chunks = {c.chunk_id: c for c in chunks}
        if self._client.collection_exists(self.collection):
            self._client.delete_collection(self.collection)
        if not chunks:
            return
        vectors = self.embedder.embed_documents([c.indexed_text for c in chunks])
        self._client.create_collection(
            self.collection,
            vectors_config=VectorParams(size=int(vectors.shape[1]), distance=Distance.COSINE),
        )
        batch = 256
        for i in range(0, len(chunks), batch):
            self._client.upsert(
                self.collection,
                points=[
                    PointStruct(
                        id=str(uuid.uuid5(_POINT_NAMESPACE, c.chunk_id)),
                        vector=vectors[i + j].tolist(),
                        payload={"chunk_id": c.chunk_id},
                    )
                    for j, c in enumerate(chunks[i : i + batch])
                ],
            )

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        if not self._chunks:
            return []
        vector = self.embedder.embed_query(query)[0].tolist()
        points = self._client.query_points(self.collection, query=vector, limit=k).points
        results = [
            ScoredChunk(self._chunks[str(p.payload["chunk_id"])], float(p.score))
            for p in points
            if p.payload is not None
        ]
        return sorted(results, key=lambda r: (-r.score, r.chunk.chunk_id))


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredChunk]], k: int = 60
) -> list[ScoredChunk]:
    """Fuse ranked lists: ``rrf(d) = sum over lists L containing d of 1 / (k + rank_L(d))``.

    Ranks are 1-based. RRF uses only *ranks*, never raw scores, so it needs no score
    normalisation between BM25 (unbounded) and cosine similarity ([-1, 1]). k=60 is the
    value from Cormack et al. (2009); larger k flattens the advantage of top ranks.
    """
    fused: dict[str, float] = defaultdict(float)
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, sc in enumerate(ranking, start=1):
            fused[sc.chunk.chunk_id] += 1.0 / (k + rank)
            chunks[sc.chunk.chunk_id] = sc.chunk
    order = sorted(fused, key=lambda cid: (-fused[cid], cid))
    return [ScoredChunk(chunks[cid], fused[cid]) for cid in order]


class HybridRetriever:
    """Runs each sub-retriever to depth ``k`` and fuses the lists with RRF."""

    def __init__(self, retrievers: Sequence[Retriever], rrf_k: int = 60) -> None:
        if not retrievers:
            raise ValueError("HybridRetriever needs at least one retriever")
        self.retrievers = list(retrievers)
        self.rrf_k = rrf_k

    def index(self, chunks: Sequence[Chunk]) -> None:
        for r in self.retrievers:
            r.index(chunks)

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        rankings = [r.search(query, k) for r in self.retrievers]
        return reciprocal_rank_fusion(rankings, self.rrf_k)[:k]
