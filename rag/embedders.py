"""Embedders: Ollama (real), a deterministic fake (tests/CI), and a caching wrapper."""

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

import httpx
import numpy as np

from rag.cache import ARRAY_CODEC, CacheMissError, DiskCache, make_key
from rag.http import raise_for_status
from rag.interfaces import Embedder, Vectors

_TOKEN = re.compile(r"\w+")


def _l2_normalise(x: Vectors) -> Vectors:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalised: Vectors = (x / norms).astype(np.float32)
    return normalised


class OllamaEmbedder:
    """Calls Ollama's ``/api/embed`` endpoint directly with httpx.

    No LangChain wrapper: one HTTP call is easier to read, test (httpx.MockTransport)
    and debug than a framework abstraction, and it exposes batching explicitly.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        query_prefix: str = "",
        document_prefix: str = "",
        batch_size: int = 32,
        timeout_s: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.batch_size = batch_size
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout_s)

    @property
    def identity(self) -> Mapping[str, Any]:
        return {
            "provider": "ollama",
            "model": self.model,
            "query_prefix": self.query_prefix,
            "document_prefix": self.document_prefix,
        }

    def _embed(self, texts: Sequence[str]) -> Vectors:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            resp = self._client.post("/api/embed", json={"model": self.model, "input": batch})
            raise_for_status(resp)
            vectors = resp.json()["embeddings"]
            if len(vectors) != len(batch):
                raise RuntimeError(f"Ollama returned {len(vectors)} vectors for {len(batch)} texts")
            out.extend(vectors)
        return np.asarray(out, dtype=np.float32).reshape(len(texts), -1)

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        return self._embed([self.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> Vectors:
        return self._embed([self.query_prefix + text])


class FakeEmbedder:
    """Feature-hashing bag of words, L2-normalised. Deterministic and dependency-free.

    Similar texts (shared words) get high cosine similarity, which is enough to test that
    dense retrieval and fusion are wired correctly. It is *not* a semantic model.
    Uses blake2b rather than ``hash()``, which is salted per Python process.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    @property
    def identity(self) -> Mapping[str, Any]:
        return {"provider": "fake", "dim": self.dim}

    def _vector(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for token in _TOKEN.findall(text.lower()):
            h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        return v

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        arr = np.asarray([self._vector(t) for t in texts], dtype=np.float32)
        return _l2_normalise(arr.reshape(len(texts), self.dim))

    def embed_query(self, text: str) -> Vectors:
        return self.embed_documents([text])


class CachedEmbedder:
    """Wraps any Embedder; caches each text's vector separately.

    Per-text (not per-batch) keys mean that re-chunking a corpus only embeds the chunks
    that actually changed. Missing texts are still embedded in one batched call.
    """

    def __init__(self, inner: Embedder, cache: DiskCache) -> None:
        self.inner = inner
        self.cache = cache

    @property
    def identity(self) -> Mapping[str, Any]:
        return self.inner.identity

    def _key(self, role: str, text: str) -> str:
        return make_key(
            "embedding", {"embedder": dict(self.inner.identity), "role": role, "text": text}
        )

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        if self.cache.mode == "off":
            return self.inner.embed_documents(texts)
        keys = [self._key("document", t) for t in texts]
        found: dict[int, Vectors] = {}
        missing: list[int] = []
        for i, key in enumerate(keys):
            data = self.cache.get(key)
            if data is None:
                missing.append(i)
            else:
                found[i] = ARRAY_CODEC.decode(data)
        self.cache.hits += len(found)
        self.cache.misses += len(missing)
        if missing:
            if self.cache.mode == "readonly":
                raise CacheMissError(
                    f"{len(missing)} document embeddings missing in readonly cache"
                )
            computed = self.inner.embed_documents([texts[i] for i in missing])
            for row, i in enumerate(missing):
                vec = computed[row]
                found[i] = vec
                self.cache.put(keys[i], ARRAY_CODEC.encode(vec))
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([found[i] for i in range(len(texts))]).astype(np.float32)

    def embed_query(self, text: str) -> Vectors:
        value, _ = self.cache.get_or_compute(
            "embedding",
            {"embedder": dict(self.inner.identity), "role": "query", "text": text},
            lambda: self.inner.embed_query(text),
            ARRAY_CODEC,
        )
        return value
