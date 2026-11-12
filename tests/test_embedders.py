import json
from collections.abc import Mapping, Sequence
from typing import Any

import httpx
import numpy as np
import pytest

from rag.cache import CacheMissError, DiskCache
from rag.embedders import CachedEmbedder, FakeEmbedder, OllamaEmbedder
from rag.interfaces import Embedder, Vectors


def test_fake_embedder_is_deterministic_and_normalised() -> None:
    e = FakeEmbedder(dim=64)
    a = e.embed_documents(["hello world", "goodbye"])
    b = FakeEmbedder(dim=64).embed_documents(["hello world", "goodbye"])
    assert a.shape == (2, 64)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, rtol=1e-6)


def test_fake_embedder_similar_texts_are_closer() -> None:
    e = FakeEmbedder(dim=256)
    q = e.embed_query("oauth2 scopes")[0]
    near, far = e.embed_documents(["oauth2 scopes explained", "background tasks"])
    assert q @ near > q @ far


def test_satisfies_protocol() -> None:
    assert isinstance(FakeEmbedder(), Embedder)


def _ollama_with(handler: list[dict[str, Any]]) -> OllamaEmbedder:
    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        handler.append(body)
        return httpx.Response(
            200, json={"embeddings": [[float(len(t)), 1.0] for t in body["input"]]}
        )

    client = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(respond))
    return OllamaEmbedder(
        model="m", query_prefix="q: ", document_prefix="d: ", batch_size=2, client=client
    )


def test_ollama_embedder_applies_prefixes_and_batches() -> None:
    requests: list[dict[str, Any]] = []
    e = _ollama_with(requests)
    vecs = e.embed_documents(["a", "bb", "ccc"])
    assert [r["input"] for r in requests] == [["d: a", "d: bb"], ["d: ccc"]]
    assert all(r["model"] == "m" for r in requests)
    assert vecs.dtype == np.float32
    assert vecs[:, 0].tolist() == [4.0, 5.0, 6.0]
    e.embed_query("x")
    assert requests[-1]["input"] == ["q: x"]


class _CountingEmbedder:
    def __init__(self) -> None:
        self.inner = FakeEmbedder(dim=16)
        self.document_calls: list[list[str]] = []

    @property
    def identity(self) -> Mapping[str, Any]:
        return self.inner.identity

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        self.document_calls.append(list(texts))
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> Vectors:
        return self.inner.embed_query(text)


def test_cached_embedder_only_embeds_misses(cache: DiskCache) -> None:
    inner = _CountingEmbedder()
    e = CachedEmbedder(inner, cache)
    first = e.embed_documents(["a", "b"])
    second = e.embed_documents(["b", "c", "a"])
    assert inner.document_calls == [["a", "b"], ["c"]]
    np.testing.assert_array_equal(second[0], first[1])
    np.testing.assert_array_equal(second[2], first[0])


def test_cached_embedder_query_and_document_keys_differ(cache: DiskCache) -> None:
    e = CachedEmbedder(_CountingEmbedder(), cache)
    e.embed_documents(["same"])
    before = cache.misses
    e.embed_query("same")
    assert cache.misses == before + 1


def test_cached_embedder_readonly_miss(tmp_path: Any) -> None:
    ro = DiskCache(tmp_path / "c.sqlite", mode="readonly")
    with pytest.raises(CacheMissError):
        CachedEmbedder(FakeEmbedder(), ro).embed_documents(["x"])
    ro.close()
