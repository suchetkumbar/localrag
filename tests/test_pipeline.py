from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from rag.cache import DiskCache
from rag.chunkers import HeadingChunker
from rag.config import PipelineConfig
from rag.embedders import FakeEmbedder
from rag.factory import build_pipeline
from rag.generators import CachedGenerator, FakeGenerator
from rag.pipeline import RAGPipeline
from rag.prompts import load_prompt
from rag.retrievers import BM25Retriever, DenseRetriever, HybridRetriever
from rag.types import Document


class _ReverseReranker:
    """Prefers chunks with *lower* first-stage rank, to prove reranking is applied."""

    @property
    def identity(self) -> Mapping[str, Any]:
        return {"provider": "reverse"}

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        return [float(i) for i in range(len(texts))]


def _pipeline(sample_doc: Document, **kwargs: Any) -> tuple[RAGPipeline, FakeGenerator]:
    gen = FakeGenerator("Scopes limit permissions [1].")
    retriever = HybridRetriever([DenseRetriever(FakeEmbedder()), BM25Retriever()])
    pipe = RAGPipeline(HeadingChunker(), retriever, gen, load_prompt("answer_v1"), **kwargs)
    pipe.index([sample_doc])
    return pipe, gen


def test_answer_end_to_end_with_fakes(sample_doc: Document) -> None:
    pipe, gen = _pipeline(sample_doc, candidate_k=4, top_k=2)
    answer = pipe.answer("What are OAuth2 scopes?")

    assert len(pipe.chunks) == 4
    assert answer.ranking[0].chunk.title == "Security > OAuth2 > Scopes"
    assert len(answer.ranking) <= 4
    assert [c.chunk for c in answer.context] == [c.chunk for c in answer.ranking[:2]]
    assert answer.generation.text == "Scopes limit permissions [1]."
    assert set(answer.timings_ms) == {"retrieve_ms", "generate_ms"}

    system, user = gen.calls[0]
    assert system == load_prompt("answer_v1").system
    assert "[1] (source: tutorial/security.md)\nSecurity > OAuth2 > Scopes" in user
    assert "[3]" not in user  # only top_k chunks are shown to the LLM
    assert user.endswith("Question: What are OAuth2 scopes?")


def test_reranker_reorders_ranking(sample_doc: Document) -> None:
    plain, _ = _pipeline(sample_doc, candidate_k=4, top_k=2)
    reranked, _ = _pipeline(sample_doc, candidate_k=4, top_k=2, reranker=_ReverseReranker())
    before = [s.chunk.chunk_id for s in plain.retrieve("OAuth2 scopes").ranking]
    result = reranked.retrieve("OAuth2 scopes")
    assert [s.chunk.chunk_id for s in result.ranking] == list(reversed(before))
    assert "rerank_ms" in result.timings_ms


def test_no_results_still_calls_generator(sample_doc: Document) -> None:
    gen = FakeGenerator()
    pipe = RAGPipeline(HeadingChunker(), BM25Retriever(), gen, load_prompt("answer_v1"))
    pipe.index([sample_doc])
    answer = pipe.answer("zzzz qqqq")
    assert answer.context == []
    assert "(no passages retrieved)" in gen.calls[0][1]


def test_top_k_must_not_exceed_candidate_k(sample_doc: Document) -> None:
    with pytest.raises(ValueError):
        _pipeline(sample_doc, candidate_k=2, top_k=3)


@pytest.mark.parametrize("retriever", ["dense", "bm25", "hybrid"])
@pytest.mark.parametrize("chunker", ["fixed", "heading"])
def test_build_pipeline_from_config(
    tmp_path: Path, sample_doc: Document, retriever: str, chunker: str
) -> None:
    cfg = PipelineConfig.model_validate(
        {
            "name": "test",
            "chunker": {"kind": chunker},
            "embedder": {"kind": "fake"},
            "retrieval": {"retriever": {"kind": retriever}, "candidate_k": 10, "top_k": 3},
            "generator": {"kind": "fake", "response": "ok"},
            "cache": {"path": str(tmp_path / "c.sqlite")},
        }
    )
    cache = DiskCache(cfg.cache.path, cfg.cache.mode)
    pipe = build_pipeline(cfg, cache)
    pipe.index([sample_doc])
    first = pipe.answer("OAuth2 scopes")
    second = pipe.answer("OAuth2 scopes")
    assert first.ranking and first.generation.text == "ok"
    assert isinstance(pipe.generator, CachedGenerator)
    assert (first.generation.cached, second.generation.cached) == (False, True)
    cache.close()


def test_build_pipeline_with_reranker_config_is_lazy(tmp_path: Path) -> None:
    cfg = PipelineConfig.model_validate(
        {
            "name": "test",
            "embedder": {"kind": "fake"},
            "reranker": {"kind": "cross_encoder"},
            "generator": {"kind": "fake"},
            "cache": {"mode": "off"},
        }
    )
    pipe = build_pipeline(cfg)
    assert pipe.reranker is not None
