from pathlib import Path

import pytest
from pydantic import ValidationError

from rag.config import (
    FixedChunkerConfig,
    HeadingChunkerConfig,
    HybridRetrieverConfig,
    PipelineConfig,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_baseline_config_loads() -> None:
    cfg = load_config(ROOT / "configs" / "baseline.yaml")
    assert cfg.name == "baseline"
    assert isinstance(cfg.chunker, FixedChunkerConfig)
    assert cfg.retrieval.retriever.kind == "dense"
    assert cfg.reranker is None
    assert cfg.generator.kind == "ollama"


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ValidationError, match="chunk_szie"):
        PipelineConfig.model_validate({"name": "x", "chunker": {"kind": "fixed", "chunk_szie": 10}})


def test_discriminator_selects_implementation() -> None:
    cfg = PipelineConfig.model_validate(
        {
            "name": "x",
            "chunker": {"kind": "heading"},
            "retrieval": {"retriever": {"kind": "hybrid"}},
        }
    )
    assert isinstance(cfg.chunker, HeadingChunkerConfig)
    assert isinstance(cfg.retrieval.retriever, HybridRetrieverConfig)
    assert cfg.retrieval.retriever.rrf_k == 60


def test_unknown_kind_is_rejected() -> None:
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate({"name": "x", "chunker": {"kind": "magic"}})


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValidationError, match="overlap"):
        FixedChunkerConfig(size=100, overlap=100)


def test_top_k_cannot_exceed_candidate_k() -> None:
    with pytest.raises(ValidationError, match="top_k"):
        PipelineConfig.model_validate({"name": "x", "retrieval": {"candidate_k": 3, "top_k": 5}})


def test_fingerprint_ignores_name_and_cache_but_not_settings() -> None:
    a = PipelineConfig.model_validate({"name": "a"})
    b = PipelineConfig.model_validate({"name": "b", "cache": {"mode": "readonly"}})
    c = PipelineConfig.model_validate({"name": "a", "retrieval": {"top_k": 3}})
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()


def test_load_config_rejects_non_mapping(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(p)
