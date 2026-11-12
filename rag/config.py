"""Experiment configuration, loaded from YAML.

Every model is ``extra="forbid"``: a typo such as ``chunk_szie`` fails at load time
instead of silently falling back to a default and producing a misleading experiment.
Component choices are *discriminated unions* keyed on ``kind``, so each implementation
declares exactly the parameters it accepts.
"""

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- corpus


class CorpusConfig(StrictModel):
    path: Path = Path("data/corpus/fastapi")
    patterns: tuple[str, ...] = ("**/*.md",)


# --------------------------------------------------------------------------- chunkers


class FixedChunkerConfig(StrictModel):
    """Fixed-size character windows with overlap, snapped to word boundaries."""

    kind: Literal["fixed"] = "fixed"
    size: int = Field(default=1000, gt=0)
    overlap: int = Field(default=150, ge=0)

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> Self:
        if self.overlap >= self.size:
            raise ValueError("overlap must be smaller than size")
        return self


class HeadingChunkerConfig(StrictModel):
    """One chunk per markdown section; oversized sections fall back to fixed windows."""

    kind: Literal["heading"] = "heading"
    max_heading_level: int = Field(default=3, ge=1, le=6)
    max_size: int = Field(default=1500, gt=0)
    overlap: int = Field(default=150, ge=0)

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> Self:
        if self.overlap >= self.max_size:
            raise ValueError("overlap must be smaller than max_size")
        return self


ChunkerConfig = Annotated[FixedChunkerConfig | HeadingChunkerConfig, Field(discriminator="kind")]


# --------------------------------------------------------------------------- embedders


class OllamaEmbedderConfig(StrictModel):
    kind: Literal["ollama"] = "ollama"
    model: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"
    # nomic-embed-text was trained with task prefixes; omitting them measurably hurts
    # retrieval. Other models (e.g. mxbai) use different or no prefixes.
    query_prefix: str = "search_query: "
    document_prefix: str = "search_document: "
    batch_size: int = Field(default=32, gt=0)
    timeout_s: float = 120.0


class FakeEmbedderConfig(StrictModel):
    """Deterministic hashing embedder for tests and offline CI. Not a real model."""

    kind: Literal["fake"] = "fake"
    dim: int = Field(default=256, gt=0)


EmbedderConfig = Annotated[OllamaEmbedderConfig | FakeEmbedderConfig, Field(discriminator="kind")]


# --------------------------------------------------------------------------- retrievers


class DenseParams(StrictModel):
    location: str = ":memory:"  # or a directory path for an on-disk Qdrant collection
    collection: str = "chunks"


class BM25Params(StrictModel):
    k1: float = Field(default=1.5, ge=0)
    b: float = Field(default=0.75, ge=0, le=1)


class DenseRetrieverConfig(DenseParams):
    kind: Literal["dense"] = "dense"


class BM25RetrieverConfig(BM25Params):
    kind: Literal["bm25"] = "bm25"


class HybridRetrieverConfig(StrictModel):
    """Dense + BM25 fused with reciprocal rank fusion."""

    kind: Literal["hybrid"] = "hybrid"
    rrf_k: int = Field(default=60, gt=0)
    dense: DenseParams = DenseParams()
    bm25: BM25Params = BM25Params()


RetrieverConfig = Annotated[
    DenseRetrieverConfig | BM25RetrieverConfig | HybridRetrieverConfig,
    Field(discriminator="kind"),
]


class RetrievalConfig(StrictModel):
    retriever: RetrieverConfig = DenseRetrieverConfig()
    # First-stage depth. This is also the depth of the ranking that retrieval metrics
    # see, so it must be >= the largest k reported (recall@10).
    candidate_k: int = Field(default=50, gt=0)
    # How many chunks are given to the generator as context.
    top_k: int = Field(default=5, gt=0)

    @model_validator(mode="after")
    def _top_k_within_candidates(self) -> Self:
        if self.top_k > self.candidate_k:
            raise ValueError("top_k must be <= candidate_k")
        return self


# --------------------------------------------------------------------------- reranker


class CrossEncoderRerankerConfig(StrictModel):
    kind: Literal["cross_encoder"] = "cross_encoder"
    model: str = "BAAI/bge-reranker-base"
    batch_size: int = Field(default=16, gt=0)


# --------------------------------------------------------------------------- generators


class SamplingParams(StrictModel):
    temperature: float = Field(default=0.0, ge=0)
    seed: int = 0
    max_tokens: int = Field(default=512, gt=0)


class OllamaGeneratorConfig(StrictModel):
    kind: Literal["ollama"] = "ollama"
    model: str = "llama3"
    base_url: str = "http://localhost:11434"
    # Ollama silently truncates prompts longer than num_ctx, which would drop retrieved
    # context without any error, so it is set explicitly. 4096 covers the worst case
    # (8000 context chars ~ 2000 tokens + system prompt + 512 answer tokens) and still
    # fits llama3-8B plus the embedder on a 6 GB GPU; 8192 ran out of VRAM there.
    num_ctx: int = Field(default=4096, gt=0)
    timeout_s: float = 300.0
    sampling: SamplingParams = SamplingParams()


class OpenAICompatibleGeneratorConfig(StrictModel):
    """Any OpenAI-compatible /chat/completions endpoint (Groq, OpenRouter, Gemini, ...)."""

    kind: Literal["openai_compatible"] = "openai_compatible"
    base_url: str = "https://api.groq.com/openai/v1"
    model: str = "openai/gpt-oss-120b"
    api_key_env: str = "GROQ_API_KEY"  # name of the env var, never the key itself
    timeout_s: float = 120.0
    sampling: SamplingParams = SamplingParams()


class FakeGeneratorConfig(StrictModel):
    kind: Literal["fake"] = "fake"
    response: str = "This is a fake answer [1]."


GeneratorConfig = Annotated[
    OllamaGeneratorConfig | OpenAICompatibleGeneratorConfig | FakeGeneratorConfig,
    Field(discriminator="kind"),
]


class PromptConfig(StrictModel):
    name: str = "answer_v1"  # resolves to rag/prompts/<name>.system.txt and .user.txt
    max_context_chars: int = Field(default=8000, gt=0)


# --------------------------------------------------------------------------- cache


CacheMode = Literal["rw", "readonly", "off"]


class CacheConfig(StrictModel):
    path: Path = Path(".cache/rag.sqlite")
    # rw: read, compute on miss, write. readonly: a miss is an error (used in CI so a
    # run can never silently call a model). off: always compute, store nothing.
    mode: CacheMode = "rw"


# --------------------------------------------------------------------------- root


class PipelineConfig(StrictModel):
    name: str
    corpus: CorpusConfig = CorpusConfig()
    chunker: ChunkerConfig = FixedChunkerConfig()
    embedder: EmbedderConfig = OllamaEmbedderConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    reranker: CrossEncoderRerankerConfig | None = None
    generator: GeneratorConfig = OllamaGeneratorConfig()
    prompt: PromptConfig = PromptConfig()
    cache: CacheConfig = CacheConfig()

    def fingerprint(self) -> str:
        """Short hash of everything that affects results (cache settings excluded)."""
        payload = self.model_dump(mode="json", exclude={"name", "cache"})
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def load_config(path: Path | str) -> PipelineConfig:
    with Path(path).open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return PipelineConfig.model_validate(raw)
