"""Build pipeline components from a validated PipelineConfig.

This is the only module that maps config ``kind`` values to classes. Adding an
implementation means: a config model in config.py, a class, and one branch here.
Plain ``match`` statements are used instead of a plugin registry: there are only a
handful of implementations, and explicit code is easier to follow than decorators.
"""

from rag.cache import DiskCache
from rag.chunkers import FixedSizeChunker, HeadingChunker
from rag.config import (
    BM25RetrieverConfig,
    ChunkerConfig,
    DenseRetrieverConfig,
    EmbedderConfig,
    FakeEmbedderConfig,
    FakeGeneratorConfig,
    FixedChunkerConfig,
    GeneratorConfig,
    HeadingChunkerConfig,
    HybridRetrieverConfig,
    OllamaEmbedderConfig,
    OllamaGeneratorConfig,
    OpenAICompatibleGeneratorConfig,
    PipelineConfig,
    RetrieverConfig,
)
from rag.embedders import CachedEmbedder, FakeEmbedder, OllamaEmbedder
from rag.generators import (
    CachedGenerator,
    FakeGenerator,
    OllamaGenerator,
    OpenAICompatibleGenerator,
)
from rag.interfaces import Chunker, Embedder, Generator, Reranker, Retriever
from rag.pipeline import RAGPipeline
from rag.prompts import load_prompt
from rag.rerankers import CachedReranker, CrossEncoderReranker
from rag.retrievers import BM25Retriever, DenseRetriever, HybridRetriever


def build_chunker(cfg: ChunkerConfig) -> Chunker:
    match cfg:
        case FixedChunkerConfig():
            return FixedSizeChunker(size=cfg.size, overlap=cfg.overlap)
        case HeadingChunkerConfig():
            return HeadingChunker(
                max_heading_level=cfg.max_heading_level, max_size=cfg.max_size, overlap=cfg.overlap
            )


def build_embedder(cfg: EmbedderConfig, cache: DiskCache) -> Embedder:
    inner: Embedder
    match cfg:
        case OllamaEmbedderConfig():
            inner = OllamaEmbedder(
                model=cfg.model,
                base_url=cfg.base_url,
                query_prefix=cfg.query_prefix,
                document_prefix=cfg.document_prefix,
                batch_size=cfg.batch_size,
                timeout_s=cfg.timeout_s,
            )
        case FakeEmbedderConfig():
            inner = FakeEmbedder(dim=cfg.dim)
    return CachedEmbedder(inner, cache)


def build_retriever(cfg: RetrieverConfig, embedder: Embedder) -> Retriever:
    match cfg:
        case DenseRetrieverConfig():
            return DenseRetriever(embedder, location=cfg.location, collection=cfg.collection)
        case BM25RetrieverConfig():
            return BM25Retriever(k1=cfg.k1, b=cfg.b)
        case HybridRetrieverConfig():
            dense = DenseRetriever(
                embedder, location=cfg.dense.location, collection=cfg.dense.collection
            )
            bm25 = BM25Retriever(k1=cfg.bm25.k1, b=cfg.bm25.b)
            return HybridRetriever([dense, bm25], rrf_k=cfg.rrf_k)


def build_generator(cfg: GeneratorConfig, cache: DiskCache) -> Generator:
    inner: Generator
    match cfg:
        case OllamaGeneratorConfig():
            inner = OllamaGenerator(
                model=cfg.model,
                base_url=cfg.base_url,
                num_ctx=cfg.num_ctx,
                sampling=cfg.sampling,
                timeout_s=cfg.timeout_s,
            )
        case OpenAICompatibleGeneratorConfig():
            inner = OpenAICompatibleGenerator(
                base_url=cfg.base_url,
                model=cfg.model,
                api_key_env=cfg.api_key_env,
                sampling=cfg.sampling,
                timeout_s=cfg.timeout_s,
            )
        case FakeGeneratorConfig():
            inner = FakeGenerator(response=cfg.response)
    return CachedGenerator(inner, cache)


def build_pipeline(cfg: PipelineConfig, cache: DiskCache | None = None) -> RAGPipeline:
    cache = cache or DiskCache(cfg.cache.path, cfg.cache.mode)
    embedder = build_embedder(cfg.embedder, cache)
    reranker: Reranker | None = None
    if cfg.reranker is not None:
        reranker = CachedReranker(
            CrossEncoderReranker(model=cfg.reranker.model, batch_size=cfg.reranker.batch_size),
            cache,
        )
    return RAGPipeline(
        chunker=build_chunker(cfg.chunker),
        retriever=build_retriever(cfg.retrieval.retriever, embedder),
        generator=build_generator(cfg.generator, cache),
        prompt=load_prompt(cfg.prompt.name),
        candidate_k=cfg.retrieval.candidate_k,
        top_k=cfg.retrieval.top_k,
        max_context_chars=cfg.prompt.max_context_chars,
        reranker=reranker,
    )
