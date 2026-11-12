"""Component interfaces.

These are ``typing.Protocol`` classes (structural typing): any object with the right
methods satisfies them, with no inheritance required. That keeps test fakes trivial and
means third-party objects can be adapted without wrapper class hierarchies.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from rag.types import Chunk, Document, Generation, ScoredChunk

Vectors = npt.NDArray[np.float32]
"""A 2-D float32 array of shape (n_texts, dim)."""


@runtime_checkable
class Chunker(Protocol):
    def chunk(self, doc: Document) -> list[Chunk]:
        """Split one document into chunks whose offsets satisfy the Chunk invariant."""
        ...


@runtime_checkable
class Embedder(Protocol):
    @property
    def identity(self) -> Mapping[str, Any]:
        """Everything that changes the output vectors (model, prefixes...). Used as cache key."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> Vectors: ...

    def embed_query(self, text: str) -> Vectors:
        """Returns shape (1, dim)."""
        ...


@runtime_checkable
class Retriever(Protocol):
    def index(self, chunks: Sequence[Chunk]) -> None:
        """Build the index. Calling it again replaces the previous index."""
        ...

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        """Return at most ``k`` results, best first. Ties are broken by chunk_id."""
        ...


@runtime_checkable
class Reranker(Protocol):
    @property
    def identity(self) -> Mapping[str, Any]: ...

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Relevance score per text, higher is better. Sorting lives in ``rerank()``."""
        ...


@runtime_checkable
class Generator(Protocol):
    @property
    def identity(self) -> Mapping[str, Any]:
        """Model and sampling params. Used as cache key together with the prompt."""
        ...

    def generate(self, system: str, user: str) -> Generation: ...
