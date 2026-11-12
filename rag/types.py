"""Plain data types shared by every pipeline component.

These are frozen dataclasses rather than pydantic models: they are created in hot loops
(thousands of chunks, tens of thousands of scored results) and never need validation
after construction.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Document:
    """A source document after loading.

    ``doc_id`` is the path relative to the corpus root (POSIX separators), so it is
    stable across machines and runs. Golden-dataset spans refer to offsets in ``text``.
    """

    doc_id: str
    text: str
    title: str = ""


@dataclass(frozen=True, slots=True)
class Chunk:
    """A contiguous span of a document.

    Invariant: ``text == document.text[start:end]``. Keeping exact character offsets is
    what lets the golden dataset label *spans* instead of chunk IDs, so the same labels
    work for every chunking strategy (see DESIGN.md, "Labelling spans, not chunks").

    ``title`` is optional context (e.g. the heading path "Security > OAuth2") that is
    prepended when the chunk is indexed or shown to the LLM, without breaking the
    offset invariant.
    """

    chunk_id: str
    doc_id: str
    text: str
    start: int
    end: int
    title: str = ""

    @property
    def indexed_text(self) -> str:
        """The text that embedders, BM25 and the LLM actually see."""
        return f"{self.title}\n\n{self.text}" if self.title else self.text


def make_chunk_id(doc_id: str, start: int, end: int) -> str:
    """Readable, deterministic chunk ID, e.g. ``tutorial/security.md#1200-1712``.

    Deterministic IDs are required for caching and for comparing two runs item by item;
    the previous implementation used random UUIDs, so IDs changed on every ingest.
    """
    return f"{doc_id}#{start}-{end}"


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A chunk with a retrieval or reranking score. Higher is always better."""

    chunk: Chunk
    score: float


@dataclass(frozen=True, slots=True)
class Generation:
    """One LLM completion plus the operational data the eval harness reports."""

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cached: bool = False
