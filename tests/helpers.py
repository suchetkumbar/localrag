"""Test helpers shared across test modules (conftest.py is for fixtures only)."""

from rag.golden.schema import Evidence, GoldenItem, Provenance, Review
from rag.types import Chunk, make_chunk_id


def make_chunks(texts: list[str], doc_id: str = "doc.md") -> list[Chunk]:
    """Chunks laid out back to back in a fictional document."""
    chunks: list[Chunk] = []
    pos = 0
    for text in texts:
        chunks.append(
            Chunk(make_chunk_id(doc_id, pos, pos + len(text)), doc_id, text, pos, pos + len(text))
        )
        pos += len(text) + 1
    return chunks


def golden_item(item_id: str, doc: str, quote: str, verified: bool = True) -> GoldenItem:
    """A factual item whose evidence is ``quote`` inside ``doc`` (doc_id "ports.md")."""
    start = doc.index(quote)
    review = Review(status="accepted") if verified else Review()
    return GoldenItem(
        id=item_id,
        question=f"Question about {quote[:20]}?",
        reference_answer=quote,
        question_type="factual",
        evidence=(Evidence(doc_id="ports.md", start=start, end=start + len(quote), quote=quote),),
        human_verified=verified,
        review=review,
        provenance=Provenance(drafted_by="t", draft_prompt="t"),
        corpus_fingerprint="fp",
    )
