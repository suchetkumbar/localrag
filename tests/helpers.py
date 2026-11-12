"""Test helpers shared across test modules (conftest.py is for fixtures only)."""

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
