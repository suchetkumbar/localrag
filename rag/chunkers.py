"""Chunking strategies.

Both chunkers only ever emit *spans* of the original text (start/end offsets), never
rewritten text. That guarantees ``chunk.text == doc.text[start:end]``, which the
golden dataset and the retrieval metrics rely on.

Sizes are measured in characters, not tokens. Characters are deterministic and need no
tokenizer dependency; for English prose one token is roughly 4 characters, so the
default 1000-character window is about 250 tokens.
"""

import re
from itertools import pairwise

from rag.types import Chunk, Document, make_chunk_id


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink [start, end) so it neither starts nor ends with whitespace."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def window_spans(text: str, start: int, end: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Split ``text[start:end]`` into windows of at most ``size`` chars.

    Window ends snap back to the last whitespace in the second half of the window so
    words are not cut in half; a window with no whitespace there is cut hard. Each next
    window starts ``overlap`` chars before the previous end, snapped forward to a word
    start. Returned spans are whitespace-trimmed and never empty.
    """
    spans: list[tuple[int, int]] = []
    pos = start
    while pos < end:
        win_end = min(pos + size, end)
        if win_end < end:
            # Only snap within the part of the window beyond the overlap, so every window
            # advances by at least one character past the overlap region.
            lo = pos + max(size // 2, overlap + 1)
            cut = max(text.rfind(" ", lo, win_end), text.rfind("\n", lo, win_end))
            if cut > pos:
                win_end = cut
        s, e = _trim(text, pos, win_end)
        if s < e:
            spans.append((s, e))
        if win_end >= end:
            break
        next_pos = max(win_end - overlap, pos + 1)
        # Snap forward to the start of a word, unless that would skip the whole overlap.
        if 0 < next_pos < win_end and not text[next_pos - 1].isspace():
            ws = re.search(r"\s", text[next_pos:win_end])
            if ws is not None:
                next_pos += ws.start() + 1
        pos = next_pos
    return spans


class FixedSizeChunker:
    def __init__(self, size: int = 1000, overlap: int = 150) -> None:
        if overlap >= size:
            raise ValueError("overlap must be smaller than size")
        self.size = size
        self.overlap = overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        spans = window_spans(doc.text, 0, len(doc.text), self.size, self.overlap)
        return [
            Chunk(
                chunk_id=make_chunk_id(doc.doc_id, s, e),
                doc_id=doc.doc_id,
                text=doc.text[s:e],
                start=s,
                end=e,
                title=doc.title,
            )
            for s, e in spans
        ]


_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")


def find_headings(text: str, max_level: int) -> list[tuple[int, int, str]]:
    """Return ``(offset, level, title)`` for ATX headings outside fenced code blocks.

    Fences matter for technical docs: a ``# comment`` inside a ```python block is not a
    heading, and treating it as one would split code examples in half.
    """
    headings: list[tuple[int, int, str]] = []
    fence: str | None = None
    offset = 0
    for line in text.split("\n"):
        fence_match = _FENCE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker[0] * 3
            elif marker.startswith(fence):
                fence = None
        elif fence is None:
            m = _HEADING.match(line)
            if m and len(m.group(1)) <= max_level:
                headings.append((offset, len(m.group(1)), m.group(2)))
        offset += len(line) + 1
    return headings


class HeadingChunker:
    """One chunk per markdown section (split at headings up to ``max_heading_level``).

    Each chunk's ``title`` is the heading path, e.g. "Security > OAuth2 > Scopes", so a
    section like "Scopes" stays retrievable for a question about OAuth2 scopes even
    though the word OAuth2 only appears in a parent heading. Sections longer than
    ``max_size`` are split with the same windowing as FixedSizeChunker.
    """

    def __init__(
        self, max_heading_level: int = 3, max_size: int = 1500, overlap: int = 150
    ) -> None:
        if overlap >= max_size:
            raise ValueError("overlap must be smaller than max_size")
        self.max_heading_level = max_heading_level
        self.max_size = max_size
        self.overlap = overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.text
        headings = find_headings(text, self.max_heading_level)
        # Section boundaries: document start (preamble), each heading, document end.
        bounds = [0] + [off for off, _, _ in headings if off > 0] + [len(text)]
        titles_at: dict[int, str] = {}
        stack: list[tuple[int, str]] = []
        for off, level, title in headings:
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            titles_at[off] = " > ".join(t for _, t in stack)

        chunks: list[Chunk] = []
        for sec_start, sec_end in pairwise(bounds):
            title = titles_at.get(sec_start, doc.title)
            for s, e in window_spans(text, sec_start, sec_end, self.max_size, self.overlap):
                chunks.append(
                    Chunk(
                        chunk_id=make_chunk_id(doc.doc_id, s, e),
                        doc_id=doc.doc_id,
                        text=text[s:e],
                        start=s,
                        end=e,
                        title=title,
                    )
                )
        return chunks
