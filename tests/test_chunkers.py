import pytest

from rag.chunkers import FixedSizeChunker, HeadingChunker, find_headings, window_spans
from rag.types import Chunk, Document

LOREM = " ".join(f"word{i:03d}" for i in range(400))  # 3199 chars, no newlines


def assert_offset_invariant(doc: Document, chunks: list[Chunk]) -> None:
    for c in chunks:
        assert c.text == doc.text[c.start : c.end]
        assert c.text == c.text.strip() and c.text
        assert c.chunk_id == f"{doc.doc_id}#{c.start}-{c.end}"


def assert_covers_all_words(doc: Document, chunks: list[Chunk]) -> None:
    covered = [False] * len(doc.text)
    for c in chunks:
        for i in range(c.start, c.end):
            covered[i] = True
    missed = [i for i, ch in enumerate(doc.text) if not ch.isspace() and not covered[i]]
    assert missed == []


class TestWindowSpans:
    def test_short_text_is_one_span(self) -> None:
        assert window_spans("  hello world  ", 0, 15, 100, 10) == [(2, 13)]

    def test_hard_cut_when_no_whitespace(self) -> None:
        assert window_spans("a" * 25, 0, 25, 10, 0) == [(0, 10), (10, 20), (20, 25)]

    def test_respects_size_and_snaps_to_words(self) -> None:
        spans = window_spans(LOREM, 0, len(LOREM), 100, 20)
        for s, e in spans:
            assert e - s <= 100
            assert s == 0 or LOREM[s - 1] == " "  # starts at a word boundary
            assert e == len(LOREM) or LOREM[e] == " "  # ends at a word boundary

    def test_overlap_between_consecutive_windows(self) -> None:
        spans = window_spans(LOREM, 0, len(LOREM), 100, 20)
        for (_, e1), (s2, _) in zip(spans, spans[1:], strict=False):
            assert s2 < e1  # windows overlap
            assert e1 - s2 <= 20

    def test_large_overlap_still_terminates(self) -> None:
        spans = window_spans(LOREM, 0, len(LOREM), 100, 99)
        assert len(spans) < len(LOREM)
        assert spans[-1][1] == len(LOREM)


class TestFixedSizeChunker:
    def test_invariants(self) -> None:
        doc = Document("d.md", LOREM)
        chunks = FixedSizeChunker(size=200, overlap=30).chunk(doc)
        assert len(chunks) > 1
        assert_offset_invariant(doc, chunks)
        assert_covers_all_words(doc, chunks)

    def test_deterministic(self, sample_doc: Document) -> None:
        a = FixedSizeChunker(120, 20).chunk(sample_doc)
        b = FixedSizeChunker(120, 20).chunk(sample_doc)
        assert a == b

    def test_invalid_overlap(self) -> None:
        with pytest.raises(ValueError):
            FixedSizeChunker(size=10, overlap=10)

    def test_empty_document(self) -> None:
        assert FixedSizeChunker().chunk(Document("e.md", "   \n ")) == []


class TestHeadingChunker:
    def test_headings_inside_code_fences_are_ignored(self, sample_doc: Document) -> None:
        titles = [t for _, _, t in find_headings(sample_doc.text, 6)]
        assert titles == ["Security", "OAuth2", "Scopes", "HTTP Basic Auth"]

    def test_max_heading_level(self, sample_doc: Document) -> None:
        titles = [t for _, _, t in find_headings(sample_doc.text, 2)]
        assert "Scopes" not in titles

    def test_one_chunk_per_section_with_heading_path(self, sample_doc: Document) -> None:
        chunks = HeadingChunker(max_heading_level=3, max_size=1000).chunk(sample_doc)
        assert [c.title for c in chunks] == [
            "Security",
            "Security > OAuth2",
            "Security > OAuth2 > Scopes",
            "Security > HTTP Basic Auth",  # level-2 heading pops "OAuth2 > Scopes"
        ]
        assert chunks[2].text.startswith("### Scopes")
        assert "def get_user" in chunks[1].text  # code block stays with its section
        assert_offset_invariant(sample_doc, chunks)
        assert_covers_all_words(sample_doc, chunks)

    def test_indexed_text_includes_heading_path(self, sample_doc: Document) -> None:
        scopes = HeadingChunker().chunk(sample_doc)[2]
        assert scopes.indexed_text.startswith("Security > OAuth2 > Scopes\n\n### Scopes")

    def test_oversized_section_is_windowed(self) -> None:
        doc = Document("big.md", "# Big\n\n" + LOREM)
        chunks = HeadingChunker(max_size=300, overlap=50).chunk(doc)
        assert len(chunks) > 5
        assert all(len(c.text) <= 300 for c in chunks)
        assert all(c.title == "Big" for c in chunks)
        assert_offset_invariant(doc, chunks)
        assert_covers_all_words(doc, chunks)

    def test_preamble_before_first_heading(self) -> None:
        doc = Document("p.md", "Intro text.\n\n# First\n\nBody.", title="Doc Title")
        chunks = HeadingChunker().chunk(doc)
        assert [(c.title, c.text) for c in chunks] == [
            ("Doc Title", "Intro text."),
            ("First", "# First\n\nBody."),
        ]
