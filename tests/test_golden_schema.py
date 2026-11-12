from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from rag.golden.schema import (
    Evidence,
    GoldenItem,
    Provenance,
    Review,
    UnverifiedItemError,
    load_golden,
    read_jsonl,
    validate_against_corpus,
    write_jsonl,
)
from rag.golden.units import fence_at, merge_adjacent, split_units
from rag.prompts import REFUSAL
from rag.types import Document

DOC = Document("d.md", "FastAPI uses port 8000 by default. It is fast.")


def make_item(**overrides: Any) -> GoldenItem:
    data: dict[str, Any] = {
        "id": "fa-1",
        "question": "What port does FastAPI use?",
        "reference_answer": "8000",
        "question_type": "factual",
        "evidence": [Evidence(doc_id="d.md", start=0, end=34, quote=DOC.text[0:34])],
        "provenance": Provenance(drafted_by="test", draft_prompt="x"),
        "corpus_fingerprint": "fp",
    }
    data.update(overrides)
    return GoldenItem.model_validate(data)


ACCEPTED = Review(status="accepted", reviewer="me")


class TestSchema:
    def test_valid_item(self) -> None:
        assert make_item().human_verified is False

    def test_quote_must_match_span_length(self) -> None:
        with pytest.raises(ValidationError, match="quote length"):
            Evidence(doc_id="d.md", start=0, end=5, quote="abc")

    def test_answerable_needs_evidence(self) -> None:
        with pytest.raises(ValidationError, match="evidence"):
            make_item(evidence=[])

    def test_unanswerable_rules(self) -> None:
        ok = make_item(question_type="unanswerable", evidence=[], reference_answer=REFUSAL)
        assert ok.question_type == "unanswerable"
        with pytest.raises(ValidationError, match="refusal"):
            make_item(question_type="unanswerable", evidence=[], reference_answer="no idea")
        with pytest.raises(ValidationError, match="must not have evidence"):
            make_item(question_type="unanswerable", reference_answer=REFUSAL)

    def test_multi_hop_needs_two_spans(self) -> None:
        with pytest.raises(ValidationError, match="two evidence"):
            make_item(question_type="multi_hop")

    def test_verified_flag_must_match_review(self) -> None:
        assert make_item(review=ACCEPTED, human_verified=True).human_verified
        with pytest.raises(ValidationError, match="human_verified"):
            make_item(human_verified=True)  # pending review
        with pytest.raises(ValidationError, match="human_verified"):
            make_item(review=ACCEPTED)  # accepted but flag false

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_item(relevant_chunk_ids=["x"])


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    items = [make_item(id="a"), make_item(id="b", review=ACCEPTED, human_verified=True)]
    path = tmp_path / "g.jsonl"
    write_jsonl(path, items)
    assert read_jsonl(path) == items
    assert not list(tmp_path.glob("*.tmp"))  # atomic write cleaned up


def test_load_golden_refuses_unverified(tmp_path: Path) -> None:
    path = tmp_path / "g.jsonl"
    write_jsonl(path, [make_item()])
    with pytest.raises(UnverifiedItemError):
        load_golden(path)
    assert len(load_golden(path, allow_unverified=True)) == 1


def test_read_jsonl_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "g.jsonl"
    path.write_text('{"id": "broken"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="g.jsonl:1"):
        read_jsonl(path)


def test_validate_against_corpus() -> None:
    docs = {"d.md": DOC}
    assert validate_against_corpus([make_item()], docs, "fp") == []
    wrong_quote = make_item(evidence=[Evidence(doc_id="d.md", start=0, end=4, quote="Nope")])
    other_doc = make_item(id="x", evidence=[Evidence(doc_id="gone.md", start=0, end=1, quote="F")])
    problems = validate_against_corpus(
        [make_item(), make_item(), wrong_quote, other_doc], docs, "other-fp"
    )
    text = "\n".join(problems)
    assert "duplicate id" in text
    assert "same question" in text
    assert "quote does not match" in text
    assert "unknown document gone.md" in text
    assert "different corpus fingerprint" in text


class TestUnits:
    def test_sentences_lines_and_code_blocks(self) -> None:
        text = "First one. Second one.\n\n* item\n\n```python\nx = 1\n\ny = 2\n```\nLast."
        units = [text[u.start : u.end] for u in split_units(text)]
        assert units == [
            "First one.",
            "Second one.",
            "* item",
            "```python\nx = 1\n\ny = 2\n```",
            "Last.",
        ]

    def test_abbreviations_do_not_split(self) -> None:
        text = "Wrap the app (e.g. Flask) first. Then mount it."
        assert [text[u.start : u.end] for u in split_units(text)] == [
            "Wrap the app (e.g. Flask) first.",
            "Then mount it.",
        ]

    def test_span_starting_inside_code_block(self) -> None:
        text = "Intro.\n```python\nline_a = 1\nline_b = 2\n```\nAfter."
        start = text.index("line_b")
        assert fence_at(text, start) == "```"
        units = [text[u.start : u.end] for u in split_units(text, start)]
        assert units == ["line_b = 2\n```", "After."]  # code stays one unit, not prose

    def test_offsets_are_absolute(self) -> None:
        text = "skip this. Keep that."
        (unit,) = split_units(text, start=11)
        assert text[unit.start : unit.end] == "Keep that."

    def test_merge_adjacent(self) -> None:
        text = "A one. B two. C three."
        units = split_units(text)
        merged = merge_adjacent([units[0], units[1]], text)
        assert [text[u.start : u.end] for u in merged] == ["A one. B two."]
        apart = merge_adjacent([units[0], units[2]], text)
        assert len(apart) == 2


def test_write_retries_when_target_is_briefly_locked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import time

    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(src: str, dst: str) -> None:
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("locked by another process")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky_replace)
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    write_jsonl(tmp_path / "g.jsonl", [make_item()])
    assert calls["n"] == 3
    assert len(read_jsonl(tmp_path / "g.jsonl")) == 1
