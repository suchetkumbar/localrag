import json
from collections.abc import Mapping
from typing import Any

import pytest

from rag.golden.draft import (
    DraftConfig,
    DraftRejectedError,
    _tokens,
    candidate_sections,
    check_question,
    draft_items,
    extract_json,
    is_duplicate,
    load_prompts,
    parse_draft,
)
from rag.golden.units import Unit
from rag.prompts import REFUSAL
from rag.types import Document, Generation

PROSE = (
    "FastAPI validates request bodies with Pydantic models. "
    "Invalid data returns an error with status code 422. "
    "The error body lists every field that failed validation. "
    "You can customise the response by overriding the exception handler. "
    "This works for both JSON bodies and form data sent by browsers. "
)
DOC = Document(
    "tutorial/body.md",
    f"# Body\n\n{PROSE}\n\n" + "".join(f"## Part {n}\n\n{PROSE}\n\n" for n in range(1, 6)),
    title="Body",
)


class ScriptedGenerator:
    """Answers each prompt by citing the first unit label it contains."""

    QUESTIONS = (
        "How does FastAPI validate request bodies?",
        "Which status code signals invalid input data?",
        "Can I customise validation error responses?",
        "Why are form submissions also checked?",
        "When does the error body list failing fields?",
    )

    def __init__(self, question: str | None = None) -> None:
        self.question = question
        self.prompts: list[str] = []

    @property
    def identity(self) -> Mapping[str, Any]:
        return {"provider": "scripted", "model": "test"}

    def generate(self, system: str, user: str) -> Generation:
        self.prompts.append(user)
        n = len(self.prompts)
        if "Passage A" in user:
            body = {
                "question": "Why do both checks matter together?",
                "answer": "Both.",
                "evidence_units": ["a2", "b3"],
            }
        elif "does NOT answer" in user:
            body = {"question": f"Does FastAPI support feature {n}?", "why_unanswerable": "x"}
        elif "specific number" in user:
            body = {
                "question": f"What status code is returned {n}?",
                "answer": "422",
                "evidence_units": ["u3"],
            }
        else:
            q = self.question or self.QUESTIONS[(n - 1) % len(self.QUESTIONS)]
            body = {"question": q, "answer": "With Pydantic.", "evidence_units": ["u2"]}
        return Generation(json.dumps(body), "test", 1, 1, 0.0)


LABELS = {"u1": Unit(0, 5), "u2": Unit(6, 10), "a1": Unit(0, 5), "b1": Unit(6, 10)}


class TestParsing:
    def test_extract_json_tolerates_surrounding_text(self) -> None:
        assert extract_json('Sure! {"question": "x"} hope this helps') == {"question": "x"}

    @pytest.mark.parametrize("raw", ["no json here", "{not: valid}", "[1, 2]"])
    def test_extract_json_rejects(self, raw: str) -> None:
        with pytest.raises(DraftRejectedError):
            extract_json(raw)

    @pytest.mark.parametrize(
        ("question", "reason"),
        [
            ("What is FastAPI", "not_a_question"),
            ("Why?", "question_length"),
            ("What does the passage say about ports?", "refers_to_source"),
            ("What is stated in u3 about ports?", "refers_to_source"),
        ],
    )
    def test_check_question_rejects(self, question: str, reason: str) -> None:
        with pytest.raises(DraftRejectedError) as exc:
            check_question(question)
        assert exc.value.reason == reason

    def test_check_question_allows_unit_tests(self) -> None:
        assert check_question("How do I write unit tests  for my app?") == (
            "How do I write unit tests for my app?"
        )

    def test_parse_factual(self) -> None:
        raw = json.dumps(
            {
                "question": "How are bodies validated?",
                "answer": "Pydantic.",
                "evidence_units": ["U2", "u1"],
            }
        )
        assert parse_draft(raw, "factual", LABELS) == (
            "How are bodies validated?",
            "Pydantic.",
            ["u1", "u2"],
        )

    def test_parse_rejects_unknown_units(self) -> None:
        raw = json.dumps(
            {"question": "How are bodies validated?", "answer": "x", "evidence_units": ["u9"]}
        )
        with pytest.raises(DraftRejectedError, match="unknown_units"):
            parse_draft(raw, "factual", LABELS)

    def test_numeric_needs_a_number(self) -> None:
        raw = json.dumps(
            {
                "question": "What code is returned on error?",
                "answer": "an error code",
                "evidence_units": ["u1"],
            }
        )
        with pytest.raises(DraftRejectedError, match="numeric_without_number"):
            parse_draft(raw, "numeric", LABELS)

    def test_multi_hop_needs_both_passages(self) -> None:
        raw = json.dumps(
            {"question": "How do the two relate here?", "answer": "x", "evidence_units": ["a1"]}
        )
        with pytest.raises(DraftRejectedError, match="single_hop"):
            parse_draft(raw, "multi_hop", LABELS)

    def test_unanswerable_gets_refusal(self) -> None:
        raw = json.dumps(
            {"question": "Does FastAPI support CORBA natively?", "why_unanswerable": "x"}
        )
        assert parse_draft(raw, "unanswerable", LABELS) == (
            "Does FastAPI support CORBA natively?",
            REFUSAL,
            [],
        )


def test_duplicate_detection() -> None:
    seen = [_tokens("How does FastAPI validate request bodies?")]
    assert is_duplicate("How does FastAPI validate the request bodies?", seen, 0.75)
    assert not is_duplicate("Which port does uvicorn use by default?", seen, 0.75)


def test_candidate_sections_require_prose() -> None:
    code_only = Document("c.md", "# Code\n\n```python\n" + "x = 1\n" * 200 + "```\n")
    sections = candidate_sections([DOC, code_only], DraftConfig(min_prose_chars=100))
    assert {s.chunk.doc_id for s in sections} == {"tutorial/body.md"}
    assert all(s.has_prose_number for s in sections)  # "422" appears in prose


def test_draft_items_end_to_end() -> None:
    cfg = DraftConfig(
        targets={"factual": 1, "numeric": 1, "multi_hop": 1, "unanswerable": 1},
        min_prose_chars=100,
        max_items_per_doc=10,
    )
    gen = ScriptedGenerator()
    items, stats = draft_items(cfg, [DOC], gen, corpus_fingerprint="fp")

    assert [i.question_type for i in items] == ["factual", "numeric", "multi_hop", "unanswerable"]
    assert all(not i.human_verified and i.review.status == "pending" for i in items)
    assert all(i.corpus_fingerprint == "fp" for i in items)
    assert items[0].provenance.drafted_by == "scripted/test"
    assert items[0].provenance.draft_prompt == load_prompts().version_hash
    for item in items:
        for ev in item.evidence:
            assert DOC.text[ev.start : ev.end] == ev.quote
    assert items[1].evidence[0].quote == "Invalid data returns an error with status code 422."
    assert len(items[2].evidence) == 2  # one span from each section
    assert items[3].evidence == () and items[3].reference_answer == REFUSAL
    assert dict(stats.accepted) == {"factual": 1, "numeric": 1, "multi_hop": 1, "unanswerable": 1}
    assert any('Start the question with "' in p for p in gen.prompts)


def test_draft_items_is_resumable_and_deterministic() -> None:
    cfg = DraftConfig(
        targets={"factual": 2, "numeric": 0, "multi_hop": 0, "unanswerable": 0}, min_prose_chars=100
    )
    first, _ = draft_items(cfg, [DOC], ScriptedGenerator(), corpus_fingerprint="fp")
    again, _ = draft_items(cfg, [DOC], ScriptedGenerator(), corpus_fingerprint="fp")
    assert [i.id for i in first] == [i.id for i in again]
    gen = ScriptedGenerator()
    resumed, stats = draft_items(cfg, [DOC], gen, corpus_fingerprint="fp", existing=first)
    assert resumed == first and gen.prompts == []  # targets already met: no LLM calls
    assert not stats.attempted


def test_rejected_drafts_are_counted_not_kept() -> None:
    cfg = DraftConfig(
        targets={"factual": 1, "numeric": 0, "multi_hop": 0, "unanswerable": 0}, min_prose_chars=100
    )
    gen = ScriptedGenerator(question="What does the passage say about bodies?")
    items, stats = draft_items(cfg, [DOC], gen, corpus_fingerprint="fp")
    assert items == []
    assert stats.rejected["refers_to_source"] == len(gen.prompts) > 0
