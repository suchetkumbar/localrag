"""Golden item schema and JSONL I/O.

Evidence is stored as character spans in the prepared corpus (``doc_id``, ``start``,
``end``) plus the quoted text, never as chunk IDs: chunk IDs depend on the chunker, spans
do not. A chunk is relevant to an item if it overlaps one of the item's spans; the
retrieval metrics (Milestone 3) derive per-config relevance from that.
"""

import json
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag.fileio import write_lines_atomic
from rag.prompts import REFUSAL
from rag.types import Document

QuestionType = Literal["factual", "numeric", "multi_hop", "unanswerable"]
QUESTION_TYPES: tuple[QuestionType, ...] = ("factual", "numeric", "multi_hop", "unanswerable")
ReviewStatus = Literal["pending", "accepted", "edited", "rejected"]
REVIEW_STATUSES: tuple[ReviewStatus, ...] = ("pending", "accepted", "edited", "rejected")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Evidence(_Model):
    doc_id: str
    start: int = Field(ge=0)
    end: int
    quote: str

    @model_validator(mode="after")
    def _span_matches_quote_length(self) -> Self:
        if self.end <= self.start:
            raise ValueError("evidence end must be greater than start")
        if len(self.quote) != self.end - self.start:
            raise ValueError("quote length does not match the span")
        return self


class Review(_Model):
    status: ReviewStatus = "pending"
    reviewer: str | None = None
    reviewed_at: datetime | None = None
    note: str = ""


class Provenance(_Model):
    drafted_by: str  # e.g. "ollama/llama3" or "human"
    draft_prompt: str  # prompt template version hash, or "manual"
    source_chunks: tuple[str, ...] = ()  # sections shown to the drafting LLM


class GoldenItem(_Model):
    id: str
    question: str = Field(min_length=5)
    reference_answer: str = Field(min_length=1)
    question_type: QuestionType
    evidence: tuple[Evidence, ...] = ()
    human_verified: bool = False
    review: Review = Review()
    provenance: Provenance
    corpus_fingerprint: str

    @model_validator(mode="after")
    def _consistency(self) -> Self:
        if self.question_type == "unanswerable":
            if self.evidence:
                raise ValueError("unanswerable items must not have evidence")
            if self.reference_answer != REFUSAL:
                raise ValueError("unanswerable items must use the fixed refusal as reference")
        elif not self.evidence:
            raise ValueError(f"{self.question_type} items need at least one evidence span")
        if self.question_type == "multi_hop" and len(self.evidence) < 2:
            raise ValueError("multi_hop items need at least two evidence spans")
        if self.human_verified != (self.review.status in ("accepted", "edited")):
            raise ValueError("human_verified must be true exactly when the review accepted it")
        return self


class UnverifiedItemError(ValueError):
    pass


def read_jsonl(path: Path) -> list[GoldenItem]:
    items = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                items.append(GoldenItem.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return items


def write_jsonl(path: Path, items: Iterable[GoldenItem]) -> None:
    write_lines_atomic(
        path, (json.dumps(i.model_dump(mode="json"), ensure_ascii=False) for i in items)
    )


def load_golden(path: Path, *, allow_unverified: bool = False) -> list[GoldenItem]:
    """Load a dataset for evaluation. Unreviewed items are never treated as ground truth."""
    items = read_jsonl(path)
    unverified = [i.id for i in items if not i.human_verified]
    if unverified and not allow_unverified:
        raise UnverifiedItemError(
            f"{path} has {len(unverified)} unverified items (e.g. {unverified[:3]}); "
            "review them or pass allow_unverified=True for smoke tests only"
        )
    return items


def validate_against_corpus(
    items: Iterable[GoldenItem], docs: Mapping[str, Document], fingerprint: str | None
) -> list[str]:
    """Return a list of problems (empty if the dataset matches this corpus exactly)."""
    problems: list[str] = []
    seen_ids: set[str] = set()
    seen_questions: dict[str, str] = {}
    for item in items:
        if item.id in seen_ids:
            problems.append(f"{item.id}: duplicate id")
        seen_ids.add(item.id)
        q = " ".join(item.question.lower().split())
        if q in seen_questions:
            problems.append(f"{item.id}: same question as {seen_questions[q]}")
        seen_questions.setdefault(q, item.id)
        if fingerprint is not None and item.corpus_fingerprint != fingerprint:
            problems.append(f"{item.id}: built against a different corpus fingerprint")
        for ev in item.evidence:
            doc = docs.get(ev.doc_id)
            if doc is None:
                problems.append(f"{item.id}: unknown document {ev.doc_id}")
            elif doc.text[ev.start : ev.end] != ev.quote:
                problems.append(f"{item.id}: quote does not match {ev.doc_id}[{ev.start}:{ev.end}]")
    return problems
