"""Interactive human review of drafted golden items.

The reviewer sees each pending item with its evidence highlighted in the source document,
and accepts, edits or rejects it. Only accepted or edited items get ``human_verified``.
Every decision is saved immediately (atomic write), so a session can be stopped and
resumed at any point.

Input and output are injected (``ask``/``say``) so the whole flow is unit-tested.
"""

import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from pydantic import ValidationError

from rag.chunkers import HeadingChunker
from rag.golden.schema import QUESTION_TYPES, Evidence, GoldenItem, QuestionType, Review
from rag.prompts import REFUSAL
from rag.retrievers import BM25Retriever
from rag.types import Document

CONTEXT_CHARS = 240


def find_quote(text: str, quote: str) -> tuple[int, int] | None:
    """Locate ``quote`` in ``text``: exact match first, then whitespace-insensitive.

    The fallback lets a reviewer paste a multi-line passage as one line.
    """
    quote = quote.strip()
    if not quote:
        return None
    i = text.find(quote)
    if i != -1:
        return i, i + len(quote)
    pattern = r"\s+".join(re.escape(word) for word in quote.split())
    m = re.search(pattern, text)
    return (m.start(), m.end()) if m else None


def show_in_context(doc: Document, ev: Evidence, width: int = CONTEXT_CHARS) -> str:
    before = doc.text[max(0, ev.start - width) : ev.start]
    after = doc.text[ev.end : ev.end + width]
    return f"...{before}>>>{ev.quote}<<<{after}..."


@dataclass
class ReviewSession:
    items: list[GoldenItem]
    docs: Mapping[str, Document]
    save: Callable[[list[GoldenItem]], None]
    ask: Callable[[str], str] = input
    say: Callable[[str], None] = print
    reviewer: str = "human"
    quotas: Mapping[QuestionType, int] = field(default_factory=dict)
    now: Callable[[], datetime] = lambda: datetime.now(UTC)
    _bm25: BM25Retriever | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------ main loop

    def run(self, only_type: QuestionType | None = None) -> Counter[str]:
        decisions: Counter[str] = Counter()
        queue = [
            i
            for i, item in enumerate(self.items)
            if item.review.status == "pending"
            and (only_type is None or item.question_type == only_type)
        ]
        for n, index in enumerate(queue, start=1):
            self._show(self.items[index], n, len(queue))
            action = self._decide(index)
            if action == "quit":
                break
            decisions[action] += 1
            if action != "skip":
                self.save(self.items)
        self.say(f"\nsession: {dict(decisions)}  totals: {self.progress()}")
        return decisions

    def progress(self) -> str:
        verified = Counter(i.question_type for i in self.items if i.human_verified)
        parts = [f"{q} {verified[q]}/{self.quotas.get(q, '?')}" for q in QUESTION_TYPES]
        return ", ".join(parts)

    def _decide(self, index: int) -> str:
        while True:
            choice = self.ask("[a]ccept  [e]dit  [r]eject  [s]kip  [q]uit > ").strip().lower()
            item = self.items[index]
            if choice == "a":
                self.items[index] = self._reviewed(item, "accepted")
                return "accepted"
            if choice == "r":
                note = self.ask("reason for rejecting (optional): ").strip()
                self.items[index] = self._reviewed(item, "rejected", note)
                return "rejected"
            if choice == "e":
                edited = self._edit(item)
                if edited is not None:
                    self.items[index] = edited
                    return "edited"
                self._show(item, None, None)
            elif choice == "s":
                return "skip"
            elif choice == "q":
                return "quit"

    def _reviewed(self, item: GoldenItem, status: str, note: str = "") -> GoldenItem:
        review = Review.model_validate(
            {"status": status, "reviewer": self.reviewer, "reviewed_at": self.now(), "note": note}
        )
        return item.model_copy(
            update={"review": review, "human_verified": status in ("accepted", "edited")}
        )

    # ------------------------------------------------------------------ display

    def _show(self, item: GoldenItem, n: int | None, total: int | None) -> None:
        header = f"[{n}/{total}] " if n is not None else ""
        self.say("\n" + "=" * 78)
        self.say(f"{header}{item.id}  {item.question_type}   progress: {self.progress()}")
        self.say(f"Q: {item.question}")
        self.say(f"A: {item.reference_answer}")
        for k, ev in enumerate(item.evidence, start=1):
            self.say(f"\n  evidence {k}: {ev.doc_id} [{ev.start}:{ev.end}]")
            doc = self.docs.get(ev.doc_id)
            self.say("  " + (show_in_context(doc, ev) if doc else "(document not in corpus!)"))
        if item.question_type == "unanswerable":
            self.say("\n  Should be UNANSWERABLE. Closest passages in the corpus (BM25):")
            for rank, hit in enumerate(self._closest(item.question), start=1):
                snippet = " ".join(hit.split())[:300]
                self.say(f"  {rank}. {snippet}")

    def _closest(self, question: str, k: int = 3) -> list[str]:
        if self._bm25 is None:
            chunker = HeadingChunker(max_heading_level=3, max_size=1500, overlap=0)
            self._bm25 = BM25Retriever()
            self._bm25.index([c for d in self.docs.values() for c in chunker.chunk(d)])
        return [
            f"({r.chunk.chunk_id}) {r.chunk.indexed_text}" for r in self._bm25.search(question, k)
        ]

    # ------------------------------------------------------------------ editing

    def _edit(self, item: GoldenItem) -> GoldenItem | None:
        """Field-by-field edit; Enter keeps the current value. None means cancelled."""
        question = self.ask(f"question [{item.question}]: ").strip() or item.question
        qtype_raw = self.ask(f"type {QUESTION_TYPES} [{item.question_type}]: ").strip()
        qtype = qtype_raw or item.question_type
        if qtype not in QUESTION_TYPES:
            self.say(f"unknown type {qtype!r}; edit cancelled")
            return None
        if qtype == "unanswerable":
            answer, evidence = REFUSAL, []
        else:
            default = item.reference_answer if item.reference_answer != REFUSAL else ""
            answer = self.ask(f"answer [{default}]: ").strip() or default
            evidence = self._edit_evidence(list(item.evidence))
        note = self.ask("note (optional): ").strip()
        try:
            candidate = item.model_copy(
                update={
                    "question": question,
                    "question_type": qtype,
                    "reference_answer": answer,
                    "evidence": tuple(evidence),
                }
            )
            edited = self._reviewed(
                GoldenItem.model_validate(candidate.model_dump()), "edited", note
            )
        except ValidationError as exc:
            self.say(f"invalid item, edit cancelled: {exc.errors()[0]['msg']}")
            return None
        return edited

    def _edit_evidence(self, evidence: list[Evidence]) -> list[Evidence]:
        kept: list[Evidence] = []
        for k, ev in enumerate(evidence, start=1):
            choice = self.ask(f"evidence {k} ({ev.doc_id}): [k]eep [d]elete [r]eplace > ").strip()
            if choice == "d":
                continue
            if choice == "r":
                new = self._ask_evidence(default_doc=ev.doc_id)
                kept.append(new or ev)
            else:
                kept.append(ev)
        while self.ask("add another evidence span? [y/N] ").strip().lower() == "y":
            new = self._ask_evidence(default_doc=kept[0].doc_id if kept else None)
            if new:
                kept.append(new)
        return kept

    def _ask_evidence(self, default_doc: str | None) -> Evidence | None:
        doc_id = self.ask(f"document [{default_doc or ''}]: ").strip() or default_doc
        doc = self.docs.get(doc_id or "")
        if doc is None:
            self.say(f"unknown document {doc_id!r}")
            return None
        quote = self.ask("paste the exact supporting text: ")
        span = find_quote(doc.text, quote)
        if span is None:
            self.say("text not found in that document")
            return None
        start, end = span
        return Evidence(doc_id=doc.doc_id, start=start, end=end, quote=doc.text[start:end])


def apply_quotas(
    items: Sequence[GoldenItem], quotas: Mapping[QuestionType, int]
) -> tuple[list[GoldenItem], dict[QuestionType, int]]:
    """Take the first ``quota`` verified items of each type, in file order.

    Returns the selection and, per type, how many items are still missing.
    """
    selected: list[GoldenItem] = []
    taken: Counter[str] = Counter()
    for item in items:
        if item.human_verified and taken[item.question_type] < quotas.get(item.question_type, 0):
            selected.append(item)
            taken[item.question_type] += 1
    missing = {q: quotas[q] - taken[q] for q in quotas if taken[q] < quotas[q]}
    return selected, missing
