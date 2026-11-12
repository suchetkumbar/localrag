from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rag.golden.card import END, START, stats_markdown, update_card
from rag.golden.review import ReviewSession, apply_quotas, find_quote
from rag.golden.schema import Evidence, GoldenItem, Review
from rag.prompts import REFUSAL
from rag.types import Document
from tests.test_golden_schema import DOC, make_item

NOW = datetime(2026, 11, 12, tzinfo=UTC)


def session(
    items: list[GoldenItem], answers: list[str]
) -> tuple[ReviewSession, list[str], list[int]]:
    inputs: Iterator[str] = iter(answers)
    output: list[str] = []
    saves: list[int] = []
    s = ReviewSession(
        items=items,
        docs={"d.md": DOC},
        save=lambda current: saves.append(len(current)),
        ask=lambda _prompt: next(inputs),
        say=output.append,
        reviewer="tester",
        quotas={"factual": 2},
        now=lambda: NOW,
    )
    return s, output, saves


def test_find_quote_exact_and_whitespace_insensitive() -> None:
    text = "Use the `Depends`\n   function to declare dependencies."
    assert find_quote(text, "function to") == (text.index("function"), text.index("function") + 11)
    start, end = find_quote(text, "`Depends` function") or (0, 0)
    assert text[start:end] == "`Depends`\n   function"
    assert find_quote(text, "not there") is None


def test_accept_reject_skip_and_save_after_each_decision() -> None:
    items = [make_item(id="a"), make_item(id="b"), make_item(id="c")]
    s, output, saves = session(items, ["a", "r", "too vague", "s"])
    decisions = s.run()
    assert dict(decisions) == {"accepted": 1, "rejected": 1, "skip": 1}
    assert saves == [3, 3]  # saved after accept and reject, not after skip
    a, b, c = s.items
    assert a.human_verified and a.review == Review(
        status="accepted", reviewer="tester", reviewed_at=NOW
    )
    assert not b.human_verified and b.review.status == "rejected" and b.review.note == "too vague"
    assert c.review.status == "pending"
    assert any(">>>FastAPI uses port 8000 by default.<<<" in line for line in output)


def test_quit_stops_and_only_pending_items_are_shown() -> None:
    reviewed = make_item(id="done", review=Review(status="accepted"), human_verified=True)
    s, output, _ = session([reviewed, make_item(id="todo")], ["q"])
    s.run()
    assert any("todo" in line for line in output)
    assert not any("done " in line for line in output)


def test_edit_question_answer_and_replace_evidence() -> None:
    answers = [
        "e",
        "Which port does FastAPI use by default?",  # question
        "",  # keep type
        "Port 8000.",  # answer
        "r",  # replace evidence 1
        "",  # same document
        "It is fast.",  # new quote
        "n",  # no more evidence
        "clarified",  # note
    ]
    s, _, _ = session([make_item()], answers)
    s.run()
    item = s.items[0]
    assert item.review.status == "edited" and item.human_verified
    assert item.question == "Which port does FastAPI use by default?"
    assert item.reference_answer == "Port 8000."
    assert item.evidence == (Evidence(doc_id="d.md", start=35, end=46, quote="It is fast."),)


def test_edit_to_unanswerable_clears_evidence() -> None:
    s, _, _ = session([make_item()], ["e", "", "unanswerable", ""])
    s.run()
    item = s.items[0]
    assert item.question_type == "unanswerable"
    assert item.evidence == () and item.reference_answer == REFUSAL


def test_invalid_edit_is_cancelled_and_item_stays_pending() -> None:
    # Deleting the only evidence of a factual item is invalid; the reviewer then quits.
    s, output, saves = session([make_item()], ["e", "", "", "", "d", "n", "", "q"])
    s.run()
    assert s.items[0].review.status == "pending"
    assert saves == []
    assert any("edit cancelled" in line for line in output)


def test_unanswerable_shows_closest_passages() -> None:
    item = make_item(question_type="unanswerable", evidence=[], reference_answer=REFUSAL)
    s, output, _ = session([item], ["s"])
    s.run()
    assert any("Closest passages" in line for line in output)
    assert any("FastAPI uses port 8000" in line for line in output)


def test_apply_quotas_takes_first_verified_in_order() -> None:
    ok = Review(status="accepted")
    items = [
        make_item(id="1", review=ok, human_verified=True),
        make_item(id="2"),  # pending: never selected
        make_item(id="3", review=ok, human_verified=True),
        make_item(id="4", review=ok, human_verified=True),
    ]
    selected, missing = apply_quotas(items, {"factual": 2, "numeric": 1})
    assert [i.id for i in selected] == ["1", "3"]
    assert missing == {"numeric": 1}


def test_card_stats_are_computed_and_written(tmp_path: Path) -> None:
    ok = Review(status="accepted")
    dataset = [make_item(id="1", review=ok, human_verified=True)]
    drafts = dataset + [make_item(id="2", review=Review(status="rejected"))]
    stats = stats_markdown(dataset, drafts)
    assert "| factual | 1 | 100% |" in stats
    assert "1 accepted as-is, 0 edited, 1 rejected" in stats
    card = tmp_path / "README.md"
    card.write_text(f"# Card\n\n{START}\nold\n{END}\n\nKnown limits.\n", encoding="utf-8")
    update_card(card, stats)
    text = card.read_text(encoding="utf-8")
    assert (
        "OLD-STATS" not in text and "| factual | 1 |" in text and text.endswith("Known limits.\n")
    )


def test_card_without_markers_fails(tmp_path: Path) -> None:
    card = tmp_path / "README.md"
    card.write_text("no markers", encoding="utf-8")
    with pytest.raises(ValueError, match="markers"):
        update_card(card, "x")


def test_review_session_handles_missing_document() -> None:
    item = make_item(evidence=[Evidence(doc_id="gone.md", start=0, end=1, quote="x")])
    s, output, _ = session([item], ["s"])
    s.docs = {"d.md": Document("d.md", "text")}
    s.run()
    assert any("document not in corpus" in line for line in output)
