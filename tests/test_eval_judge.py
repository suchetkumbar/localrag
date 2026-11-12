"""Tests for the LLM judge, per-item evaluation, aggregation and judge validation."""

import json
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from rag.eval import __main__ as eval_cli
from rag.eval.evaluate import EvalConfig, ItemResult, aggregate, evaluate_item
from rag.eval.judge import JudgeError, LLMJudge, load_rubric, parse_correctness, parse_faithfulness
from rag.eval.labels import (
    HumanLabel,
    LabelSession,
    agreement_report,
    format_report,
    read_labels,
    read_results,
    sample_for_labeling,
    write_labels,
    write_results,
)
from rag.golden.schema import Evidence, GoldenItem, Provenance
from rag.pipeline import Answer
from rag.prompts import REFUSAL
from rag.types import Chunk, Generation, ScoredChunk

NOW = datetime(2026, 11, 12, tzinfo=UTC)


class ScriptedJudgeModel:
    """Returns canned JSON depending on which rubric is asking."""

    def __init__(self, label: str = "correct", supported: tuple[bool, ...] = (True,)) -> None:
        self.label = label
        self.supported = supported
        self.calls: list[str] = []

    @property
    def identity(self) -> Mapping[str, Any]:
        return {"provider": "scripted", "model": "judge"}

    def generate(self, system: str, user: str) -> Generation:
        self.calls.append(user)
        if "Reference answer" in user:
            body: dict[str, Any] = {"reason": "matches", "label": self.label}
        else:
            body = {
                "claims": [{"claim": f"c{i}", "supported": s} for i, s in enumerate(self.supported)]
            }
        return Generation(json.dumps(body), "judge", 1, 1, 0.0)


# --------------------------------------------------------------------------- rubrics + parsing


def test_rubrics_load_with_versions() -> None:
    for name in ("correctness_v1", "faithfulness_v1"):
        rubric = load_rubric(name)
        assert rubric.system and rubric.user
        assert rubric.version.startswith(f"{name}@") and len(rubric.version) == len(name) + 13


def test_rubric_needs_both_sections(tmp_path: Path) -> None:
    (tmp_path / "bad.txt").write_text("### system\nonly system\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sections"):
        load_rubric("bad", directory=tmp_path)


def test_parse_correctness() -> None:
    v = parse_correctness('Here: {"reason": "same facts", "label": "Partial"}', "r@1")
    assert (v.label, v.score, v.rubric) == ("partial", 0.5, "r@1")
    with pytest.raises(JudgeError, match="label"):
        parse_correctness('{"label": "mostly right"}', "r@1")
    with pytest.raises(JudgeError, match="JSON"):
        parse_correctness("I think it is correct.", "r@1")


def test_parse_faithfulness() -> None:
    v = parse_faithfulness(
        '{"claims": [{"claim": "a", "supported": true}, {"claim": "b", "supported": false}]}', "f@1"
    )
    assert v.score == 0.5 and v.faithful is False
    empty = parse_faithfulness('{"claims": []}', "f@1")
    assert empty.score is None and empty.faithful is None
    with pytest.raises(JudgeError, match="malformed"):
        parse_faithfulness('{"claims": [{"claim": "a", "supported": "yes"}]}', "f@1")
    with pytest.raises(JudgeError, match="claims"):
        parse_faithfulness('{"verdict": "ok"}', "f@1")


def test_judge_prompts_contain_inputs() -> None:
    model = ScriptedJudgeModel()
    judge = LLMJudge(model)
    judge.correctness("Q?", "REF", "ANS")
    judge.faithfulness("ANS", ["first passage", "second passage"])
    assert "Q?" in model.calls[0] and "REF" in model.calls[0] and "ANS" in model.calls[0]
    assert "[1] first passage" in model.calls[1] and "[2] second passage" in model.calls[1]
    assert judge.identity["correctness_rubric"].startswith("correctness_v1@")


# --------------------------------------------------------------------------- evaluate_item

DOC_TEXT = "x" * 2000


def _item(qtype: str = "factual", item_id: str = "fa-1") -> GoldenItem:
    unanswerable = qtype == "unanswerable"
    return GoldenItem(
        id=item_id,
        question="Which port does FastAPI use?",
        reference_answer=REFUSAL if unanswerable else "8000",
        question_type=qtype,  # type: ignore[arg-type]
        evidence=()
        if unanswerable
        else (Evidence(doc_id="d.md", start=500, end=560, quote="x" * 60),),
        provenance=Provenance(drafted_by="t", draft_prompt="t"),
        corpus_fingerprint="fp",
    )


def _answer(text: str, latency: float = 100.0) -> Answer:
    chunks = [Chunk(f"d.md#{s}-{s + 200}", "d.md", "x" * 200, s, s + 200) for s in (0, 450, 1000)]
    ranking = [ScoredChunk(c, 1.0 - i / 10) for i, c in enumerate(chunks)]
    return Answer(
        question="Which port does FastAPI use?",
        ranking=ranking,
        context=ranking[:2],
        generation=Generation(text, "llama3", 120, 8, latency),
        timings_ms={"retrieve_ms": 5.0, "generate_ms": latency},
    )


def test_evaluate_answerable_item() -> None:
    model = ScriptedJudgeModel(label="correct", supported=(True, False))
    r = evaluate_item(_item(), _answer("Port 8000 [2]."), EvalConfig(), LLMJudge(model), "ollama")
    assert r.retrieval is not None
    assert r.retrieval["recall@1"] == 0.0 and r.retrieval["recall@3"] == 1.0
    assert r.retrieval["mrr"] == 0.5
    assert [x.relevant for x in r.ranking] == [False, True, False]
    assert r.citations.precision == 1.0
    assert r.judge is not None
    assert r.judge.correctness_label == "correct" and r.judge.faithfulness == 0.5
    assert r.ops.cost_usd == 0.0 and r.ops.prompt_tokens == 120
    assert not r.refused


def test_refusal_on_answerable_is_incorrect_without_judge_call() -> None:
    model = ScriptedJudgeModel()
    r = evaluate_item(_item(), _answer(REFUSAL), EvalConfig(), LLMJudge(model), "ollama")
    assert r.refused
    assert r.judge is not None and r.judge.correctness_label == "incorrect"
    assert r.judge.faithfulness is None
    assert model.calls == []


def test_unanswerable_item() -> None:
    model = ScriptedJudgeModel()
    refused = evaluate_item(
        _item("unanswerable"), _answer(REFUSAL), EvalConfig(), LLMJudge(model), "ollama"
    )
    assert refused.retrieval is None and refused.refused
    assert refused.judge is not None and refused.judge.correctness_label is None
    hallucinated = evaluate_item(
        _item("unanswerable"), _answer("Up to 100 [1]."), EvalConfig(), LLMJudge(model), "ollama"
    )
    assert not hallucinated.refused
    assert (
        hallucinated.judge is not None and hallucinated.judge.faithfulness == 1.0
    )  # judged anyway


def test_judge_errors_are_recorded_not_raised() -> None:
    class Broken(ScriptedJudgeModel):
        def generate(self, system: str, user: str) -> Generation:
            return Generation("not json", "judge", 1, 1, 0.0)

    r = evaluate_item(_item(), _answer("Port 8000."), EvalConfig(), LLMJudge(Broken()), "ollama")
    assert r.judge is not None and len(r.judge.errors) == 2
    assert r.judge.correctness_label is None


def test_aggregate() -> None:
    cfg = EvalConfig()
    good = evaluate_item(
        _item(item_id="a"),
        _answer("8000 [2].", 100),
        cfg,
        LLMJudge(ScriptedJudgeModel("correct")),
        "ollama",
    )
    partial = evaluate_item(
        _item(item_id="b"),
        _answer("80 [1].", 300),
        cfg,
        LLMJudge(ScriptedJudgeModel("partial")),
        "ollama",
    )
    refusal = evaluate_item(
        _item(item_id="c"), _answer(REFUSAL, 50), cfg, LLMJudge(ScriptedJudgeModel()), "ollama"
    )
    unans = evaluate_item(
        _item("unanswerable", "u"),
        _answer(REFUSAL, 50),
        cfg,
        LLMJudge(ScriptedJudgeModel()),
        "ollama",
    )
    m = aggregate([good, partial, refusal, unans])

    assert m["correctness"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert m["accuracy"] == pytest.approx(1 / 3)
    assert m["refusal_accuracy"] == 1.0
    assert m["false_refusal_rate"] == pytest.approx(1 / 3)
    assert m["recall@3"] == 1.0  # same ranking for every answerable item
    assert m["citation_validity"] == 1.0 and m["citation_precision"] == pytest.approx(0.5)
    assert m["cost_per_query_usd"] == 0.0
    assert m["latency_ms"]["model_latency_ms"]["p50"] == pytest.approx(75.0)
    assert m["n"]["answerable"] == 3 and m["n"]["unanswerable"] == 1
    assert m["by_type"]["unanswerable"]["refused"] == 1.0


def test_aggregate_cost_unknown_if_any_price_missing() -> None:
    r = evaluate_item(_item(), _answer("8000."), EvalConfig(), None, "https://api.example/v1")
    assert r.judge is None and r.ops.cost_usd is None
    assert aggregate([r])["cost_per_query_usd"] is None


# --------------------------------------------------------------------------- judge validation


def _results() -> list[ItemResult]:
    cfg = EvalConfig()
    labels = ["correct", "correct", "partial", "incorrect", "correct", "incorrect"]
    out = []
    for i, label in enumerate(labels):
        model = ScriptedJudgeModel(label=label, supported=(i % 2 == 0,))
        out.append(
            evaluate_item(
                _item(item_id=f"q{i}"), _answer(f"Answer {i} [1]."), cfg, LLMJudge(model), "ollama"
            )
        )
    out.append(
        evaluate_item(
            _item(item_id="refused"),
            _answer(REFUSAL),
            cfg,
            LLMJudge(ScriptedJudgeModel()),
            "ollama",
        )
    )
    return out


def test_results_roundtrip(tmp_path: Path) -> None:
    results = _results()
    write_results(tmp_path / "items.jsonl", results)
    assert read_results(tmp_path / "items.jsonl") == results


def test_sample_is_reproducible_and_skips_refusals() -> None:
    results = _results()
    a, b = sample_for_labeling(results, 4, seed=7), sample_for_labeling(results, 4, seed=7)
    assert [r.id for r in a] == [r.id for r in b]
    assert "refused" not in {r.id for r in sample_for_labeling(results, 100, seed=7)}


def test_label_session_is_blind_and_saves(tmp_path: Path) -> None:
    results = _results()[:2]
    answers: Iterator[str] = iter(["x", "c", "y", "", "i", "-", "unclear"])
    output: list[str] = []
    saved: list[int] = []
    session = LabelSession(
        items=results,
        labels={},
        save=lambda labels: saved.append(len(labels)),
        ask=lambda _p: next(answers),
        say=output.append,
        now=lambda: NOW,
    )
    assert session.run() == 2
    assert saved == [1, 2]
    assert session.labels["q0"].correctness == "correct" and session.labels["q0"].faithful is True
    assert session.labels["q1"].faithful is None and session.labels["q1"].note == "unclear"
    shown = "\n".join(output)
    assert "matches" not in shown  # the judge's reason is never shown to the labeler
    write_labels(tmp_path / "l.jsonl", session.labels)
    assert read_labels(tmp_path / "l.jsonl") == session.labels


def test_agreement_report_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    results = _results()
    # The human agrees with the judge on every item except q2 (judge: partial).
    human = {"correct": "correct", "partial": "incorrect", "incorrect": "incorrect"}
    labels = {}
    for r in results[:6]:
        assert r.judge is not None and r.judge.correctness_label is not None
        labels[r.id] = HumanLabel(
            id=r.id,
            correctness=human[r.judge.correctness_label],  # type: ignore[arg-type]
            faithful=r.judge.faithful,
            labeler="t",
            labeled_at=NOW,
        )
    report = agreement_report(results, labels)
    assert report.correctness is not None and report.faithfulness is not None
    assert report.correctness.n == 6 and report.correctness.percent == pytest.approx(5 / 6)
    assert report.faithfulness.kappa == pytest.approx(1.0)
    assert "disagreement" in format_report(report)

    write_results(tmp_path / "items.jsonl", results)
    write_labels(tmp_path / "labels.jsonl", labels)
    out = tmp_path / "agreement.json"
    code = eval_cli.main(
        [
            "agreement",
            "--results",
            str(tmp_path / "items.jsonl"),
            "--labels",
            str(tmp_path / "labels.jsonl"),
            "--out",
            str(out),
        ]
    )
    assert code == (0 if report.trusted else 1)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["correctness"]["n"] == 6 and data["kappa_threshold"] == 0.6
    assert "kappa=" in capsys.readouterr().out
