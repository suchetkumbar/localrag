"""Tests for bootstrap statistics, the matrix, and end-to-end experiment runs."""

import json
from pathlib import Path
from typing import Any

import pytest

from rag import cli
from rag.config import PipelineConfig
from rag.corpus import FINGERPRINT_FILE
from rag.eval.evaluate import EvalConfig
from rag.experiment.matrix import README_END, README_START, matrix_configs, write_summary
from rag.experiment.report import compare, config_diff
from rag.experiment.run import RunOptions, load_run, run_experiment
from rag.experiment.stats import bootstrap_mean, paired_bootstrap_delta
from rag.golden.schema import Evidence, GoldenItem, Provenance, Review, write_jsonl
from rag.prompts import REFUSAL

# --------------------------------------------------------------------------- statistics


class TestBootstrap:
    def test_constant_sample_has_zero_width(self) -> None:
        est = bootstrap_mean([0.5] * 20)
        assert (est.mean, est.low, est.high, est.n) == (0.5, 0.5, 0.5, 20)

    def test_interval_contains_mean_and_is_reproducible(self) -> None:
        values = [0.0, 1.0] * 50
        a, b = bootstrap_mean(values, seed=3), bootstrap_mean(values, seed=3)
        assert a == b
        assert a.low < 0.5 < a.high
        # Standard error of a mean of 100 fair coin flips is 0.05: CI roughly +/- 0.1.
        assert 0.07 < 0.5 - a.low < 0.13 and 0.07 < a.high - 0.5 < 0.13

    def test_smaller_samples_give_wider_intervals(self) -> None:
        small = bootstrap_mean([0.0, 1.0] * 5)
        large = bootstrap_mean([0.0, 1.0] * 200)
        assert (small.high - small.low) > 2 * (large.high - large.low)

    def test_paired_delta(self) -> None:
        a = {f"q{i}": float(i % 2) for i in range(40)}
        same = paired_bootstrap_delta(a, a)
        assert (same.mean, same.low, same.high) == (0.0, 0.0, 0.0)
        assert not same.excludes_zero()
        shifted = paired_bootstrap_delta(a, {k: v + 0.1 for k, v in a.items()})
        assert shifted.mean == pytest.approx(0.1) and shifted.excludes_zero()
        # Only shared items are compared.
        partial = paired_bootstrap_delta(a, {"q1": 0.0})
        assert partial.n == 1 and partial.mean == -1.0

    def test_errors(self) -> None:
        with pytest.raises(ValueError):
            bootstrap_mean([])
        with pytest.raises(ValueError, match="share no items"):
            paired_bootstrap_delta({"a": 1.0}, {"b": 1.0})


# --------------------------------------------------------------------------- matrix


def test_matrix_has_twelve_comparable_variants() -> None:
    base = PipelineConfig.model_validate({"name": "base", "generator": {"kind": "fake"}})
    variants = matrix_configs(base)
    assert len(variants) == 12
    assert len({v.name for v in variants}) == 12 and len({v.fingerprint() for v in variants}) == 12
    assert sum(v.reranker is not None for v in variants) == 6
    assert {v.retrieval.retriever.kind for v in variants} == {"dense", "bm25", "hybrid"}
    sizes = {
        getattr(v.chunker, "size", None) or getattr(v.chunker, "max_size", None) for v in variants
    }
    assert sizes == {1000}  # both chunkers capped at the same size
    assert all(v.generator.kind == "fake" for v in variants)  # base settings kept


def test_config_diff() -> None:
    a = {"name": "a", "chunker": {"kind": "fixed", "size": 1000}, "reranker": None}
    b = {"name": "b", "chunker": {"kind": "heading", "size": 1000}, "reranker": {"kind": "x"}}
    assert config_diff(a, b) == {
        "chunker.kind": ("fixed", "heading"),
        "name": ("a", "b"),
        "reranker": (None, {"kind": "x"}),
    }


# --------------------------------------------------------------------------- end to end

SECTION = (
    "## Ports\n\nThe development server listens on port 8000 by default. "
    "You can change it with the --port option.\n\n"
    "## Validation\n\nInvalid request bodies return status code 422 with details of every error.\n"
)


def _item(item_id: str, doc: str, quote: str, verified: bool = True) -> GoldenItem:
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


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, Any]:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    doc = "# Server\n\n" + SECTION
    (corpus / "ports.md").write_text(doc, encoding="utf-8")
    (corpus / FINGERPRINT_FILE).write_text("fp\n", encoding="utf-8")
    items = [
        _item("q1", doc, "The development server listens on port 8000 by default."),
        _item(
            "q2", doc, "Invalid request bodies return status code 422 with details of every error."
        ),
        GoldenItem(
            id="u1",
            question="Does FastAPI ship a GraphQL server?",
            reference_answer=REFUSAL,
            question_type="unanswerable",
            human_verified=True,
            review=Review(status="accepted"),
            provenance=Provenance(drafted_by="t", draft_prompt="t"),
            corpus_fingerprint="fp",
        ),
    ]
    dataset = tmp_path / "v1.jsonl"
    write_jsonl(dataset, items)

    def config(name: str, retriever: str) -> PipelineConfig:
        return PipelineConfig.model_validate(
            {
                "name": name,
                "corpus": {"path": str(corpus)},
                "chunker": {"kind": "heading", "max_size": 300, "overlap": 50},
                "embedder": {"kind": "fake"},
                "retrieval": {"retriever": {"kind": retriever}, "candidate_k": 10, "top_k": 2},
                "generator": {"kind": "fake", "response": "Port 8000 [1]."},
                "cache": {"path": str(tmp_path / "cache.sqlite")},
            }
        )

    judge_json = json.dumps(
        {"reason": "ok", "label": "correct", "claims": [{"claim": "port 8000", "supported": True}]}
    )
    eval_cfg = EvalConfig.model_validate({"judge": {"kind": "fake", "response": judge_json}})
    return {"tmp": tmp_path, "dataset": dataset, "config": config, "eval": eval_cfg, "doc": doc}


def test_full_run_writes_all_outputs(workspace: dict[str, Any]) -> None:
    out = run_experiment(
        workspace["config"]("bm25", "bm25"),
        workspace["dataset"],
        workspace["eval"],
        results_root=workspace["tmp"] / "results",
        log=lambda _m: None,
    )
    assert {p.name for p in out.iterdir()} == {
        "config.yaml",
        "items.jsonl",
        "metrics.json",
        "report.md",
    }
    payload, items = load_run(out)
    run, metrics, intervals = payload["run"], payload["metrics"], payload["intervals"]
    assert run["mode"] == "full" and run["dataset"]["all_verified"] and run["dataset"]["items"] == 3
    assert run["judge"]["correctness_rubric"].startswith("correctness_v1@")
    assert metrics["n"]["answerable"] == 2 and metrics["n"]["unanswerable"] == 1
    assert metrics["recall@5"] == 1.0  # tiny corpus: evidence is always retrieved
    assert metrics["refusal_accuracy"] == 0.0  # the fake model never refuses
    assert intervals["recall@5"]["mean"] == metrics["recall@5"]  # same source of truth
    assert len(items) == 3
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "| recall@5 |" in report and "Answered an unanswerable question (1): `u1`" in report
    assert "Question about" not in report  # no question text in committed reports


def test_retrieval_only_run_has_no_answer_metrics(workspace: dict[str, Any]) -> None:
    out = run_experiment(
        workspace["config"]("dense", "dense"),
        workspace["dataset"],
        workspace["eval"],
        RunOptions(retrieval_only=True),
        results_root=workspace["tmp"] / "results",
        log=lambda _m: None,
    )
    payload, items = load_run(out)
    assert out.name.endswith("-retrieval") and payload["run"]["mode"] == "retrieval"
    assert payload["metrics"]["refusal_accuracy"] is None  # not 0: nothing was generated
    assert payload["metrics"]["correctness"] is None
    assert "recall@5" in payload["intervals"] and "faithfulness" not in payload["intervals"]
    assert all(not r.generated for r in items)


def test_run_refuses_unverified_or_mismatched_data(
    workspace: dict[str, Any], tmp_path: Path
) -> None:
    unverified = tmp_path / "drafts.jsonl"
    write_jsonl(
        unverified,
        [
            _item(
                "d1", workspace["doc"], "You can change it with the --port option.", verified=False
            )
        ],
    )
    cfg = workspace["config"]("x", "bm25")
    with pytest.raises(ValueError, match="unverified"):
        run_experiment(cfg, unverified, workspace["eval"], log=lambda _m: None)
    out = run_experiment(
        cfg,
        unverified,
        workspace["eval"],
        RunOptions(allow_unverified=True),
        results_root=tmp_path / "results",
        log=lambda _m: None,
    )
    assert "Unverified dataset" in (out / "report.md").read_text(encoding="utf-8")

    (tmp_path / "corpus" / FINGERPRINT_FILE).write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match the corpus"):
        run_experiment(cfg, workspace["dataset"], workspace["eval"], log=lambda _m: None)


def test_compare_and_cli(workspace: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    results = workspace["tmp"] / "results"
    a = run_experiment(
        workspace["config"]("a", "bm25"),
        workspace["dataset"],
        workspace["eval"],
        results_root=results,
        log=lambda _m: None,
    )
    b = run_experiment(
        workspace["config"]("b", "hybrid"),
        workspace["dataset"],
        workspace["eval"],
        results_root=results,
        log=lambda _m: None,
    )
    deltas, regressions = compare(load_run(a)[1], load_run(b)[1])
    assert {d.metric for d in deltas} >= {"recall@5", "mrr", "accuracy"}
    assert all(d.delta is not None for d in deltas)
    out = workspace["tmp"] / "cmp.md"
    assert cli.main(["compare", a.name, b.name, "--results", str(results), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "`retrieval.retriever.kind` | bm25 | hybrid" in text
    assert "Items that got worse" in text
    assert "Delta [95% CI]" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="no run found"):
        cli.main(["compare", "missing-run", b.name, "--results", str(results)])


def test_summary_updates_readme_only_when_publishable(workspace: dict[str, Any]) -> None:
    results = workspace["tmp"] / "results"
    full = run_experiment(
        workspace["config"]("a", "bm25"),
        workspace["dataset"],
        workspace["eval"],
        results_root=results,
        log=lambda _m: None,
    )
    retrieval = run_experiment(
        workspace["config"]("b", "bm25"),
        workspace["dataset"],
        workspace["eval"],
        RunOptions(retrieval_only=True),
        results_root=results,
        log=lambda _m: None,
    )
    readme = workspace["tmp"] / "README.md"
    original = f"# Project\n{README_START}\nplaceholder\n{README_END}\n"
    readme.write_text(original, encoding="utf-8")

    assert write_summary([full, retrieval], results / "m.md", readme) is False
    assert readme.read_text(encoding="utf-8") == original
    assert "Not publishable" in (results / "m.md").read_text(encoding="utf-8")

    assert write_summary([full], results / "m.md", readme) is True
    text = readme.read_text(encoding="utf-8")
    assert "placeholder" not in text and "| [`a`](" in text and "Source:" in text
