"""Tests for the regression gate, cache record/replay and the smoke subset."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from rag import cli
from rag.cache import JSON_CODEC, CacheMissError, DiskCache
from rag.experiment.gate import (
    COMMENT_MARKER,
    Baseline,
    GateResult,
    evaluate_gate,
    read_baseline,
    render_gate,
)
from rag.golden.review import stratified_sample
from rag.golden.schema import Review
from rag.prompts import REFUSAL
from tests.test_golden_schema import make_item

RUN: dict[str, Any] = {
    "config_name": "baseline",
    "dataset": {"sha256": "ds", "items": 10},
    "corpus_fingerprint": "fp",
    "judge": {
        "provider": "ollama",
        "model": "llama3",
        "correctness_rubric": "c@1",
        "faithfulness_rubric": "f@1",
    },
}


def _baseline(
    per_item: dict[str, dict[str, float]], judge: dict[str, Any] | None = None
) -> Baseline:
    return Baseline(
        run_id="base",
        config_fingerprint="cfg",
        dataset_sha256="ds",
        corpus_fingerprint="fp",
        judge=judge if judge is not None else RUN["judge"],
        git_commit="abc",
        created_at=datetime(2026, 11, 12, tzinfo=UTC),
        means={m: sum(v.values()) / len(v) for m, v in per_item.items()},
        per_item=per_item,
    )


def _ones(n: int) -> dict[str, float]:
    return {f"q{i:02d}": 1.0 for i in range(n)}


def _status(result: GateResult, metric: str) -> str:
    return next(r.status for r in result.rows if r.metric == metric)


class TestGateRule:
    def test_no_change_passes(self) -> None:
        values = {"recall@5": _ones(40), "faithfulness": _ones(40)}
        result = evaluate_gate(_baseline(values), values, RUN)
        assert result.passed and _status(result, "recall@5") == "pass"

    def test_clear_regression_fails(self) -> None:
        current = _ones(40) | {f"q{i:02d}": 0.0 for i in range(10)}  # 25-point drop
        result = evaluate_gate(_baseline({"recall@5": _ones(40)}), {"recall@5": current}, RUN)
        assert not result.passed and _status(result, "recall@5") == "FAIL"

    def test_drop_within_threshold_passes(self) -> None:
        current = _ones(40) | {"q00": 0.0}  # 2.5 points: below the 3-point threshold
        result = evaluate_gate(_baseline({"recall@5": _ones(40)}), {"recall@5": current}, RUN)
        assert result.passed and _status(result, "recall@5") == "pass"

    def test_large_but_noisy_drop_only_warns(self) -> None:
        # 10 points on 10 items, from a single item: the interval still includes zero.
        current = _ones(10) | {"q00": 0.0}
        result = evaluate_gate(_baseline({"recall@5": _ones(10)}), {"recall@5": current}, RUN)
        assert result.passed and _status(result, "recall@5") == "warn"

    def test_ungated_metric_never_fails(self) -> None:
        base = {"recall@5": _ones(40), "accuracy": _ones(40)}
        current = {"recall@5": _ones(40), "accuracy": {k: 0.0 for k in _ones(40)}}
        result = evaluate_gate(_baseline(base), current, RUN)
        assert result.passed and _status(result, "accuracy") == "warn"

    def test_lower_is_better_direction(self) -> None:
        base = {"false_refusal_rate": {k: 0.0 for k in _ones(40)}}
        worse = {"false_refusal_rate": _ones(40)}  # refusing everything
        result = evaluate_gate(_baseline(base), worse, RUN, gated=["false_refusal_rate"])
        assert _status(result, "false_refusal_rate") == "FAIL"
        better = evaluate_gate(_baseline(worse), base, RUN, gated=["false_refusal_rate"])
        assert _status(better, "false_refusal_rate") == "pass"

    def test_missing_gated_metric_fails(self) -> None:
        result = evaluate_gate(_baseline({"faithfulness": _ones(40)}), {"recall@5": _ones(40)}, RUN)
        assert not result.passed
        assert any("missing" in note for note in result.notes)

    def test_notes_on_changed_conditions(self) -> None:
        run = RUN | {
            "dataset": {"sha256": "other", "items": 10},
            "judge": RUN["judge"] | {"faithfulness_rubric": "f@2"},
        }
        result = evaluate_gate(_baseline({"recall@5": _ones(10)}), {"recall@5": _ones(10)}, run)
        text = " ".join(result.notes)
        assert "dataset differs" in text and "rubric changed" in text

    def test_comment_rendering(self) -> None:
        current = _ones(40) | {f"q{i:02d}": 0.0 for i in range(10)}
        base = _baseline({"recall@5": _ones(40)})
        result = evaluate_gate(base, {"recall@5": current}, RUN)
        text = render_gate(result, base, RUN)
        assert text.startswith(COMMENT_MARKER)
        assert "**FAILED**" in text and "| **recall@5** | 1.000 | 0.750 |" in text
        assert "accepted" in render_gate(result, base, RUN, accepted=True)


class TestRecordReplay:
    def test_bundle_receives_hits_and_new_entries(self, tmp_path: Path) -> None:
        main = DiskCache(tmp_path / "main.sqlite")
        main.get_or_compute("k", {"old": 1}, lambda: "cached before", JSON_CODEC)
        main.close()
        recorder = DiskCache(tmp_path / "main.sqlite", record_to=tmp_path / "bundle.sqlite")
        recorder.get_or_compute("k", {"old": 1}, lambda: "unused", JSON_CODEC)  # hit
        recorder.get_or_compute("k", {"new": 1}, lambda: "computed", JSON_CODEC)  # miss
        recorder.close()

        replay = DiskCache(tmp_path / "bundle.sqlite", mode="readonly")
        assert replay.get_or_compute("k", {"old": 1}, lambda: "x", JSON_CODEC)[0] == "cached before"
        assert replay.get_or_compute("k", {"new": 1}, lambda: "x", JSON_CODEC)[0] == "computed"
        with pytest.raises(CacheMissError):
            replay.get_or_compute("k", {"never": 1}, lambda: "x", JSON_CODEC)
        replay.close()

    def test_invalid_record_targets(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="different file"):
            DiskCache(tmp_path / "a.sqlite", record_to=tmp_path / "a.sqlite")
        with pytest.raises(ValueError, match="disabled"):
            DiskCache(tmp_path / "a.sqlite", mode="off", record_to=tmp_path / "b.sqlite")


def test_stratified_sample_keeps_proportions() -> None:
    ok = Review(status="accepted")
    items = (
        [
            make_item(id=f"f{i}", review=ok, human_verified=True, question=f"Factual question {i}?")
            for i in range(30)
        ]
        + [
            make_item(
                id=f"u{i}",
                question_type="unanswerable",
                evidence=[],
                reference_answer=REFUSAL,
                review=ok,
                human_verified=True,
                question=f"Unanswerable question {i}?",
            )
            for i in range(10)
        ]
        + [make_item(id="pending")]
    )
    sample = stratified_sample(items, 8, seed=1)
    assert len(sample) == 8
    assert sum(i.question_type == "unanswerable" for i in sample) == 2  # 10/40 of 8
    assert "pending" not in {i.id for i in sample}
    assert [i.id for i in sample] == [i.id for i in stratified_sample(items, 8, seed=1)]
    with pytest.raises(ValueError, match="only 40 are verified"):
        stratified_sample(items, 41, seed=1)


def test_gate_cli_baseline_then_replay_from_bundle(
    workspace: dict[str, Any], tmp_path: Path
) -> None:
    cfg = workspace["config"]("gate", "hybrid")
    config_path = tmp_path / "gate.yaml"
    config_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8")
    eval_path = tmp_path / "eval.yaml"
    eval_path.write_text(
        yaml.safe_dump(workspace["eval"].model_dump(mode="json")), encoding="utf-8"
    )
    baseline, bundle = tmp_path / "baseline.json", tmp_path / "bundle.sqlite"
    common = [
        "--config",
        str(config_path),
        "--dataset",
        str(workspace["dataset"]),
        "--eval-config",
        str(eval_path),
        "--baseline",
        str(baseline),
        "--results",
        str(tmp_path / "results"),
    ]
    # 1. Locally: write the baseline and record every model call into the bundle.
    assert cli.main(["gate", *common, "--write-baseline", "--record-to", str(bundle)]) == 0
    assert read_baseline(baseline).per_item["recall@5"]

    # 2. In CI: replay from the bundle alone, readonly. Same code + data -> no change.
    comment = tmp_path / "gate.md"
    replay = [
        "--cache-path",
        str(bundle),
        "--cache-mode",
        "readonly",
        "--comment-out",
        str(comment),
    ]
    assert cli.main(["gate", *common, *replay]) == 0
    text = comment.read_text(encoding="utf-8")
    assert "RAG eval gate: passed" in text and "FAIL" not in text

    # 3. A change that needs new model calls cannot silently call a model in CI.
    changed = cfg.model_copy(
        update={"generator": cfg.generator.model_copy(update={"response": "Changed."})}
    )
    config_path.write_text(yaml.safe_dump(changed.model_dump(mode="json")), encoding="utf-8")
    with pytest.raises(CacheMissError):
        cli.main(["gate", *common, *replay])
