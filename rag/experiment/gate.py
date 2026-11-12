"""The CI regression gate: compare a run against the main-branch baseline.

Rule, for each gated metric (recall@5 and faithfulness):

    FAIL if  drop > threshold                 (default 3 points, i.e. 0.03)
        and  the 95% paired-bootstrap interval of the change lies entirely on the
             "worse" side of zero.

Both conditions are needed. The threshold alone would fail PRs on noise: on a 40-item
smoke set one item flipping moves recall@5 by 2.5 points, and a few flips are routine.
The interval alone would fail PRs on tiny but consistent changes nobody cares about. A
drop beyond the threshold whose interval still includes zero is reported as a warning.

The baseline file holds only numbers and item IDs (no questions), so it can live in the
public repository. CI reads it from the *base* branch, so a PR cannot pass by editing
its own baseline; intentional regressions are accepted with a PR label instead.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from rag.config import StrictModel
from rag.eval.evaluate import HEADLINE_METRICS, LOWER_IS_BETTER, ItemResult, per_item_values
from rag.experiment.stats import Estimate, paired_bootstrap_delta

GATED_METRICS = ("recall@5", "faithfulness")
DEFAULT_THRESHOLD = 0.03
COMMENT_MARKER = "<!-- rag-eval-gate -->"
Status = Literal["pass", "warn", "FAIL", "n/a"]


class Baseline(StrictModel):
    schema_version: int = 1
    run_id: str
    config_fingerprint: str
    dataset_sha256: str
    corpus_fingerprint: str | None
    judge: dict[str, Any] | None
    git_commit: str | None
    created_at: datetime
    means: dict[str, float | None]
    per_item: dict[str, dict[str, float]]  # metric -> item id -> value


def baseline_from_run(payload: Mapping[str, Any], results: Sequence[ItemResult]) -> Baseline:
    run = payload["run"]
    values = per_item_values(results)
    return Baseline(
        run_id=run["run_id"],
        config_fingerprint=run["config_fingerprint"],
        dataset_sha256=run["dataset"]["sha256"],
        corpus_fingerprint=run["corpus_fingerprint"],
        judge=run["judge"],
        git_commit=run["git"]["commit"],
        created_at=datetime.now(UTC),
        means={m: (sum(v.values()) / len(v) if v else None) for m, v in values.items()},
        per_item={m: dict(sorted(v.items())) for m, v in values.items() if v},
    )


def write_baseline(path: Path, baseline: Baseline) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = baseline.model_dump(mode="json")
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")


def read_baseline(path: Path) -> Baseline:
    return Baseline.model_validate_json(path.read_text(encoding="utf-8"))


@dataclass(frozen=True, slots=True)
class GateRow:
    metric: str
    base: float | None
    current: float | None
    delta: Estimate | None  # current - base, paired over shared items
    gated: bool
    status: Status


@dataclass(frozen=True, slots=True)
class GateResult:
    rows: tuple[GateRow, ...]
    notes: tuple[str, ...]
    threshold: float

    @property
    def passed(self) -> bool:
        return not any(r.status == "FAIL" for r in self.rows)


def _status(metric: str, delta: Estimate | None, gated: bool, threshold: float) -> Status:
    if delta is None:
        return "n/a"
    lower_better = metric in LOWER_IS_BETTER
    drop = delta.mean if lower_better else -delta.mean
    surely_worse = delta.low > 0 if lower_better else delta.high < 0
    if drop > threshold and surely_worse:
        return "FAIL" if gated else "warn"
    if drop > threshold:
        return "warn"
    return "pass"


def evaluate_gate(
    baseline: Baseline,
    current: Mapping[str, Mapping[str, float]],
    run: Mapping[str, Any],
    threshold: float = DEFAULT_THRESHOLD,
    gated: Sequence[str] = GATED_METRICS,
) -> GateResult:
    rows = []
    notes = []
    for metric in HEADLINE_METRICS:
        base_values = baseline.per_item.get(metric, {})
        cur_values = current.get(metric, {})
        delta = (
            paired_bootstrap_delta(base_values, cur_values)
            if base_values.keys() & cur_values.keys()
            else None
        )
        is_gated = metric in gated
        status = _status(metric, delta, is_gated, threshold)
        if is_gated and base_values and delta is None:
            status = "FAIL"
            notes.append(f"`{metric}` is in the baseline but missing from this run.")
        rows.append(
            GateRow(
                metric=metric,
                base=baseline.means.get(metric),
                current=sum(cur_values.values()) / len(cur_values) if cur_values else None,
                delta=delta,
                gated=is_gated,
                status=status,
            )
        )
    if run["dataset"]["sha256"] != baseline.dataset_sha256:
        notes.append("The dataset differs from the baseline's; only shared item IDs are compared.")
    if run["corpus_fingerprint"] != baseline.corpus_fingerprint:
        notes.append("The corpus differs from the baseline's.")
    base_rubrics = _rubrics(baseline.judge)
    if base_rubrics and base_rubrics != _rubrics(run["judge"]):
        notes.append(
            "The judge or its rubric changed since the baseline: judged metrics may move "
            "because the measuring stick changed, not the system."
        )
    return GateResult(rows=tuple(rows), notes=tuple(notes), threshold=threshold)


def _rubrics(judge: Mapping[str, Any] | None) -> tuple[Any, ...]:
    if not judge:
        return ()
    return tuple(
        judge.get(k) for k in ("provider", "model", "correctness_rubric", "faithfulness_rubric")
    )


def _f(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_gate(
    result: GateResult, baseline: Baseline, run: Mapping[str, Any], accepted: bool = False
) -> str:
    if result.passed:
        verdict = "passed"
    elif accepted:
        verdict = "failed, **accepted** by the `accept-regression` label"
    else:
        verdict = "**FAILED**"
    lines = [
        COMMENT_MARKER,
        f"### RAG eval gate: {verdict}",
        "",
        f"`{run['config_name']}` on {run['dataset']['items']} smoke items vs baseline "
        f"`{baseline.run_id}` (commit `{(baseline.git_commit or 'unknown')[:10]}`).",
        f"Gated: {', '.join(f'`{m}`' for m in GATED_METRICS)}. A gated metric fails if it drops"
        f" more than {result.threshold * 100:.0f} points **and** its 95% paired-bootstrap"
        " interval is entirely below zero.",
        "",
        "| Metric | Baseline | This PR | Change [95% CI] | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for r in result.rows:
        if r.base is None and r.current is None:
            continue
        name = f"**{r.metric}**" if r.gated else r.metric
        change = r.delta.format() if r.delta else "n/a"
        lines.append(f"| {name} | {_f(r.base)} | {_f(r.current)} | {change} | {r.status} |")
    if result.notes:
        lines += [""] + [f"> {note}" for note in result.notes]
    return "\n".join(lines) + "\n"
