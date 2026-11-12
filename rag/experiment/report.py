"""Markdown reports for one run, and comparisons between two runs.

Reports list item IDs, never question text: the golden dataset is kept out of git, and
reports may be committed. The full per-item detail lives in ``items.jsonl``.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rag.eval.evaluate import HEADLINE_METRICS, LOWER_IS_BETTER, ItemResult, per_item_values
from rag.experiment.stats import Estimate, paired_bootstrap_delta

# Metrics checked per item when listing what got worse between two runs.
_ITEM_METRICS = (
    "recall@5",
    "correctness",
    "faithfulness",
    "refusal_accuracy",
    "false_refusal_rate",
)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_report(payload: Mapping[str, Any], results: Sequence[ItemResult]) -> str:
    run, metrics, intervals = payload["run"], payload["metrics"], payload["intervals"]
    lines = [f"# Run `{run['run_id']}`", ""]
    if not run["dataset"]["all_verified"]:
        lines += [
            "> **Unverified dataset.** Some items have not been reviewed by a human. These",
            "> numbers are a smoke test, not results.",
            "",
        ]
    if run["mode"] == "retrieval":
        lines += ["> Retrieval-only run: no answers were generated or judged.", ""]
    git, ds, cache = run["git"], run["dataset"], run["cache"]
    dirty = " (uncommitted changes)" if git["dirty"] else ""
    corpus = (run["corpus_fingerprint"] or "n/a")[:12]
    lines += [
        f"- Config: `{run['config_name']}` (fingerprint `{run['config_fingerprint']}`)",
        f"- Dataset: `{ds['path']}` ({ds['items']} items, sha256 `{ds['sha256'][:12]}`)",
        f"- Corpus fingerprint: `{corpus}`, {run['chunks']} chunks",
        f"- Judge: {_judge_name(run['judge'])}",
        f"- Code: `{(git['commit'] or 'unknown')[:10]}`{dirty}",
        f"- Duration: {run['duration_s']} s; cache hits {cache['hits']}, misses {cache['misses']}",
        "",
        "## Headline metrics",
        "",
        "Means with 95% bootstrap confidence intervals over items.",
        "",
        "| Metric | Mean | 95% CI | n |",
        "|---|---:|---:|---:|",
    ]
    for m in HEADLINE_METRICS:
        if m in intervals:
            iv = intervals[m]
            arrow = " (lower is better)" if m in LOWER_IS_BETTER else ""
            ci = f"[{iv['low']:.3f}, {iv['high']:.3f}]"
            lines.append(f"| {m}{arrow} | {iv['mean']:.3f} | {ci} | {iv['n']} |")

    if metrics.get("by_type"):
        keys = sorted({k for v in metrics["by_type"].values() for k in v})
        lines += ["", "## By question type", "", "| Type | n | " + " | ".join(keys) + " |"]
        lines.append("|---|---:|" + "---:|" * len(keys))
        for qtype, values in metrics["by_type"].items():
            n = metrics["n"]["by_type"].get(qtype, 0)
            lines.append(
                f"| {qtype} | {n} | " + " | ".join(_fmt(values.get(k)) for k in keys) + " |"
            )

    lines += [
        "",
        "## Operations",
        "",
        "| Stage | p50 ms | p95 ms | mean ms | n |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage, s in metrics["latency_ms"].items():
        lines.append(
            f"| {stage} | {s['p50']:.0f} | {s['p95']:.0f} | {s['mean']:.0f} | {int(s['n'])} |"
        )
    lines += [
        "",
        f"- Tokens per query: prompt {_fmt(metrics['prompt_tokens_mean'], 0)},"
        f" completion {_fmt(metrics['completion_tokens_mean'], 0)}",
        f"- Cost per query (USD): {_cost(metrics['cost_per_query_usd'])}",
        f"- Judge errors: {metrics['n']['judge_errors']}",
        "",
        "Latencies of cached model calls are the original call's latency. Retrieval always runs",
        "live, but query embeddings come from the cache when available, so on a re-run",
        "retrieval latency is mostly search time.",
    ]
    lines += ["", "## Items to inspect", ""] + _inspect(results)
    return "\n".join(lines) + "\n"


def _judge_name(judge: Mapping[str, Any] | None) -> str:
    if not judge:
        return "none"
    rubrics = ", ".join(
        str(judge[k]) for k in ("correctness_rubric", "faithfulness_rubric") if k in judge
    )
    return f"`{judge.get('provider')}/{judge.get('model')}` ({rubrics})"


def _cost(value: float | None) -> str:
    if value is None:
        return "unknown (no price configured for this model)"
    return "0 (local model)" if value == 0 else f"{value:.6f}"


def _inspect(results: Sequence[ItemResult]) -> list[str]:
    missed = [r.id for r in results if r.retrieval and r.retrieval.get("recall@10") == 0]
    wrong = [
        r.id
        for r in results
        if r.judge and r.judge.correctness_label == "incorrect" and not r.refused
    ]
    false_refusals = [r.id for r in results if r.answerable and r.refused]
    hallucinated = [r.id for r in results if r.generated and not r.answerable and not r.refused]
    unfaithful = [r.id for r in results if r.judge and r.judge.faithful is False]
    groups = [
        ("No evidence in the top 10", missed),
        ("Judged incorrect", wrong),
        ("Refused an answerable question", false_refusals),
        ("Answered an unanswerable question", hallucinated),
        ("Unsupported claims", unfaithful),
    ]
    lines = []
    for title, ids in groups:
        if ids:
            shown = ", ".join(f"`{i}`" for i in ids[:20]) + (" ..." if len(ids) > 20 else "")
            lines.append(f"- {title} ({len(ids)}): {shown}")
    return lines or ["- Nothing flagged."]


# --------------------------------------------------------------------------- compare


@dataclass(frozen=True, slots=True)
class MetricDelta:
    metric: str
    a: float | None
    b: float | None
    delta: Estimate | None  # paired over items present in both runs

    @property
    def verdict(self) -> str:
        if self.delta is None or not self.delta.excludes_zero():
            return "no clear change"
        improved = (
            (self.delta.mean < 0) if self.metric in LOWER_IS_BETTER else (self.delta.mean > 0)
        )
        return "better" if improved else "WORSE"


@dataclass(frozen=True, slots=True)
class ItemRegression:
    id: str
    changes: dict[str, tuple[float, float]]  # metric -> (A, B), only metrics that got worse


def compare(
    a: Sequence[ItemResult], b: Sequence[ItemResult]
) -> tuple[list[MetricDelta], list[ItemRegression]]:
    va, vb = per_item_values(a), per_item_values(b)
    deltas = []
    for m in HEADLINE_METRICS:
        mean_a = sum(va[m].values()) / len(va[m]) if va[m] else None
        mean_b = sum(vb[m].values()) / len(vb[m]) if vb[m] else None
        delta = paired_bootstrap_delta(va[m], vb[m]) if va[m].keys() & vb[m].keys() else None
        if mean_a is not None or mean_b is not None:
            deltas.append(MetricDelta(m, mean_a, mean_b, delta))

    regressions = []
    for item_id in sorted({r.id for r in a} & {r.id for r in b}):
        changes = {}
        for m in _ITEM_METRICS:
            if item_id in va[m] and item_id in vb[m]:
                before, after = va[m][item_id], vb[m][item_id]
                worse = after > before if m in LOWER_IS_BETTER else after < before
                if worse:
                    changes[m] = (before, after)
        if changes:
            regressions.append(ItemRegression(item_id, changes))
    regressions.sort(key=lambda r: (-len(r.changes), r.id))
    return deltas, regressions


def render_compare(
    name_a: str,
    name_b: str,
    deltas: Sequence[MetricDelta],
    regressions: Sequence[ItemRegression],
    config_diff: Mapping[str, tuple[Any, Any]],
    limit: int = 30,
) -> str:
    lines = [f"# Compare `{name_a}` (A) -> `{name_b}` (B)", ""]
    if config_diff:
        lines += ["| Config key | A | B |", "|---|---|---|"]
        lines += [f"| `{k}` | {_fmt(x)} | {_fmt(y)} |" for k, (x, y) in config_diff.items()]
        lines.append("")
    lines += [
        "Delta is B - A, paired over shared items, with a 95% bootstrap interval. A change",
        "counts only if its interval excludes zero.",
        "",
        "| Metric | A | B | Delta [95% CI] | Verdict |",
        "|---|---:|---:|---:|---|",
    ]
    for d in deltas:
        delta = d.delta.format() if d.delta else "n/a"
        lines.append(f"| {d.metric} | {_fmt(d.a)} | {_fmt(d.b)} | {delta} | {d.verdict} |")
    lines += ["", f"## Items that got worse ({len(regressions)})", ""]
    for r in regressions[:limit]:
        detail = ", ".join(f"{m} {x:.2f} -> {y:.2f}" for m, (x, y) in r.changes.items())
        lines.append(f"- `{r.id}`: {detail}")
    if len(regressions) > limit:
        lines.append(f"- ... and {len(regressions) - limit} more")
    if not regressions:
        lines.append("- None.")
    return "\n".join(lines) + "\n"


def config_diff(
    a: Mapping[str, Any], b: Mapping[str, Any], prefix: str = ""
) -> dict[str, tuple[Any, Any]]:
    """Flattened keys whose values differ between two config dicts."""
    diff: dict[str, tuple[Any, Any]] = {}
    for key in sorted(a.keys() | b.keys()):
        x, y = a.get(key), b.get(key)
        path = f"{prefix}{key}"
        if isinstance(x, Mapping) and isinstance(y, Mapping):
            diff.update(config_diff(x, y, path + "."))
        elif x != y:
            diff[path] = (x, y)
    return diff
