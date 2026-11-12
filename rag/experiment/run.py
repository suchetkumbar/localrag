"""Run one pipeline config over a golden dataset and write ``results/<run_id>/``.

    results/<run_id>/
      config.yaml    the exact pipeline config that ran
      items.jsonl    one ItemResult per question (answers, rankings, judge verdicts)
      metrics.json   run manifest, aggregate metrics, 95% bootstrap intervals
      report.md      human-readable summary

The run executes in phases rather than question by question:

    1. index the corpus           (embedder)
    2. retrieve + rerank all      (embedder, cross-encoder)
    3. free the reranker
    4. generate all answers       (LLM)
    5. judge all answers          (judge LLM)

so at most one large model is needed at a time. On a 16 GB laptop the ~1 GB ONNX
reranker plus an 8B LLM can otherwise exhaust RAM.
"""

import hashlib
import json
import platform
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from rag.cache import DiskCache
from rag.config import CacheMode, PipelineConfig
from rag.corpus import read_fingerprint
from rag.eval.evaluate import (
    HEADLINE_METRICS,
    EvalConfig,
    ItemResult,
    aggregate,
    evaluate_item,
    per_item_values,
)
from rag.eval.judge import LLMJudge
from rag.eval.labels import read_results, write_results
from rag.experiment.report import render_report
from rag.experiment.stats import bootstrap_mean
from rag.factory import build_generator, build_pipeline
from rag.golden.schema import load_golden, validate_against_corpus
from rag.loaders import load_corpus
from rag.pipeline import Answer, RetrievalResult
from rag.types import Generation

Log = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class RunOptions:
    limit: int | None = None
    retrieval_only: bool = False
    judge: bool = True
    allow_unverified: bool = False
    run_id: str | None = None
    # Cache overrides: CI replays a recorded bundle with cache_mode="readonly", and a
    # local run with record_to writes the bundle (every model call the run needs).
    cache_path: Path | None = None
    cache_mode: CacheMode | None = None
    record_to: Path | None = None


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_run_id(cfg: PipelineConfig, dataset_sha: str, options: RunOptions) -> str:
    """Deterministic: the same config on the same data maps to the same directory."""
    suffix = "-retrieval" if options.retrieval_only else ""
    limit = f"-n{options.limit}" if options.limit else ""
    return f"{cfg.name}-{cfg.fingerprint()[:8]}-{dataset_sha[:8]}{limit}{suffix}"


def git_state() -> dict[str, Any]:
    """Commit and dirty flag, read-only. Missing git is not an error."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


class _Progress:
    def __init__(self, label: str, total: int, log: Log, every: int = 10) -> None:
        self.label, self.total, self.log, self.every = label, total, log, every
        self.start = time.perf_counter()

    def tick(self, done: int) -> None:
        if done % self.every and done != self.total:
            return
        elapsed = time.perf_counter() - self.start
        eta = elapsed / done * (self.total - done) if done else 0.0
        self.log(f"  {self.label}: {done}/{self.total}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s left)")


def run_experiment(
    cfg: PipelineConfig,
    dataset: Path,
    eval_cfg: EvalConfig | None = None,
    options: RunOptions | None = None,
    results_root: Path = Path("results"),
    log: Log = print,
) -> Path:
    eval_cfg = eval_cfg or EvalConfig()
    options = options or RunOptions()
    started = datetime.now(UTC)
    t_start = time.perf_counter()

    items = load_golden(dataset, allow_unverified=options.allow_unverified)
    if options.limit:
        items = items[: options.limit]
    docs = load_corpus(cfg.corpus.path, cfg.corpus.patterns)
    corpus_fp = read_fingerprint(cfg.corpus.path)
    problems = validate_against_corpus(items, {d.doc_id: d for d in docs}, corpus_fp)
    if problems:
        raise RuntimeError("dataset does not match the corpus:\n" + "\n".join(problems[:10]))

    dataset_sha = file_sha256(dataset)
    run_id = options.run_id or make_run_id(cfg, dataset_sha, options)
    out = results_root / run_id
    log(f"run {run_id}: {len(items)} items, config {cfg.name} ({cfg.fingerprint()})")

    cache = DiskCache(
        options.cache_path or cfg.cache.path,
        options.cache_mode or cfg.cache.mode,
        record_to=options.record_to,
    )
    pipeline = build_pipeline(cfg, cache)

    log("phase 1/4: indexing")
    t0 = time.perf_counter()
    chunks = pipeline.index(docs)
    index_s = time.perf_counter() - t0
    log(f"  {len(chunks)} chunks in {index_s:.1f}s")

    log("phase 2/4: retrieval" + (" + reranking" if pipeline.reranker else ""))
    progress = _Progress("retrieved", len(items), log)
    retrievals: list[RetrievalResult] = []
    for n, item in enumerate(items, start=1):
        retrievals.append(pipeline.retrieve(item.question))
        progress.tick(n)
    close = getattr(pipeline.reranker, "close", None)
    if callable(close):
        close()  # free the cross-encoder before the LLM loads

    results: list[ItemResult]
    judge: LLMJudge | None = None
    if options.retrieval_only:
        log("phase 3/4 and 4/4 skipped (retrieval only)")
        empty = Generation("", "none", 0, 0, 0.0)
        results = [
            evaluate_item(
                item,
                Answer(item.question, r.ranking, [], empty, dict(r.timings_ms)),
                eval_cfg,
                None,
                "none",
                generated=False,
            )
            for item, r in zip(items, retrievals, strict=True)
        ]
    else:
        log("phase 3/4: generation")
        progress = _Progress("generated", len(items), log)
        answers: list[Answer] = []
        for n, (item, r) in enumerate(zip(items, retrievals, strict=True), start=1):
            answers.append(pipeline.generate(item.question, r))
            progress.tick(n)
        if options.judge:
            judge = LLMJudge(build_generator(eval_cfg.judge, cache))
        log("phase 4/4: scoring" + (" with the LLM judge" if judge else ""))
        provider = str(pipeline.generator.identity.get("provider", ""))
        progress = _Progress("scored", len(items), log)
        results = []
        for n, (item, answer) in enumerate(zip(items, answers, strict=True), start=1):
            results.append(evaluate_item(item, answer, eval_cfg, judge, provider))
            progress.tick(n)

    metrics = aggregate(results)
    intervals = {
        m: _estimate(values)
        for m, values in per_item_values(results).items()
        if values and m in HEADLINE_METRICS
    }
    manifest = {
        "run_id": run_id,
        "config_name": cfg.name,
        "config_fingerprint": cfg.fingerprint(),
        "mode": "retrieval" if options.retrieval_only else "full",
        "dataset": {
            "path": dataset.as_posix(),
            "sha256": dataset_sha,
            "items": len(items),
            "limit": options.limit,
            "all_verified": all(i.human_verified for i in items),
        },
        "corpus_fingerprint": corpus_fp,
        "chunks": len(chunks),
        "judge": dict(judge.identity) if judge else None,
        "eval": eval_cfg.model_dump(mode="json", exclude={"prices"}),
        "git": git_state(),
        "python": platform.python_version(),
        "started_at": started.isoformat(),
        "duration_s": round(time.perf_counter() - t_start, 1),
        "index_s": round(index_s, 1),
        "cache": {"hits": cache.hits, "misses": cache.misses},
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    write_results(out / "items.jsonl", results)
    payload = {"run": manifest, "metrics": metrics, "intervals": intervals}
    (out / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out / "report.md").write_text(render_report(payload, results), encoding="utf-8")
    cache.close()
    log(f"done in {manifest['duration_s']}s -> {out}")
    return out


def _estimate(values: dict[str, float]) -> dict[str, float]:
    est = bootstrap_mean(list(values.values()))
    return {"mean": est.mean, "low": est.low, "high": est.high, "n": est.n}


def load_run(run_dir: Path) -> tuple[dict[str, Any], list[ItemResult]]:
    payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return payload, read_results(run_dir / "items.jsonl")
