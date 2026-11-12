"""Score one pipeline answer against one golden item, and aggregate a run.

Per item:
* retrieval metrics on the full ranking (answerable items only),
* refusal: did the system refuse? Correct for unanswerable items, a false refusal otherwise,
* citations: are the cited passages real (validity) and do they hold evidence (precision),
* judge: correctness against the reference (answerable items) and faithfulness to the
  retrieved context (any non-refusal),
* operations: per-stage latency, model latency, tokens, cost.

Refusals are graded without the judge: a refusal to an answerable question is
"incorrect" by the rubric's own definition, and a refusal makes no claims to check.
"""

from collections import Counter
from collections.abc import Sequence
from typing import Any

from pydantic import Field

from rag.config import (
    GeneratorConfig,
    OllamaGeneratorConfig,
    SamplingParams,
    StrictModel,
)
from rag.eval.answers import citation_result, is_refusal
from rag.eval.judge import CORRECTNESS_SCORE, JudgeError, LLMJudge
from rag.eval.ops import Price, latency_summary, query_cost
from rag.eval.retrieval import covers, retrieval_metrics
from rag.golden.schema import GoldenItem, QuestionType
from rag.pipeline import Answer


class EvalConfig(StrictModel):
    judge: GeneratorConfig = OllamaGeneratorConfig(
        sampling=SamplingParams(temperature=0.0, max_tokens=600, json_mode=True)
    )
    min_span_coverage: float = Field(default=0.5, gt=0, le=1)
    recall_ks: tuple[int, ...] = (1, 3, 5, 10)
    ndcg_k: int = 10
    # Model name -> price. Leave empty for local models; never guess a price.
    prices: dict[str, Price] = Field(default_factory=dict)


class _Record(StrictModel):
    pass


class RetrievedRef(_Record):
    chunk_id: str
    doc_id: str
    start: int
    end: int
    score: float
    relevant: bool


class CitationRecord(_Record):
    cited: tuple[int, ...]
    valid: tuple[int, ...]
    relevant: tuple[int, ...]
    validity: float | None
    precision: float | None


class JudgeRecord(_Record):
    correctness_label: str | None = None
    correctness_reason: str = ""
    faithfulness: float | None = None
    faithful: bool | None = None
    claims: tuple[tuple[str, bool], ...] = ()
    errors: tuple[str, ...] = ()
    identity: dict[str, Any] = Field(default_factory=dict)


class OpsRecord(_Record):
    timings_ms: dict[str, float]
    model_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    cached: bool
    cost_usd: float | None


class ItemResult(_Record):
    id: str
    question_type: QuestionType
    question: str
    reference_answer: str
    answer: str
    generated: bool = True  # False in retrieval-only runs: no answer metrics then
    refused: bool
    retrieval: dict[str, float] | None
    ranking: tuple[RetrievedRef, ...]  # top 10, for inspecting and comparing runs
    context: tuple[RetrievedRef, ...]  # what the LLM saw; citation [n] is context[n-1]
    context_texts: tuple[str, ...]
    citations: CitationRecord
    judge: JudgeRecord | None
    ops: OpsRecord

    @property
    def answerable(self) -> bool:
        return self.question_type != "unanswerable"


def evaluate_item(
    item: GoldenItem,
    answer: Answer,
    cfg: EvalConfig,
    judge: LLMJudge | None,
    generator_provider: str,
    *,
    generated: bool = True,
) -> ItemResult:
    """``generated=False`` scores retrieval only: ``answer`` then carries a ranking and an
    empty generation, and no refusal, citation or judge metrics are computed."""
    evidence = item.evidence
    text = answer.generation.text if generated else ""
    refused = generated and is_refusal(text)

    def ref(sc: Any) -> RetrievedRef:
        c = sc.chunk
        relevant = any(covers(c, ev, cfg.min_span_coverage) for ev in evidence)
        return RetrievedRef(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            start=c.start,
            end=c.end,
            score=sc.score,
            relevant=relevant,
        )

    retrieval = None
    if item.question_type != "unanswerable":
        retrieval = retrieval_metrics(
            [sc.chunk for sc in answer.ranking],
            evidence,
            ks=cfg.recall_ks,
            ndcg_k=cfg.ndcg_k,
            min_fraction=cfg.min_span_coverage,
        )
    citations = citation_result(
        text, [sc.chunk for sc in answer.context], evidence, cfg.min_span_coverage
    )
    context_texts = tuple(sc.chunk.indexed_text for sc in answer.context)

    judge_record = None
    if judge is not None and generated:
        judge_record = _judge(item, text, refused, context_texts, judge)

    g = answer.generation
    return ItemResult(
        id=item.id,
        question_type=item.question_type,
        question=item.question,
        reference_answer=item.reference_answer,
        answer=text,
        generated=generated,
        refused=refused,
        retrieval=retrieval,
        ranking=tuple(ref(sc) for sc in answer.ranking[:10]),
        context=tuple(ref(sc) for sc in answer.context),
        context_texts=context_texts,
        citations=CitationRecord(
            cited=citations.cited,
            valid=citations.valid,
            relevant=citations.relevant,
            validity=citations.validity,
            precision=citations.precision,
        ),
        judge=judge_record,
        ops=OpsRecord(
            timings_ms=dict(answer.timings_ms),
            model_latency_ms=g.latency_ms,
            prompt_tokens=g.prompt_tokens,
            completion_tokens=g.completion_tokens,
            cached=g.cached,
            cost_usd=query_cost(
                generator_provider, g.model, g.prompt_tokens, g.completion_tokens, cfg.prices
            ),
        ),
    )


def _judge(
    item: GoldenItem, text: str, refused: bool, context: Sequence[str], judge: LLMJudge
) -> JudgeRecord:
    fields: dict[str, Any] = {"identity": dict(judge.identity)}
    errors: list[str] = []
    if item.question_type != "unanswerable":
        if refused:
            fields["correctness_label"] = "incorrect"
            fields["correctness_reason"] = "refused to answer an answerable question"
        else:
            try:
                verdict = judge.correctness(item.question, item.reference_answer, text)
                fields["correctness_label"] = verdict.label
                fields["correctness_reason"] = verdict.reason
            except JudgeError as exc:
                errors.append(f"correctness: {exc}")
    if not refused:
        try:
            faith = judge.faithfulness(text, context)
            fields["faithfulness"] = faith.score
            fields["faithful"] = faith.faithful
            fields["claims"] = tuple((c.claim, c.supported) for c in faith.claims)
        except JudgeError as exc:
            errors.append(f"faithfulness: {exc}")
    return JudgeRecord(errors=tuple(errors), **fields)


# --------------------------------------------------------------------------- aggregation

# Correctness labels as scores; kept as plain str keys because records store plain str.
_LABEL_SCORE: dict[str, float] = {str(label): score for label, score in CORRECTNESS_SCORE.items()}

# Headline metrics in report order. Higher is better for all of them except
# false_refusal_rate.
HEADLINE_METRICS = (
    "recall@1",
    "recall@3",
    "recall@5",
    "recall@10",
    "mrr",
    "ndcg@10",
    "correctness",
    "accuracy",
    "faithfulness",
    "refusal_accuracy",
    "false_refusal_rate",
    "citation_precision",
)
LOWER_IS_BETTER = frozenset({"false_refusal_rate"})


def per_item_values(results: Sequence[ItemResult]) -> dict[str, dict[str, float]]:
    """For each headline metric, the value of every item on which it is defined.

    This is the single source for both the means and the bootstrap intervals, so the
    two can never disagree. Keys are item IDs, which makes paired comparisons possible.
    """
    values: dict[str, dict[str, float]] = {m: {} for m in HEADLINE_METRICS}
    for r in results:
        if r.retrieval is not None:
            for key, v in r.retrieval.items():
                if key in values:
                    values[key][r.id] = v
        if not r.generated:
            continue
        if r.answerable:
            values["false_refusal_rate"][r.id] = float(r.refused)
            label = r.judge.correctness_label if r.judge else None
            if label is not None:
                values["correctness"][r.id] = _LABEL_SCORE[label]
                values["accuracy"][r.id] = float(label == "correct")
        else:
            values["refusal_accuracy"][r.id] = float(r.refused)
        if r.judge is not None and r.judge.faithfulness is not None:
            values["faithfulness"][r.id] = r.judge.faithfulness
        if r.citations.precision is not None:
            values["citation_precision"][r.id] = r.citations.precision
    return values


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def headline(results: Sequence[ItemResult]) -> dict[str, float | None]:
    """Means of the headline metrics (None where no item defines the metric)."""
    return {m: _mean(list(v.values())) for m, v in per_item_values(results).items()}


def aggregate(results: Sequence[ItemResult]) -> dict[str, Any]:
    """Run-level metrics: headline means, plus operational statistics and counts."""
    metrics: dict[str, Any] = dict(headline(results))
    generated = [r for r in results if r.generated]
    validity = [r.citations.validity for r in generated if r.citations.validity is not None]
    metrics["citation_validity"] = _mean(validity)

    costs = [r.ops.cost_usd for r in generated]
    known_costs = [c for c in costs if c is not None]
    metrics["cost_per_query_usd"] = (
        _mean(known_costs) if generated and len(known_costs) == len(costs) else None
    )
    metrics["prompt_tokens_mean"] = _mean([float(r.ops.prompt_tokens) for r in generated])
    metrics["completion_tokens_mean"] = _mean([float(r.ops.completion_tokens) for r in generated])
    stages = [
        {
            **r.ops.timings_ms,
            **({"model_latency_ms": r.ops.model_latency_ms} if r.generated else {}),
        }
        for r in results
    ]
    metrics["latency_ms"] = latency_summary(stages) if results else {}

    values = per_item_values(results)
    metrics["n"] = {
        "items": len(results),
        "answerable": sum(r.answerable for r in results),
        "unanswerable": sum(not r.answerable for r in results),
        "generated": len(generated),
        "judged_correctness": len(values["correctness"]),
        "judged_faithfulness": len(values["faithfulness"]),
        "judge_errors": sum(len(r.judge.errors) for r in results if r.judge),
        "by_type": dict(Counter(r.question_type for r in results)),
    }
    types = sorted({r.question_type for r in results})
    metrics["by_type"] = {
        q: {
            k: v
            for k, v in headline([r for r in results if r.question_type == q]).items()
            if k in ("recall@5", "mrr", "accuracy", "faithfulness", "refusal_accuracy")
        }
        for q in types
    }
    return metrics
