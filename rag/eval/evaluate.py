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
) -> ItemResult:
    evidence = item.evidence
    text = answer.generation.text
    refused = is_refusal(text)

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
    if judge is not None:
        judge_record = _judge(item, text, refused, context_texts, judge)

    g = answer.generation
    return ItemResult(
        id=item.id,
        question_type=item.question_type,
        question=item.question,
        reference_answer=item.reference_answer,
        answer=text,
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


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _correctness_labels(results: Sequence[ItemResult]) -> list[str]:
    return [
        r.judge.correctness_label
        for r in results
        if r.answerable and r.judge is not None and r.judge.correctness_label is not None
    ]


def _faithfulness(results: Sequence[ItemResult]) -> list[float]:
    return [
        r.judge.faithfulness
        for r in results
        if r.judge is not None and r.judge.faithfulness is not None
    ]


def _recall(results: Sequence[ItemResult], key: str) -> list[float]:
    return [r.retrieval[key] for r in results if r.retrieval is not None and key in r.retrieval]


def headline(results: Sequence[ItemResult]) -> dict[str, float | None]:
    """The headline metrics, also used for per-question-type breakdowns."""
    labels = _correctness_labels(results)
    return {
        "recall@5": _mean(_recall(results, "recall@5")),
        "accuracy": _mean([float(label == "correct") for label in labels]),
        "faithfulness": _mean(_faithfulness(results)),
        "refused": _mean([float(r.refused) for r in results]),
    }


def aggregate(results: Sequence[ItemResult]) -> dict[str, Any]:
    """Run-level metrics. Every value is a mean over the items where it is defined, and
    the ``n`` of each is reported alongside so small denominators are visible."""
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    labels = _correctness_labels(results)
    faith = _faithfulness(results)
    validity = [r.citations.validity for r in results if r.citations.validity is not None]
    precision = [r.citations.precision for r in results if r.citations.precision is not None]
    costs = [r.ops.cost_usd for r in results]

    metrics: dict[str, Any] = {}
    for key in sorted({k for r in answerable if r.retrieval for k in r.retrieval}):
        metrics[key] = _mean(_recall(answerable, key))
    metrics["correctness"] = _mean([_LABEL_SCORE[label] for label in labels])
    metrics["accuracy"] = _mean([float(label == "correct") for label in labels])
    metrics["faithfulness"] = _mean(faith)
    metrics["refusal_accuracy"] = _mean([float(r.refused) for r in unanswerable])
    metrics["false_refusal_rate"] = _mean([float(r.refused) for r in answerable])
    metrics["citation_validity"] = _mean(validity)
    metrics["citation_precision"] = _mean(precision)
    known_costs = [c for c in costs if c is not None]
    metrics["cost_per_query_usd"] = _mean(known_costs) if len(known_costs) == len(costs) else None
    metrics["prompt_tokens_mean"] = _mean([float(r.ops.prompt_tokens) for r in results])
    metrics["completion_tokens_mean"] = _mean([float(r.ops.completion_tokens) for r in results])
    stages = [{**r.ops.timings_ms, "model_latency_ms": r.ops.model_latency_ms} for r in results]
    metrics["latency_ms"] = latency_summary(stages) if results else {}
    metrics["n"] = {
        "items": len(results),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "judged_correctness": len(labels),
        "judged_faithfulness": len(faith),
        "judge_errors": sum(len(r.judge.errors) for r in results if r.judge),
        "by_type": dict(Counter(r.question_type for r in results)),
    }
    types = sorted({r.question_type for r in results})
    metrics["by_type"] = {q: headline([r for r in results if r.question_type == q]) for q in types}
    return metrics
