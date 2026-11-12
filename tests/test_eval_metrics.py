"""Tests for answer checks, agreement statistics and operational metrics."""

import pytest

from rag.eval.agreement import agreement, cohen_kappa, percent_agreement
from rag.eval.answers import citation_result, is_refusal, parse_citations
from rag.eval.ops import Price, latency_summary, percentile, query_cost
from rag.golden.schema import Evidence
from rag.prompts import REFUSAL
from rag.types import Chunk


class TestRefusal:
    @pytest.mark.parametrize(
        "answer",
        [
            REFUSAL,
            "  i don't know based on the provided   documents ",
            "I don't know.",
            "The context does not mention a maximum number of middleware.",
            "This is not covered in the documents.",
            "There is not enough information to answer that.",
        ],
    )
    def test_detects_refusals(self, answer: str) -> None:
        assert is_refusal(answer)

    @pytest.mark.parametrize(
        "answer",
        [
            "Use `Depends` to declare a dependency [1].",
            "FastAPI returns 422 when validation fails [2].",
            # Long, substantive answer that mentions a gap in passing is not a refusal.
            "Declare the parameter with Query(max_length=50) [1]. " * 8
            + "The documents do not mention a hard upper limit.",
        ],
    )
    def test_answers_are_not_refusals(self, answer: str) -> None:
        assert not is_refusal(answer)


def test_parse_citations_formats_and_order() -> None:
    assert parse_citations("A [2]. B [1][3]. C [2, 4]. D [10]") == [2, 1, 3, 4, 10]
    assert parse_citations("no citations, but a list [x] and code a[0]") == [0]
    assert parse_citations("plain text") == []


def _chunk(start: int, end: int) -> Chunk:
    return Chunk(f"d.md#{start}-{end}", "d.md", "x" * (end - start), start, end)


def test_citation_validity_and_precision() -> None:
    context = [_chunk(0, 100), _chunk(500, 600), _chunk(900, 1000)]
    evidence = [Evidence(doc_id="d.md", start=520, end=580, quote="x" * 60)]
    result = citation_result("Yes [2]. Also [1]. And [7].", context, evidence)
    assert result.cited == (2, 1, 7)
    assert result.valid == (2, 1)  # [7] points past the 3 passages shown
    assert result.relevant == (2,)
    assert result.validity == pytest.approx(2 / 3)
    assert result.precision == pytest.approx(1 / 2)
    none = citation_result("No citations here.", context, evidence)
    assert none.validity is None and none.precision is None


class TestAgreement:
    def test_textbook_kappa(self) -> None:
        # Classic 2x2 example: 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no (n=50).
        # p_o = 35/50 = 0.7; p_e = 0.5*0.6 + 0.5*0.4 = 0.5; kappa = 0.2/0.5 = 0.4
        a = ["y"] * 25 + ["n"] * 25
        b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
        assert percent_agreement(a, b) == pytest.approx(0.7)
        assert cohen_kappa(a, b) == pytest.approx(0.4)

    def test_perfect_and_chance(self) -> None:
        labels = ["correct", "partial", "incorrect", "correct"]
        assert cohen_kappa(labels, labels) == pytest.approx(1.0)
        # Agreement no better than chance gives kappa near 0 despite 50% agreement.
        assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)

    def test_constant_judge_has_high_agreement_but_low_kappa(self) -> None:
        human = ["correct"] * 9 + ["incorrect"]
        judge = ["correct"] * 10
        assert percent_agreement(judge, human) == pytest.approx(0.9)
        assert cohen_kappa(judge, human) == pytest.approx(0.0)

    def test_degenerate_single_label(self) -> None:
        assert cohen_kappa(["x", "x"], ["x", "x"]) == 1.0

    def test_confusion_and_errors(self) -> None:
        result = agreement(["c", "c", "i"], ["c", "i", "i"])
        assert result.confusion == {("c", "c"): 1, ("c", "i"): 1, ("i", "i"): 1}
        with pytest.raises(ValueError):
            cohen_kappa(["a"], ["a", "b"])
        with pytest.raises(ValueError):
            cohen_kappa([], [])


class TestOps:
    def test_percentile_matches_numpy_linear(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        assert percentile(values, 50) == 2.5
        assert percentile(values, 0) == 1.0 and percentile(values, 100) == 4.0
        assert percentile(list(range(1, 101)), 95) == pytest.approx(95.05)
        assert percentile([7.0], 95) == 7.0
        with pytest.raises(ValueError):
            percentile([], 50)

    def test_latency_summary(self) -> None:
        summary = latency_summary([{"retrieve_ms": 10, "generate_ms": 100}, {"retrieve_ms": 30}])
        assert summary["retrieve_ms"] == {"p50": 20.0, "p95": 29.0, "mean": 20.0, "n": 2.0}
        assert summary["generate_ms"]["n"] == 1.0

    def test_cost(self) -> None:
        prices = {"paid-model": Price(input_per_mtok=1.0, output_per_mtok=4.0)}
        assert query_cost("https://x", "paid-model", 1000, 500, prices) == pytest.approx(0.003)
        assert query_cost("ollama", "llama3", 1000, 500, {}) == 0.0
        assert query_cost("https://api.groq.com/openai/v1", "unknown", 10, 10, {}) is None
