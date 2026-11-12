"""LLM-as-judge with fixed, versioned rubrics.

Each rubric is one file in ``rag/eval/prompts/`` (``### system`` and ``### user``
sections). Its version hash is stored with every verdict, so judge scores from runs with
different rubrics are never compared by accident. The judge is a normal ``Generator``,
so it is cached like any other model call and runs locally by default.

Why a rubric judge rather than embedding similarity to the reference: similarity rewards
answers that *sound like* the reference. "The default port is 8000" and "The default port
is 8080" are nearly identical as embeddings but one is wrong. A rubric that names the
failure modes (missing facts, contradictions, wrong values) grades what matters, and its
labels can be checked against a human with Cohen's kappa.
"""

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rag.interfaces import Generator
from rag.llm_json import NoJsonObjectError, extract_json_object

PROMPTS_DIR = Path(__file__).parent / "prompts"
CorrectnessLabel = Literal["correct", "partial", "incorrect"]
CORRECTNESS_SCORE: dict[CorrectnessLabel, float] = {
    "correct": 1.0,
    "partial": 0.5,
    "incorrect": 0.0,
}
_SECTION = re.compile(r"^### (system|user)\s*$", re.MULTILINE)


class JudgeError(ValueError):
    """The judge's response could not be parsed into a verdict."""


@dataclass(frozen=True, slots=True)
class Rubric:
    name: str
    system: str
    user: str

    @property
    def version(self) -> str:
        digest = hashlib.sha256(f"{self.system}\x00{self.user}".encode()).hexdigest()[:12]
        return f"{self.name}@{digest}"


def load_rubric(name: str, directory: Path = PROMPTS_DIR) -> Rubric:
    text = (directory / f"{name}.txt").read_text(encoding="utf-8")
    parts = _SECTION.split(text)
    sections = dict(zip(parts[1::2], (p.strip() for p in parts[2::2]), strict=True))
    if set(sections) != {"system", "user"}:
        raise ValueError(f"rubric {name!r} needs '### system' and '### user' sections")
    return Rubric(name=name, system=sections["system"], user=sections["user"])


@dataclass(frozen=True, slots=True)
class CorrectnessVerdict:
    label: CorrectnessLabel
    reason: str
    rubric: str

    @property
    def score(self) -> float:
        return CORRECTNESS_SCORE[self.label]


@dataclass(frozen=True, slots=True)
class Claim:
    claim: str
    supported: bool


@dataclass(frozen=True, slots=True)
class FaithfulnessVerdict:
    claims: tuple[Claim, ...]
    rubric: str

    @property
    def score(self) -> float | None:
        """Fraction of claims supported by the context; None if there are no claims."""
        if not self.claims:
            return None
        return sum(c.supported for c in self.claims) / len(self.claims)

    @property
    def faithful(self) -> bool | None:
        """Binary label for agreement with a human: every claim supported."""
        return None if not self.claims else all(c.supported for c in self.claims)


def _json(raw: str) -> dict[str, Any]:
    try:
        return extract_json_object(raw)
    except NoJsonObjectError as exc:
        raise JudgeError(f"judge returned no valid JSON: {raw[:200]!r}") from exc


def parse_correctness(raw: str, rubric: str) -> CorrectnessVerdict:
    obj = _json(raw)
    raw_label = str(obj.get("label", "")).strip().lower()
    for label in CORRECTNESS_SCORE:
        if label == raw_label:
            return CorrectnessVerdict(label=label, reason=str(obj.get("reason", "")), rubric=rubric)
    raise JudgeError(f"unknown correctness label {raw_label!r}")


def parse_faithfulness(raw: str, rubric: str) -> FaithfulnessVerdict:
    obj = _json(raw)
    claims_raw = obj.get("claims")
    if not isinstance(claims_raw, list):
        raise JudgeError("faithfulness verdict has no 'claims' list")
    claims = []
    for c in claims_raw:
        if not isinstance(c, Mapping) or not isinstance(c.get("supported"), bool):
            raise JudgeError(f"malformed claim: {c!r}")
        claims.append(Claim(claim=str(c.get("claim", "")), supported=c["supported"]))
    return FaithfulnessVerdict(claims=tuple(claims), rubric=rubric)


class LLMJudge:
    def __init__(
        self,
        generator: Generator,
        correctness: Rubric | None = None,
        faithfulness: Rubric | None = None,
    ) -> None:
        self.generator = generator
        self.correctness_rubric = correctness or load_rubric("correctness_v1")
        self.faithfulness_rubric = faithfulness or load_rubric("faithfulness_v1")

    @property
    def identity(self) -> Mapping[str, Any]:
        return {
            **self.generator.identity,
            "correctness_rubric": self.correctness_rubric.version,
            "faithfulness_rubric": self.faithfulness_rubric.version,
        }

    def correctness(self, question: str, reference: str, answer: str) -> CorrectnessVerdict:
        r = self.correctness_rubric
        user = r.user.format(question=question, reference=reference, answer=answer)
        return parse_correctness(self.generator.generate(r.system, user).text, r.version)

    def faithfulness(self, answer: str, context: Sequence[str]) -> FaithfulnessVerdict:
        r = self.faithfulness_rubric
        passages = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(context, start=1))
        user = r.user.format(context=passages or "(no passages)", answer=answer)
        return parse_faithfulness(self.generator.generate(r.system, user).text, r.version)
