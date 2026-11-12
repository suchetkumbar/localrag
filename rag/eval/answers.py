"""Answer checks that need no LLM: refusal detection and citation accuracy.

These are deterministic on purpose. A judge model is only used where meaning has to be
assessed (correctness, faithfulness); anything a regex or a span check can decide
exactly is decided exactly, which keeps those metrics cheap, reproducible and
explainable.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from rag.eval.retrieval import covers
from rag.golden.schema import Evidence
from rag.prompts import REFUSAL
from rag.types import Chunk

# The prompt asks for one exact refusal sentence, but models paraphrase it. These
# patterns catch common paraphrases; they are matched against the whole answer only when
# the answer is short, so a long answer that mentions a gap in passing is not a refusal.
_REFUSAL_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bI (?:do not|don't) know\b",
        r"\b(?:context|documents?|passages?) (?:does|do) not "
        r"(?:contain|mention|provide|include|say)",
        r"\bnot (?:mentioned|provided|covered|stated|specified|found) in the "
        r"(?:context|documents?|passages?)",
        r"\b(?:no|not enough) information (?:about|on|in|to)\b",
        r"\bcannot (?:find|answer)\b",
    )
]
_SHORT_ANSWER_CHARS = 300
# Text left after removing the refusal sentence; below this the answer is a refusal.
_MIN_SUBSTANCE_CHARS = 40
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def is_refusal(answer: str) -> bool:
    """True if the answer declines to answer and says nothing substantive.

    Models sometimes give a real answer and then append the refusal sentence anyway
    ("X does Y [1]. I don't know based on the provided documents."). That is an answer,
    and the judge must grade it, so the refusal sentence only counts when little else
    is left once it and the citation markers are removed.
    """
    normalised = " ".join(answer.split()).lower()
    refusal = REFUSAL.lower().rstrip(".")
    if refusal in normalised:
        rest = _CITATION.sub("", normalised.replace(refusal, ""))
        return len(rest.strip(" .,;:-\n")) < _MIN_SUBSTANCE_CHARS
    if len(normalised) > _SHORT_ANSWER_CHARS:
        return False
    return any(p.search(normalised) for p in _REFUSAL_PATTERNS)


def parse_citations(answer: str) -> list[int]:
    """Citation numbers in order of first appearance: "[1]", "[2][3]", "[1, 4]"."""
    seen: dict[int, None] = {}
    for m in _CITATION.finditer(answer):
        for number in m.group(1).split(","):
            seen.setdefault(int(number), None)
    return list(seen)


@dataclass(frozen=True, slots=True)
class CitationResult:
    cited: tuple[int, ...]
    valid: tuple[int, ...]  # cited numbers that refer to a passage that was shown
    relevant: tuple[int, ...]  # valid citations whose passage covers an evidence span

    @property
    def validity(self) -> float | None:
        return len(self.valid) / len(self.cited) if self.cited else None

    @property
    def precision(self) -> float | None:
        """Of the passages cited, the fraction that contain gold evidence."""
        return len(self.relevant) / len(self.valid) if self.valid else None


def citation_result(
    answer: str,
    context: Sequence[Chunk],
    evidence: Sequence[Evidence],
    min_fraction: float = 0.5,
) -> CitationResult:
    """Citation [n] refers to ``context[n - 1]``, the n-th passage shown to the LLM."""
    cited = parse_citations(answer)
    valid = [n for n in cited if 1 <= n <= len(context)]
    relevant = [
        n for n in valid if any(covers(context[n - 1], ev, min_fraction) for ev in evidence)
    ]
    return CitationResult(tuple(cited), tuple(valid), tuple(relevant))
