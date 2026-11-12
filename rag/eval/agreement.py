"""Agreement between the LLM judge and a human: percent agreement and Cohen's kappa.

Percent agreement alone is misleading when one label dominates: if 90% of answers are
correct, a judge that always says "correct" agrees 90% of the time while knowing
nothing. Cohen's kappa corrects for the agreement expected by chance:

    kappa = (p_o - p_e) / (1 - p_e)

    p_o = observed agreement = fraction of items with the same label
    p_e = chance agreement   = sum over labels L of P_judge(L) * P_human(L)

kappa = 1 is perfect agreement, 0 is chance level, negative is worse than chance. A
common reading (Landis & Koch, 1977): 0.41-0.60 moderate, 0.61-0.80 substantial. The
project trusts the judge only at kappa >= 0.6.
"""

from collections import Counter
from collections.abc import Hashable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Agreement:
    n: int
    percent: float
    kappa: float
    confusion: dict[tuple[Hashable, Hashable], int]  # (judge label, human label) -> count


def percent_agreement(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    _check(a, b)
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    _check(a, b)
    n = len(a)
    p_o = percent_agreement(a, b)
    count_a, count_b = Counter(a), Counter(b)
    p_e = sum(count_a[label] * count_b[label] for label in count_a) / (n * n)
    if p_e == 1.0:
        # Both raters used one and the same label for every item: kappa is 0/0. Report
        # 1.0 since they agree completely, but such a sample cannot validate a judge.
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def agreement(judge: Sequence[Hashable], human: Sequence[Hashable]) -> Agreement:
    confusion = Counter(zip(judge, human, strict=True))
    return Agreement(
        n=len(judge),
        percent=percent_agreement(judge, human),
        kappa=cohen_kappa(judge, human),
        confusion=dict(confusion),
    )


def _check(a: Sequence[Hashable], b: Sequence[Hashable]) -> None:
    if len(a) != len(b):
        raise ValueError("label sequences must have the same length")
    if not a:
        raise ValueError("cannot compute agreement on zero items")
