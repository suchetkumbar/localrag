"""Operational statistics: latency percentiles, token usage and cost per query."""

import math
from collections.abc import Iterable, Mapping, Sequence

from pydantic import Field

from rag.config import StrictModel


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (the same definition as numpy's default).

    The q-th percentile sits at fractional index (n - 1) * q / 100 of the sorted
    values; between two values it interpolates linearly. p50 of [1, 2, 3, 4] is 2.5.
    """
    if not values:
        raise ValueError("percentile of an empty sequence")
    if not 0 <= q <= 100:
        raise ValueError("q must be between 0 and 100")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def latency_summary(per_item: Iterable[Mapping[str, float]]) -> dict[str, dict[str, float]]:
    """p50 / p95 / mean per stage, e.g. {"generate_ms": {"p50": ..., "p95": ..., ...}}."""
    by_stage: dict[str, list[float]] = {}
    for timings in per_item:
        for stage, ms in timings.items():
            by_stage.setdefault(stage, []).append(ms)
    return {
        stage: {
            "p50": percentile(values, 50),
            "p95": percentile(values, 95),
            "mean": sum(values) / len(values),
            "n": float(len(values)),
        }
        for stage, values in sorted(by_stage.items())
    }


class Price(StrictModel):
    """USD per million tokens. Copy the numbers from the provider's pricing page."""

    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)


def query_cost(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    prices: Mapping[str, Price],
) -> float | None:
    """Estimated USD cost of one generation, or None if the price is unknown.

    Local Ollama models cost 0 (electricity aside). For anything else the price must be
    configured explicitly; it is never guessed. A free tier is configured as 0.
    """
    price = prices.get(model)
    if price is None:
        return 0.0 if provider == "ollama" else None
    return (
        prompt_tokens * price.input_per_mtok + completion_tokens * price.output_per_mtok
    ) / 1_000_000
