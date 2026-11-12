"""Bootstrap confidence intervals, so small differences are not over-claimed.

Percentile bootstrap: resample the per-item values with replacement B times, take the
mean of each resample, and use the 2.5th and 97.5th percentiles of those means as the 95%
interval. It assumes nothing about the metric's distribution, which matters here because
most per-item values are 0/1 or a few discrete levels.

Paired bootstrap for comparing two runs: both runs answered the *same* questions, so
each resample draws question indices once and applies them to both runs. Question
difficulty then cancels out in the difference, and the interval of (B - A) is much tighter
than comparing two independent intervals. If it excludes 0, the difference is unlikely
to be resampling noise on this dataset.

Resampling is seeded, so reports are reproducible. B = 2000 by default; the Monte Carlo
error of the interval endpoints is then small next to their width.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_RESAMPLES = 2000
DEFAULT_SEED = 0


@dataclass(frozen=True, slots=True)
class Estimate:
    mean: float
    low: float
    high: float
    n: int

    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0

    def format(self, digits: int = 3) -> str:
        return f"{self.mean:.{digits}f} [{self.low:.{digits}f}, {self.high:.{digits}f}]"


def bootstrap_mean(
    values: Sequence[float],
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> Estimate:
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(data), size=(resamples, len(data)))
    means = data[indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(means, [alpha, 1 - alpha])
    return Estimate(float(data.mean()), float(low), float(high), len(data))


def paired_bootstrap_delta(
    a: Mapping[str, float],
    b: Mapping[str, float],
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> Estimate:
    """Interval for mean(B - A) over the items present in both runs."""
    common = sorted(a.keys() & b.keys())
    if not common:
        raise ValueError("the runs share no items")
    deltas = [b[item] - a[item] for item in common]
    return bootstrap_mean(deltas, resamples, confidence, seed)
