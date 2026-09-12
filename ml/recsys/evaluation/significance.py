"""Significance testing for model comparisons and A/B experiments.

Two models differing by 0.002 NDCG on 1,500 sampled users have not been shown
to differ at all. Reporting that difference as "model A wins" is how teams ship
a model that is no better than the one it replaced - and then spend a quarter
explaining why the online metrics did not move.

Two tools:

* **Paired bootstrap** for offline comparisons. Paired because both models are
  scored on the *same* users, so the user-to-user variance - which is far
  larger than the model-to-model difference - cancels. An unpaired test on
  this data would need an order of magnitude more users to detect the same
  effect.
* **Two-proportion z-test** for online rate metrics (CTR, conversion), which
  is what the A/B framework reports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ComparisonResult:
    """Outcome of a paired comparison between two models."""

    baseline: str
    candidate: str
    metric: str
    baseline_mean: float
    candidate_mean: float
    difference: float
    relative: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int

    @property
    def significant(self) -> bool:
        """True when the confidence interval excludes zero."""
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def verdict(self) -> str:
        if not self.significant:
            return "no significant difference"
        return "candidate better" if self.difference > 0 else "candidate worse"

    def summary(self) -> str:
        return (
            f"{self.candidate} vs {self.baseline} on {self.metric}: "
            f"{self.candidate_mean:.5f} vs {self.baseline_mean:.5f} "
            f"(diff {self.difference:+.5f}, {self.relative:+.1%}, "
            f"95% CI [{self.ci_low:+.5f}, {self.ci_high:+.5f}], "
            f"p={self.p_value:.4f}) - {self.verdict()}"
        )


def paired_bootstrap(
    baseline_scores: dict[int, float],
    candidate_scores: dict[int, float],
    *,
    baseline_name: str = "baseline",
    candidate_name: str = "candidate",
    metric: str = "ndcg@10",
    iterations: int = 5000,
    seed: int = 42,
    alpha: float = 0.05,
) -> ComparisonResult:
    """Bootstrap confidence interval on the paired per-user difference."""
    shared = sorted(set(baseline_scores) & set(candidate_scores))
    if not shared:
        raise ValueError("the two models share no evaluated users")

    baseline = np.array([baseline_scores[u] for u in shared], dtype=float)
    candidate = np.array([candidate_scores[u] for u in shared], dtype=float)
    differences = candidate - baseline

    rng = np.random.default_rng(seed)
    n = len(differences)
    indices = rng.integers(0, n, size=(iterations, n))
    means = differences[indices].mean(axis=1)

    ci_low, ci_high = np.quantile(means, [alpha / 2, 1.0 - alpha / 2])

    # Two-sided bootstrap p-value: how often a resampled mean lands on the
    # other side of zero. The +1 smoothing avoids reporting p = 0, which would
    # overstate certainty that the resampling never actually established.
    tail = (
        float((means <= 0).sum())
        if differences.mean() >= 0
        else float((means >= 0).sum())
    )
    p_value = min(1.0, 2.0 * (tail + 1.0) / (iterations + 1.0))

    baseline_mean = float(baseline.mean())
    difference = float(differences.mean())
    return ComparisonResult(
        baseline=baseline_name,
        candidate=candidate_name,
        metric=metric,
        baseline_mean=baseline_mean,
        candidate_mean=float(candidate.mean()),
        difference=difference,
        relative=difference / baseline_mean if baseline_mean else float("nan"),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=p_value,
        n=n,
    )


@dataclass(slots=True)
class ProportionTest:
    """Two-proportion z-test result for an online rate metric."""

    control_rate: float
    treatment_rate: float
    absolute_lift: float
    relative_lift: float
    z_score: float
    p_value: float
    ci_low: float
    ci_high: float
    control_n: int
    treatment_n: int

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def summary(self) -> str:
        return (
            f"control {self.control_rate:.4%} (n={self.control_n:,}) vs "
            f"treatment {self.treatment_rate:.4%} (n={self.treatment_n:,}): "
            f"{self.relative_lift:+.2%} relative, "
            f"95% CI [{self.ci_low:+.4%}, {self.ci_high:+.4%}], p={self.p_value:.4f}"
        )


def two_proportion_test(
    control_successes: int,
    control_total: int,
    treatment_successes: int,
    treatment_total: int,
) -> ProportionTest:
    """Compare two conversion rates.

    Used by the A/B framework for CTR, add-to-cart rate and conversion rate.
    Revenue per user is *not* tested this way - it is a continuous,
    heavily-skewed quantity where a proportion test does not apply and a
    bootstrap is the right tool.
    """
    if control_total <= 0 or treatment_total <= 0:
        raise ValueError("both arms need at least one observation")

    p1 = control_successes / control_total
    p2 = treatment_successes / treatment_total
    pooled = (control_successes + treatment_successes) / (control_total + treatment_total)

    standard_error = math.sqrt(
        pooled * (1 - pooled) * (1 / control_total + 1 / treatment_total)
    )
    z = (p2 - p1) / standard_error if standard_error > 0 else 0.0
    p_value = 2.0 * (1.0 - _normal_cdf(abs(z)))

    # The confidence interval uses unpooled standard error: the pooled estimate
    # assumes the null hypothesis, which is right for the test statistic and
    # wrong for an interval around the observed difference.
    se_unpooled = math.sqrt(
        p1 * (1 - p1) / control_total + p2 * (1 - p2) / treatment_total
    )
    margin = 1.959963985 * se_unpooled

    return ProportionTest(
        control_rate=p1,
        treatment_rate=p2,
        absolute_lift=p2 - p1,
        relative_lift=(p2 - p1) / p1 if p1 > 0 else float("nan"),
        z_score=z,
        p_value=p_value,
        ci_low=(p2 - p1) - margin,
        ci_high=(p2 - p1) + margin,
        control_n=control_total,
        treatment_n=treatment_total,
    )


def sample_ratio_mismatch(
    observed: dict[str, int], expected_allocation: dict[str, float]
) -> tuple[float, bool]:
    """Chi-square check that traffic split as intended.

    The single most valuable guardrail in an experimentation system. If
    assignment is skewed - a bug, a caching layer, a crawler landing in one arm
    - then every result is invalid, and the effect usually looks like a
    convincing win. Checking this *before* reading results is what stops a
    broken experiment from shipping.
    """
    total = sum(observed.values())
    if total == 0:
        return 1.0, False

    statistic = 0.0
    for variant, share in expected_allocation.items():
        expected = total * share
        if expected <= 0:
            continue
        statistic += (observed.get(variant, 0) - expected) ** 2 / expected

    degrees = max(len(expected_allocation) - 1, 1)
    p_value = _chi_square_sf(statistic, degrees)
    return p_value, p_value < 0.001


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _chi_square_sf(statistic: float, degrees: int) -> float:
    """Survival function via the regularised upper incomplete gamma."""
    try:
        from scipy import stats

        return float(stats.chi2.sf(statistic, degrees))
    except ImportError:  # pragma: no cover
        if degrees == 1:
            return 2.0 * (1.0 - _normal_cdf(math.sqrt(max(statistic, 0.0))))
        return float("nan")


__all__ = [
    "ComparisonResult",
    "ProportionTest",
    "paired_bootstrap",
    "sample_ratio_mismatch",
    "two_proportion_test",
]
