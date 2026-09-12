"""The promotion gate (ADR-010, FR-19).

A retrained model does not become the production model because it finished
training. It becomes production only if it beats the incumbent on the metrics
that matter *and* does not regress the ones that are easy to sacrifice.

That second half is the point. Optimising NDCG alone has an obvious degenerate
solution - recommend the same popular items to everyone - so coverage and
diversity are guardrails with hard thresholds, not footnotes in a report
(R-06). A candidate that wins NDCG by collapsing catalogue coverage is
rejected, and the rejection reason is recorded.

The gate is deliberately boring and deterministic: same inputs, same verdict,
with every comparison written down. A promotion decision that cannot be
explained afterwards is not a decision, it is an accident.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from enum import StrEnum

logger = logging.getLogger(__name__)


class GateOutcome(StrEnum):
    PROMOTE = "promote"
    REJECT = "reject"
    PROMOTE_FIRST = "promote_first"


@dataclass(frozen=True, slots=True)
class GateCriterion:
    """One condition a candidate must satisfy."""

    metric: str
    #: Minimum improvement over the incumbent, as an absolute difference.
    #: Negative values allow a regression up to that size.
    min_delta: float
    #: True when higher is better.
    higher_is_better: bool = True
    #: Absolute floor the candidate must clear regardless of the incumbent.
    floor: float | None = None
    required: bool = True

    def evaluate(
        self, candidate: float | None, incumbent: float | None
    ) -> tuple[bool, str]:
        if candidate is None:
            return False, f"{self.metric}: candidate did not report this metric"

        if self.floor is not None:
            if self.higher_is_better and candidate < self.floor:
                return False, (
                    f"{self.metric}: {candidate:.5f} is below the absolute floor "
                    f"{self.floor:.5f}"
                )
            if not self.higher_is_better and candidate > self.floor:
                return False, (
                    f"{self.metric}: {candidate:.5f} exceeds the absolute ceiling "
                    f"{self.floor:.5f}"
                )

        if incumbent is None:
            return True, f"{self.metric}: {candidate:.5f} (no incumbent to compare)"

        delta = candidate - incumbent if self.higher_is_better else incumbent - candidate
        if delta >= self.min_delta:
            return True, (
                f"{self.metric}: {candidate:.5f} vs {incumbent:.5f} "
                f"(delta {delta:+.5f}, required {self.min_delta:+.5f})"
            )
        return False, (
            f"{self.metric}: {candidate:.5f} vs {incumbent:.5f} "
            f"(delta {delta:+.5f}, required {self.min_delta:+.5f})"
        )


#: Default gate.
#:
#: `ndcg@10` must genuinely improve - the 0.001 floor on the delta stops a
#: coin-flip difference from triggering a deployment, since promoting on noise
#: means the production model random-walks with every retrain.
#:
#: `coverage` may fall slightly but not collapse, and `hit_rate@10` must not
#: regress at all: it is the metric closest to what a user experiences.
DEFAULT_CRITERIA: tuple[GateCriterion, ...] = (
    GateCriterion("ndcg@10", min_delta=0.001, higher_is_better=True),
    GateCriterion("hit_rate@10", min_delta=0.0, higher_is_better=True),
    GateCriterion("coverage", min_delta=-0.05, higher_is_better=True, floor=0.05),
    GateCriterion(
        "personalisation", min_delta=-0.10, higher_is_better=True, required=False
    ),
)


@dataclass(slots=True)
class GateDecision:
    """The verdict, with every comparison that produced it."""

    outcome: GateOutcome
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    candidate_metrics: dict[str, float] = field(default_factory=dict)
    incumbent_metrics: dict[str, float] = field(default_factory=dict)
    decided_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.UTC))

    @property
    def should_promote(self) -> bool:
        return self.outcome in {GateOutcome.PROMOTE, GateOutcome.PROMOTE_FIRST}

    def summary(self) -> str:
        lines = [f"Promotion gate: {self.outcome.value.upper()}"]
        for line in self.passed:
            lines.append(f"  PASS  {line}")
        for line in self.failed:
            lines.append(f"  FAIL  {line}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "outcome": self.outcome.value,
            "passed": self.passed,
            "failed": self.failed,
            "candidate_metrics": self.candidate_metrics,
            "incumbent_metrics": self.incumbent_metrics,
            "decided_at": self.decided_at.isoformat(),
        }


def evaluate_gate(
    candidate_metrics: dict[str, float],
    incumbent_metrics: dict[str, float] | None,
    criteria: tuple[GateCriterion, ...] = DEFAULT_CRITERIA,
) -> GateDecision:
    """Compare a candidate against the incumbent and decide."""
    passed: list[str] = []
    failed: list[str] = []

    for criterion in criteria:
        ok, message = criterion.evaluate(
            candidate_metrics.get(criterion.metric),
            (incumbent_metrics or {}).get(criterion.metric),
        )
        if ok:
            passed.append(message)
        elif criterion.required:
            failed.append(message)
        else:
            # Advisory criteria are reported but do not block. Making every
            # metric blocking would mean no model is ever promoted.
            passed.append(f"(advisory) {message}")

    if not incumbent_metrics:
        # Nothing is serving yet, so the only question is whether the candidate
        # clears its absolute floors.
        outcome = GateOutcome.PROMOTE_FIRST if not failed else GateOutcome.REJECT
    else:
        outcome = GateOutcome.PROMOTE if not failed else GateOutcome.REJECT

    decision = GateDecision(
        outcome=outcome,
        passed=passed,
        failed=failed,
        candidate_metrics=dict(candidate_metrics),
        incumbent_metrics=dict(incumbent_metrics or {}),
    )
    logger.info("%s", decision.summary())
    return decision


__all__ = [
    "DEFAULT_CRITERIA",
    "GateCriterion",
    "GateDecision",
    "GateOutcome",
    "evaluate_gate",
]
