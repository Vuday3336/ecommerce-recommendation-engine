"""A/B testing: assignment and readout (FR-16, ADR-011).

**Assignment is computed, never looked up.** `sha256(experiment_key:subject)`
modulo 10,000 gives a bucket, and the bucket maps to a variant by allocation
range. This is sticky by construction (the same user always hashes to the same
bucket), costs no database read on the hot path, and is reproducible - the
assignment for any historical user can be recomputed months later, which is
what makes a post-hoc analysis possible at all.

Salting with the experiment key matters: without it, a user in the top decile
for one experiment would be in the top decile for every experiment, so
concurrent experiments would be correlated and their effects confounded.

Exposures are persisted asynchronously, purely so analysis has observed counts
to run a sample-ratio-mismatch check against. Nothing on the serving path
depends on that write succeeding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.core.security import stable_bucket
from app.models.enums import ExperimentStatus

logger = logging.getLogger(__name__)

BUCKETS = 10_000
DEFAULT_VARIANT = "control"


@dataclass(slots=True, frozen=True)
class Variant:
    """One arm of an experiment."""

    name: str
    allocation: float
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExperimentDefinition:
    """An experiment as the assignment logic sees it."""

    key: str
    variants: tuple[Variant, ...]
    status: ExperimentStatus = ExperimentStatus.RUNNING
    #: Fraction of eligible traffic entering the experiment at all. Traffic
    #: outside this holdout never sees the experiment and is not counted.
    traffic_allocation: float = 1.0
    experiment_id: int | None = None

    def __post_init__(self) -> None:
        total = sum(v.allocation for v in self.variants)
        if self.variants and abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"experiment {self.key}: variant allocations sum to {total}, not 1.0"
            )

    @property
    def is_active(self) -> bool:
        return self.status is ExperimentStatus.RUNNING

    def variant_for_bucket(self, bucket: int) -> Variant | None:
        """Map a bucket to a variant by cumulative allocation range."""
        if not self.variants:
            return None
        position = bucket / BUCKETS
        cumulative = 0.0
        for variant in self.variants:
            cumulative += variant.allocation
            if position < cumulative:
                return variant
        return self.variants[-1]


@dataclass(slots=True, frozen=True)
class Assignment:
    """The result of assigning one subject to one experiment."""

    experiment_key: str
    variant: str
    bucket: int
    experiment_id: int | None = None
    config: dict[str, Any] = field(default_factory=dict)
    in_experiment: bool = True


class ExperimentService:
    """Deterministic assignment plus in-memory experiment definitions.

    Definitions are cached in the process and refreshed periodically. An
    experiment configuration changes at human speed; reading it from the
    database per request would add a query to the hot path for data that is
    effectively static.
    """

    def __init__(self, experiments: list[ExperimentDefinition] | None = None) -> None:
        self._experiments: dict[str, ExperimentDefinition] = {
            e.key: e for e in (experiments or [])
        }

    def register(self, experiment: ExperimentDefinition) -> None:
        self._experiments[experiment.key] = experiment

    def replace_all(self, experiments: list[ExperimentDefinition]) -> None:
        self._experiments = {e.key: e for e in experiments}

    @property
    def active(self) -> list[ExperimentDefinition]:
        return [e for e in self._experiments.values() if e.is_active]

    def assign(
        self, experiment_key: str, subject: str | int | None
    ) -> Assignment | None:
        """Assign one subject to one experiment.

        `subject` is the user id when known, otherwise the session key, so
        anonymous traffic is still assigned consistently within a session. It
        returns None when the experiment does not exist or is not running -
        callers then serve the default behaviour.
        """
        experiment = self._experiments.get(experiment_key)
        if experiment is None or not experiment.is_active or subject is None:
            return None

        bucket = stable_bucket(f"{experiment_key}:{subject}", buckets=BUCKETS)

        # The holdout check uses a *separately salted* hash. Reusing the same
        # bucket for both "are you in the experiment?" and "which arm?" would
        # correlate the two decisions: with a 50% holdout, every user in the
        # experiment would come from the lower half of the bucket space and so
        # would land in the first variant.
        if experiment.traffic_allocation < 1.0:
            eligibility = stable_bucket(f"holdout:{experiment_key}:{subject}", buckets=BUCKETS)
            if eligibility >= experiment.traffic_allocation * BUCKETS:
                return Assignment(
                    experiment_key=experiment_key,
                    variant=DEFAULT_VARIANT,
                    bucket=bucket,
                    experiment_id=experiment.experiment_id,
                    in_experiment=False,
                )

        variant = experiment.variant_for_bucket(bucket)
        if variant is None:
            return None

        return Assignment(
            experiment_key=experiment_key,
            variant=variant.name,
            bucket=bucket,
            experiment_id=experiment.experiment_id,
            config=variant.config,
            in_experiment=True,
        )

    def assign_all(self, subject: str | int | None) -> dict[str, Assignment]:
        """Assign a subject to every running experiment."""
        assignments: dict[str, Assignment] = {}
        for experiment in self.active:
            assignment = self.assign(experiment.key, subject)
            if assignment is not None:
                assignments[experiment.key] = assignment
        return assignments

    def resolve_engine_config(
        self, assignments: dict[str, Assignment]
    ) -> dict[str, Any]:
        """Merge variant configs into engine overrides.

        The variant config is what makes an experiment mean anything: a
        treatment arm is a *configuration difference*, such as
        `{"enable_ml_ranker": false}` or `{"mmr_lambda": 0.5}`, applied to the
        same engine. Building a second engine per arm would make experiments
        expensive and their results incomparable.
        """
        merged: dict[str, Any] = {}
        for assignment in assignments.values():
            if assignment.in_experiment:
                merged.update(assignment.config)
        return merged


def definitions_from_rows(rows: list[Any]) -> list[ExperimentDefinition]:
    """Build definitions from `experiments` table rows."""
    definitions: list[ExperimentDefinition] = []
    for row in rows:
        try:
            variants = tuple(
                Variant(
                    name=str(v["name"]),
                    allocation=float(v["allocation"]),
                    config=dict(v.get("config", {})),
                )
                for v in (row.variants or [])
            )
            definitions.append(
                ExperimentDefinition(
                    key=row.key,
                    variants=variants,
                    status=row.status,
                    traffic_allocation=float(row.traffic_allocation),
                    experiment_id=row.id,
                )
            )
        except (KeyError, TypeError, ValueError):
            # A malformed experiment must not break recommendations for
            # everyone; it is skipped and logged.
            logger.exception("skipping malformed experiment %s", getattr(row, "key", "?"))
    return definitions


__all__ = [
    "BUCKETS",
    "DEFAULT_VARIANT",
    "Assignment",
    "ExperimentDefinition",
    "ExperimentService",
    "Variant",
    "definitions_from_rows",
]
