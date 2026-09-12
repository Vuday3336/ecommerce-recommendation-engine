"""Recommendation analytics and experiment readout (FR-15, FR-16).

Turns the serving log into the numbers a business actually asks for: did people
click, did they buy, and did the treatment arm do better than control.

**Everything here reads from `recommendations` joined to its outcome tables.**
That is the whole reason the serving log records a `recommendation_id` per item
and the frontend echoes it back on click - without that join key, "did this
recommendation work?" is unanswerable and CTR becomes a guess.

**Impressions, not responses, are the CTR denominator.** A rail below the fold
was served but never seen. Counting it would depress CTR for exactly the
surfaces that are working, so the denominator is the impression table, which
the frontend writes from an IntersectionObserver.

When the database is unavailable every method returns an empty result with an
explicit `available: false`, rather than raising or inventing numbers. A
dashboard showing "no data yet" is honest; one showing zeros is a lie.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.models.recommendations import (
    Recommendation,
    RecommendationClick,
    RecommendationConversion,
    RecommendationImpression,
)

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_DAYS = 30


@dataclass(slots=True)
class FunnelMetrics:
    """Impression -> click -> conversion for one slice of traffic."""

    label: str
    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    revenue: float = 0.0
    users: int = 0

    @property
    def ctr(self) -> float:
        return self.clicks / self.impressions if self.impressions else 0.0

    @property
    def conversion_rate(self) -> float:
        """Conversions per *click*, not per impression.

        Click-to-conversion isolates the quality of the recommendation from the
        quality of the rail placement; impression-to-conversion mixes the two
        and moves whenever the page layout changes.
        """
        return self.conversions / self.clicks if self.clicks else 0.0

    @property
    def revenue_per_user(self) -> float:
        return self.revenue / self.users if self.users else 0.0

    @property
    def average_order_value(self) -> float:
        return self.revenue / self.conversions if self.conversions else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "impressions": self.impressions,
            "clicks": self.clicks,
            "conversions": self.conversions,
            "revenue": round(self.revenue, 2),
            "users": self.users,
            "ctr": round(self.ctr, 5),
            "conversion_rate": round(self.conversion_rate, 5),
            "revenue_per_user": round(self.revenue_per_user, 4),
            "average_order_value": round(self.average_order_value, 2),
        }


@dataclass(slots=True)
class ExperimentReadout:
    """Control versus treatment, with the guardrails that make it trustworthy."""

    experiment_key: str
    arms: list[FunnelMetrics] = field(default_factory=list)
    tests: dict[str, Any] = field(default_factory=dict)
    sample_ratio_p_value: float | None = None
    sample_ratio_mismatch: bool = False
    minimum_sample_size: int = 1000
    ready: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_key": self.experiment_key,
            "arms": [arm.to_dict() for arm in self.arms],
            "tests": self.tests,
            "sample_ratio_p_value": self.sample_ratio_p_value,
            "sample_ratio_mismatch": self.sample_ratio_mismatch,
            "minimum_sample_size": self.minimum_sample_size,
            "ready": self.ready,
            "notes": self.notes,
        }


class AnalyticsService:
    """Reads the serving log and reports on it."""

    def __init__(self, session_factory: Any | None = None) -> None:
        self._session_factory = session_factory

    @property
    def available(self) -> bool:
        return self._session_factory is not None

    # -- helpers ----------------------------------------------------------

    def _since(self, days: int) -> dt.datetime:
        return dt.datetime.now(dt.UTC) - dt.timedelta(days=days)

    def _base_query(self, since: dt.datetime) -> Select:
        """Recommendations joined to their outcomes.

        Outer joins on every outcome table: an item that was served but never
        seen, or seen but never clicked, must still appear in the denominator.
        Inner joins here would silently report CTR over only the rows that
        already converted, which is always close to 100%.
        """
        return (
            select(Recommendation)
            .outerjoin(
                RecommendationImpression,
                RecommendationImpression.recommendation_id == Recommendation.id,
            )
            .outerjoin(
                RecommendationClick,
                RecommendationClick.recommendation_id == Recommendation.id,
            )
            .outerjoin(
                RecommendationConversion,
                RecommendationConversion.recommendation_id == Recommendation.id,
            )
            .where(Recommendation.served_at >= since)
        )

    def _aggregate(
        self, session: Session, group_column: Any, since: dt.datetime, extra_where: Any = None
    ) -> list[FunnelMetrics]:
        query = (
            select(
                group_column.label("bucket"),
                func.count(func.distinct(RecommendationImpression.id)).label("impressions"),
                func.count(func.distinct(RecommendationClick.id)).label("clicks"),
                func.count(func.distinct(RecommendationConversion.id)).label("conversions"),
                func.coalesce(func.sum(RecommendationConversion.revenue), 0).label("revenue"),
                func.count(func.distinct(Recommendation.user_id)).label("users"),
            )
            .select_from(Recommendation)
            .outerjoin(
                RecommendationImpression,
                RecommendationImpression.recommendation_id == Recommendation.id,
            )
            .outerjoin(
                RecommendationClick,
                RecommendationClick.recommendation_id == Recommendation.id,
            )
            .outerjoin(
                RecommendationConversion,
                RecommendationConversion.recommendation_id == Recommendation.id,
            )
            .where(Recommendation.served_at >= since)
            .group_by(group_column)
        )
        if extra_where is not None:
            query = query.where(extra_where)

        return [
            FunnelMetrics(
                label=str(row.bucket),
                impressions=int(row.impressions or 0),
                clicks=int(row.clicks or 0),
                conversions=int(row.conversions or 0),
                revenue=float(row.revenue or 0.0),
                users=int(row.users or 0),
            )
            for row in session.execute(query).all()
        ]

    # -- public -----------------------------------------------------------

    def overview(self, *, days: int = DEFAULT_WINDOW_DAYS) -> dict[str, Any]:
        """Headline business metrics for the dashboard."""
        if not self.available:
            return self._unavailable()

        since = self._since(days)
        try:
            with self._session_factory() as session:
                by_surface = self._aggregate(session, Recommendation.surface, since)
                by_source = self._aggregate(session, Recommendation.source, since)
                by_model = self._aggregate(session, Recommendation.model_version_id, since)

                total = FunnelMetrics("all")
                for row in by_surface:
                    total.impressions += row.impressions
                    total.clicks += row.clicks
                    total.conversions += row.conversions
                    total.revenue += row.revenue
                total.users = session.scalar(
                    select(func.count(func.distinct(Recommendation.user_id))).where(
                        Recommendation.served_at >= since
                    )
                ) or 0

                fallback_rate = self._fallback_rate(session, since)
        except SQLAlchemyError:
            logger.exception("analytics query failed")
            return self._unavailable("the analytics query failed")

        return {
            "available": True,
            "window_days": days,
            "total": total.to_dict(),
            "by_surface": [row.to_dict() for row in by_surface],
            "by_source": [row.to_dict() for row in by_source],
            "by_model_version": [row.to_dict() for row in by_model],
            "fallback_rate": fallback_rate,
        }

    def _fallback_rate(self, session: Session, since: dt.datetime) -> float:
        """Share of items served by a degraded strategy.

        The health metric that no other signal exposes: when the ranker fails,
        latency improves and errors stay at zero.
        """
        rows = session.execute(
            select(Recommendation.strategy, func.count())
            .where(Recommendation.served_at >= since)
            .group_by(Recommendation.strategy)
        ).all()
        total = sum(int(count) for _, count in rows)
        if total == 0:
            return 0.0
        degraded = sum(
            int(count)
            for strategy, count in rows
            if str(getattr(strategy, "value", strategy))
            not in {"ranker", "hybrid"}
        )
        return round(degraded / total, 5)

    def top_products(self, *, days: int = DEFAULT_WINDOW_DAYS, limit: int = 25) -> dict[str, Any]:
        """Most recommended, most clicked and most converted products."""
        if not self.available:
            return self._unavailable()

        since = self._since(days)
        try:
            with self._session_factory() as session:
                rows = session.execute(
                    select(
                        Recommendation.product_id,
                        func.count(Recommendation.id).label("served"),
                        func.count(func.distinct(RecommendationClick.id)).label("clicks"),
                        func.count(func.distinct(RecommendationConversion.id)).label(
                            "conversions"
                        ),
                        func.coalesce(func.sum(RecommendationConversion.revenue), 0).label(
                            "revenue"
                        ),
                    )
                    .select_from(Recommendation)
                    .outerjoin(
                        RecommendationClick,
                        RecommendationClick.recommendation_id == Recommendation.id,
                    )
                    .outerjoin(
                        RecommendationConversion,
                        RecommendationConversion.recommendation_id == Recommendation.id,
                    )
                    .where(Recommendation.served_at >= since)
                    .group_by(Recommendation.product_id)
                    .order_by(func.count(Recommendation.id).desc())
                    .limit(limit)
                ).all()
        except SQLAlchemyError:
            logger.exception("top-products query failed")
            return self._unavailable("the analytics query failed")

        return {
            "available": True,
            "window_days": days,
            "products": [
                {
                    "product_id": int(row.product_id),
                    "served": int(row.served),
                    "clicks": int(row.clicks or 0),
                    "conversions": int(row.conversions or 0),
                    "revenue": round(float(row.revenue or 0.0), 2),
                    "ctr": round((row.clicks or 0) / row.served, 5) if row.served else 0.0,
                }
                for row in rows
            ],
        }

    def experiment(
        self,
        experiment_key: str,
        *,
        days: int = DEFAULT_WINDOW_DAYS,
        minimum_sample_size: int = 1000,
        expected_allocation: dict[str, float] | None = None,
    ) -> ExperimentReadout:
        """Control versus treatment, with the guardrails checked first.

        Order matters. The sample-ratio check runs *before* the significance
        tests, and a mismatch marks the readout not-ready regardless of how
        good the lift looks. A skewed split invalidates every number, and a
        broken experiment usually produces a convincing-looking win.
        """
        readout = ExperimentReadout(
            experiment_key=experiment_key, minimum_sample_size=minimum_sample_size
        )
        if not self.available:
            readout.notes.append(
                "No database connection: experiment results need the serving log."
            )
            return readout

        since = self._since(days)
        try:
            with self._session_factory() as session:
                readout.arms = self._aggregate(
                    session,
                    Recommendation.variant,
                    since,
                    extra_where=Recommendation.variant.isnot(None),
                )
        except SQLAlchemyError:
            logger.exception("experiment query failed")
            readout.notes.append("The experiment query failed.")
            return readout

        if len(readout.arms) < 2:
            readout.notes.append(
                "Fewer than two arms have served traffic; nothing to compare yet."
            )
            return readout

        from recsys.evaluation.significance import (
            sample_ratio_mismatch,
            two_proportion_test,
        )

        observed = {arm.label: arm.users for arm in readout.arms}
        allocation = expected_allocation or {
            label: 1.0 / len(observed) for label in observed
        }
        readout.sample_ratio_p_value, readout.sample_ratio_mismatch = sample_ratio_mismatch(
            observed, allocation
        )
        if readout.sample_ratio_mismatch:
            readout.notes.append(
                "Sample-ratio mismatch: traffic did not split as intended, so these "
                "results are not trustworthy. Investigate assignment before reading "
                "any lift."
            )

        control = next(
            (arm for arm in readout.arms if arm.label == "control"), readout.arms[0]
        )
        treatments = [arm for arm in readout.arms if arm is not control]

        for treatment in treatments:
            tests: dict[str, Any] = {}
            if control.impressions and treatment.impressions:
                ctr_test = two_proportion_test(
                    control.clicks, control.impressions,
                    treatment.clicks, treatment.impressions,
                )
                tests["ctr"] = {
                    "control": round(ctr_test.control_rate, 5),
                    "treatment": round(ctr_test.treatment_rate, 5),
                    "relative_lift": round(ctr_test.relative_lift, 5),
                    "p_value": round(ctr_test.p_value, 5),
                    "significant": ctr_test.significant,
                    "summary": ctr_test.summary(),
                }
            if control.clicks and treatment.clicks:
                cvr_test = two_proportion_test(
                    control.conversions, control.clicks,
                    treatment.conversions, treatment.clicks,
                )
                tests["conversion_rate"] = {
                    "control": round(cvr_test.control_rate, 5),
                    "treatment": round(cvr_test.treatment_rate, 5),
                    "relative_lift": round(cvr_test.relative_lift, 5),
                    "p_value": round(cvr_test.p_value, 5),
                    "significant": cvr_test.significant,
                    "summary": cvr_test.summary(),
                }
            # Revenue per user is deliberately *not* significance-tested here.
            # It is continuous and heavily skewed - a handful of large orders
            # dominate - so a proportion test does not apply and a t-test would
            # be badly calibrated. It is reported as a magnitude only; a
            # bootstrap over per-user revenue is the correct test.
            tests["revenue_per_user"] = {
                "control": round(control.revenue_per_user, 4),
                "treatment": round(treatment.revenue_per_user, 4),
                "note": "reported without a significance test; see the docstring",
            }
            readout.tests[treatment.label] = tests

        smallest = min(arm.users for arm in readout.arms)
        readout.ready = (
            smallest >= minimum_sample_size and not readout.sample_ratio_mismatch
        )
        if smallest < minimum_sample_size:
            readout.notes.append(
                f"Smallest arm has {smallest} users, below the {minimum_sample_size} "
                "minimum. Results are not yet readable - peeking early and stopping "
                "on a favourable number is how false positives ship."
            )
        return readout

    @staticmethod
    def _unavailable(reason: str = "no database connection") -> dict[str, Any]:
        return {
            "available": False,
            "reason": reason,
            "total": FunnelMetrics("all").to_dict(),
            "by_surface": [],
            "by_source": [],
            "by_model_version": [],
            "fallback_rate": 0.0,
        }


__all__ = [
    "DEFAULT_WINDOW_DAYS",
    "AnalyticsService",
    "ExperimentReadout",
    "FunnelMetrics",
]
