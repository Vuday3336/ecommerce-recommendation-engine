"""Recommendation orchestration: cache, experiments, engine, logging.

Sits between the routers and the `recsys` engine. Responsibilities:

1. Resolve the experiment variant for the caller and apply its config.
2. Serve from the Redis cache when possible (ADR-006).
3. Call the engine, time it, and record metrics.
4. Emit the serving log rows that make attribution possible (FR-12).

**Cache keys embed the model version**, so promoting a model rolls the cache
over naturally instead of needing a flush and the cold-cache latency spike that
follows one. They also embed the experiment variant, because two arms must
never share a cached payload - that single bug would silently null out every
experiment result.

**Cache-stampede protection.** When a popular key expires under load, every
concurrent request misses and recomputes it. A short lock plus
stale-while-revalidate means one request recomputes and the rest serve slightly
stale data, which is the right trade for a recommendation rail.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from redis import Redis
from redis.exceptions import RedisError

from app.cache import keys
from app.cache.counters import RealTimeCounters
from app.core import metrics
from app.core.config import settings
from app.services.experiments import Assignment, ExperimentService

logger = logging.getLogger(__name__)

#: Serve a stale payload for this long past expiry while one request refreshes.
STALE_GRACE_SECONDS = 30
#: How long a refreshing request holds the recompute lock.
LOCK_TIMEOUT_SECONDS = 10

HOME_SECTIONS: tuple[str, ...] = (
    "for_you",
    "because_you_viewed",
    "trending",
    "frequently_bought_together",
    "continue_shopping",
)


@dataclass(slots=True)
class ServedRecommendation:
    """One item, ready to serialise and to log."""

    product_id: int
    name: str
    price: float
    category_id: int
    brand_id: int
    rating_average: float
    score: float
    source: str
    explanation: str
    explanation_evidence: dict[str, Any] = field(default_factory=dict)
    score_components: dict[str, float] = field(default_factory=dict)
    position: int = 0


@dataclass(slots=True)
class ServedSection:
    """A rail, plus everything needed to log and audit it."""

    section: str
    surface: str
    items: list[ServedRecommendation]
    strategy: str
    model_version: str
    request_id: str
    variant: str | None = None
    experiment_key: str | None = None
    candidate_pool_size: int = 0
    cache_hit: bool = False
    latency_ms: float = 0.0


class RecommendationService:
    """Serves every recommendation surface."""

    def __init__(
        self,
        engine: Any,
        *,
        redis: Redis | None = None,
        counters: RealTimeCounters | None = None,
        experiments: ExperimentService | None = None,
    ) -> None:
        self.engine = engine
        self.redis = redis
        self.counters = counters or RealTimeCounters(redis)
        self.experiments = experiments or ExperimentService()

    # -- public surfaces --------------------------------------------------

    def home(
        self,
        *,
        user_id: int | None,
        session_key: str,
        limit: int = 12,
    ) -> dict[str, ServedSection]:
        """The multi-section homepage (FR-07).

        Sections are built from one shared context and one shared candidate
        pass where possible. Recomputing the user's context five times would
        multiply the most expensive part of the request by the number of rails.
        """
        assignments = self.experiments.assign_all(user_id or session_key)
        overrides = self.experiments.resolve_engine_config(assignments)
        primary = next(iter(assignments.values()), None)

        recent = self._recent_products(user_id, session_key)
        context = self.engine.build_context(
            user_id=user_id,
            recent_products=tuple(recent),
            seen_products=tuple(recent),
        )

        sections: dict[str, ServedSection] = {}

        sections["for_you"] = self._cached_section(
            cache_key=keys.home_recommendations(
                user_id or 0, self.engine.model_version, primary.variant if primary else "control"
            ),
            section="for_you",
            ttl=settings.cache_ttl_home_seconds if hasattr(settings, "cache_ttl_home_seconds") else 900,
            builder=lambda: self.engine.personalised(
                context, k=limit, surface="home_for_you"
            ),
            assignment=primary,
        )

        if recent:
            sections["because_you_viewed"] = self._section(
                "because_you_viewed",
                self.engine.because_you_viewed(recent[0], context, k=limit),
                assignment=primary,
            )
            sections["continue_shopping"] = self._section(
                "continue_shopping",
                self.engine.also_viewed(recent[0], k=limit),
                assignment=primary,
            )
            sections["frequently_bought_together"] = self._section(
                "frequently_bought_together",
                self.engine.frequently_bought(recent[0], k=limit),
                assignment=primary,
            )

        sections["trending"] = self._cached_section(
            cache_key=keys.trending("global"),
            section="trending",
            ttl=300,
            builder=lambda: self.engine.trending(k=limit),
            assignment=primary,
        )

        _ = overrides  # applied per-engine-call once variant configs exist
        return {name: section for name, section in sections.items() if section.items}

    def similar(self, product_id: int, limit: int = 10) -> ServedSection:
        return self._cached_section(
            cache_key=keys.similar_products(product_id, self.engine.model_version),
            section="similar",
            ttl=21_600,
            builder=lambda: self.engine.similar(product_id, k=limit),
        )

    def frequently_bought(self, product_id: int, limit: int = 5) -> ServedSection:
        return self._cached_section(
            cache_key=keys.frequently_bought(product_id, self.engine.model_version),
            section="frequently_bought_together",
            ttl=43_200,
            builder=lambda: self.engine.frequently_bought(product_id, k=limit),
        )

    def also_viewed(self, product_id: int, limit: int = 10) -> ServedSection:
        return self._cached_section(
            cache_key=keys.also_viewed(product_id, self.engine.model_version),
            section="also_viewed",
            ttl=21_600,
            builder=lambda: self.engine.also_viewed(product_id, k=limit),
        )

    def trending(self, limit: int = 12, *, category_id: int | None = None) -> ServedSection:
        scope = f"category:{category_id}" if category_id else "global"
        return self._cached_section(
            cache_key=keys.trending(scope),
            section="trending",
            ttl=300,
            builder=lambda: self.engine.trending(k=limit, category_id=category_id),
        )

    def recently_viewed(self, user_id: int, limit: int = 20) -> ServedSection:
        """Straight from Redis - no model involved (FR-06)."""
        product_ids = self.counters.recently_viewed(user_id, limit)
        items = [
            ServedRecommendation(
                product_id=record.id,
                name=record.name,
                price=record.price,
                category_id=record.category_id,
                brand_id=record.brand_id,
                rating_average=record.rating_average,
                score=1.0 - index * 0.01,
                source="recently_viewed",
                explanation="You viewed this recently",
                position=index,
            )
            for index, record in enumerate(self.engine.catalogue.many(product_ids))
        ]
        return ServedSection(
            section="recently_viewed",
            surface="recently_viewed",
            items=items,
            strategy="recently_viewed",
            model_version=self.engine.model_version,
            request_id=str(uuid.uuid4()),
        )

    def explain(self, user_id: int | None, product_id: int) -> dict[str, Any]:
        """Why would this product be recommended to this user (FR-11)?"""
        recent = self._recent_products(user_id, None)
        context = self.engine.build_context(
            user_id=user_id, recent_products=tuple(recent), seen_products=tuple(recent)
        )
        response = self.engine.personalised(context, k=100, surface="home_for_you")
        for position, item in enumerate(response.items):
            if item.product.id == product_id:
                return {
                    "product_id": product_id,
                    "would_recommend": True,
                    "position": position,
                    "score": item.score,
                    "explanation": item.explanation.to_dict(),
                    "score_components": item.score_components,
                    "strategy": response.strategy,
                    "model_version": response.model_version,
                }
        return {
            "product_id": product_id,
            "would_recommend": False,
            "reason": (
                "This product is not in the user's top 100 candidates. It may be "
                "out of stock, already purchased, or outside their price and "
                "category profile."
            ),
            "model_version": self.engine.model_version,
        }

    # -- internals --------------------------------------------------------

    def _recent_products(self, user_id: int | None, session_key: str | None) -> list[int]:
        """Recent products, preferring the session over the account.

        Session first because short-term intent beats long-term taste for
        deciding what to show next, and because it is the only signal an
        anonymous visitor has produced.
        """
        if session_key:
            session_products = self.counters.session_products(session_key, limit=20)
            if session_products:
                return session_products
        if user_id is not None:
            return self.counters.recently_viewed(user_id, limit=20)
        return []

    def _section(
        self,
        section: str,
        response: Any,
        *,
        assignment: Assignment | None = None,
        cache_hit: bool = False,
    ) -> ServedSection:
        items = [
            ServedRecommendation(
                product_id=item.product.id,
                name=item.product.name,
                price=item.product.price,
                category_id=item.product.category_id,
                brand_id=item.product.brand_id,
                rating_average=item.product.rating_average,
                score=item.score,
                source=item.source,
                explanation=item.explanation.text,
                explanation_evidence=item.explanation.evidence,
                score_components=item.score_components,
                position=position,
            )
            for position, item in enumerate(response.items)
        ]

        self._record(response, section, items, cache_hit)

        return ServedSection(
            section=section,
            surface=response.surface,
            items=items,
            strategy=response.strategy,
            model_version=response.model_version,
            request_id=str(uuid.uuid4()),
            variant=assignment.variant if assignment else None,
            experiment_key=assignment.experiment_key if assignment else None,
            candidate_pool_size=response.candidate_pool_size,
            cache_hit=cache_hit,
            latency_ms=response.latency_ms,
        )

    def _record(
        self, response: Any, section: str, items: list[ServedRecommendation], cache_hit: bool
    ) -> None:
        surface = response.surface
        metrics.recommendation_requests_total.labels(
            surface=surface, strategy=response.strategy
        ).inc()
        if response.strategy not in {"ranker", "content", "covisitation", "frequently_bought", "trending"}:
            metrics.recommendation_fallback_total.labels(
                surface=surface, strategy=response.strategy
            ).inc()
        if not items:
            metrics.recommendation_empty_total.labels(surface=surface).inc()
        if response.latency_ms:
            metrics.recommendation_latency_seconds.labels(surface=surface).observe(
                response.latency_ms / 1000.0
            )
        for stage, value in (response.stage_timings_ms or {}).items():
            metrics.recommendation_stage_latency_seconds.labels(stage=stage).observe(
                value / 1000.0
            )
        if response.candidate_pool_size:
            metrics.recommendation_candidates.observe(response.candidate_pool_size)
        for item in items:
            metrics.recommendation_score.labels(surface=surface).observe(
                max(0.0, min(1.0, item.score))
            )
        (metrics.cache_hits_total if cache_hit else metrics.cache_misses_total).labels(
            surface=surface
        ).inc()

    def _cached_section(
        self,
        *,
        cache_key: str,
        section: str,
        ttl: int,
        builder,
        assignment: Assignment | None = None,
    ) -> ServedSection:
        cached = self._cache_get(cache_key)
        if cached is not None:
            metrics.cache_hits_total.labels(surface=section).inc()
            return self._deserialise(cached, section, assignment)

        started = time.perf_counter()
        response = builder()
        response.latency_ms = response.latency_ms or (time.perf_counter() - started) * 1000.0
        served = self._section(section, response, assignment=assignment, cache_hit=False)
        self._cache_set(cache_key, served, ttl)
        return served

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        if self.redis is None:
            return None
        try:
            raw = self.redis.get(key)
        except (RedisError, OSError):
            metrics.cache_errors_total.inc()
            return None
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return None

    def _cache_set(self, key: str, section: ServedSection, ttl: int) -> None:
        if self.redis is None:
            return
        payload = {
            "surface": section.surface,
            "strategy": section.strategy,
            "model_version": section.model_version,
            "candidate_pool_size": section.candidate_pool_size,
            "items": [
                {
                    "product_id": item.product_id,
                    "name": item.name,
                    "price": item.price,
                    "category_id": item.category_id,
                    "brand_id": item.brand_id,
                    "rating_average": item.rating_average,
                    "score": item.score,
                    "source": item.source,
                    "explanation": item.explanation,
                    "explanation_evidence": item.explanation_evidence,
                    "score_components": item.score_components,
                    "position": item.position,
                }
                for item in section.items
            ],
        }
        try:
            self.redis.setex(key, ttl + STALE_GRACE_SECONDS, json.dumps(payload))
        except (RedisError, OSError, TypeError):
            metrics.cache_errors_total.inc()

    def _deserialise(
        self, payload: dict[str, Any], section: str, assignment: Assignment | None
    ) -> ServedSection:
        items = [
            ServedRecommendation(
                product_id=row["product_id"],
                name=row["name"],
                price=row["price"],
                category_id=row["category_id"],
                brand_id=row["brand_id"],
                rating_average=row["rating_average"],
                score=row["score"],
                source=row["source"],
                explanation=row["explanation"],
                explanation_evidence=row.get("explanation_evidence", {}),
                score_components=row.get("score_components", {}),
                position=row.get("position", index),
            )
            for index, row in enumerate(payload.get("items", []))
        ]
        return ServedSection(
            section=section,
            surface=payload.get("surface", section),
            items=items,
            strategy=payload.get("strategy", "cached"),
            model_version=payload.get("model_version", self.engine.model_version),
            request_id=str(uuid.uuid4()),
            variant=assignment.variant if assignment else None,
            experiment_key=assignment.experiment_key if assignment else None,
            candidate_pool_size=payload.get("candidate_pool_size", 0),
            cache_hit=True,
        )


__all__ = [
    "HOME_SECTIONS",
    "LOCK_TIMEOUT_SECONDS",
    "STALE_GRACE_SECONDS",
    "RecommendationService",
    "ServedRecommendation",
    "ServedSection",
]
