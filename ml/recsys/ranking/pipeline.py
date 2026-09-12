"""The two-stage recommender: retrieve, rank, re-rank, explain.

This is the object the API serves from. It implements `Recommender`, so the
evaluation harness scores it exactly like every simpler model - which is what
makes the comparison in the evaluation report apples-to-apples.

The degradation ladder lives here (architecture.md section 4.2). If the ranker
artefact is missing or raises, the pipeline falls back to the hybrid blend; if
retrieval returns nothing, it falls back to the static popularity list. A
recommendation surface never fails, and the rung that served the request is
reported on the response so the fallback rate is measurable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from recsys.candidates.generator import CandidateGenerator, CandidateSet
from recsys.config.settings import DiversityConfig
from recsys.models.base import (
    RecommendationContext,
    Recommender,
    Scored,
    normalise_scores,
    top_k,
)
from recsys.models.hybrid import HybridRecommender
from recsys.ranking.dataset import RankingDatasetBuilder

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RankedResult:
    """A served recommendation list plus everything needed to log and explain it."""

    items: list[Scored]
    strategy: str
    candidate_pool_size: int
    score_components: dict[int, dict[str, float]] = field(default_factory=dict)
    sources: dict[int, list[str]] = field(default_factory=dict)
    stage_timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def product_ids(self) -> list[int]:
        return [item.product_id for item in self.items]


class TwoStageRecommender(Recommender):
    """Candidate generation followed by learned ranking."""

    name = "two_stage_ranker"
    supports_cold_users = True

    def __init__(
        self,
        *,
        generator: CandidateGenerator,
        dataset_builder: RankingDatasetBuilder,
        ranker: Any | None,
        hybrid: HybridRecommender | None = None,
        diversity: DiversityConfig | None = None,
        content_model: Recommender | None = None,
        collaborative_model: Recommender | None = None,
        fallback: list[int] | None = None,
    ) -> None:
        self.generator = generator
        self.dataset_builder = dataset_builder
        self.ranker = ranker
        self.hybrid = hybrid
        self.diversity = diversity or DiversityConfig()
        self.content_model = content_model
        self.collaborative_model = collaborative_model
        self.fallback = fallback or []

    def fit(self, *_: object, **__: object) -> TwoStageRecommender:
        """No-op: stages are trained separately and injected."""
        return self

    # -- serving ----------------------------------------------------------

    def rank(self, context: RecommendationContext, k: int = 10) -> RankedResult:
        import time

        timings: dict[str, float] = {}

        started = time.perf_counter()
        candidates = self.generator.generate(context)
        timings["candidates"] = (time.perf_counter() - started) * 1000.0

        if not candidates.product_ids:
            return self._fallback_result(context, k, timings, candidates)

        if self.ranker is not None:
            try:
                started = time.perf_counter()
                result = self._rank_with_model(context, candidates, k)
                timings["ranking"] = (time.perf_counter() - started) * 1000.0
                result.stage_timings_ms = timings
                return result
            except Exception:
                logger.exception("ranker failed; falling back to the hybrid blend")

        return self._hybrid_result(context, candidates, k, timings)

    def _rank_with_model(
        self, context: RecommendationContext, candidates: CandidateSet, k: int
    ) -> RankedResult:
        content_scores = (
            self.content_model.score(context, candidates.product_ids)
            if self.content_model is not None
            else None
        )
        collaborative_scores = (
            self.collaborative_model.score(context, candidates.product_ids)
            if self.collaborative_model is not None
            else None
        )

        features = self.dataset_builder.build_for_user(
            context,
            candidates,
            content_similarity=content_scores,
            collaborative_score=collaborative_scores,
        )
        scores = self.ranker.predict(features)
        scored = dict(zip(candidates.product_ids, (float(s) for s in scores), strict=True))
        ranked = top_k(normalise_scores(scored), max(k * 4, 40), source="ranker")

        items = (
            self.hybrid.diversify(ranked, k)
            if self.diversity.enabled and self.hybrid is not None
            else ranked[:k]
        )
        return RankedResult(
            items=items,
            strategy="ranker",
            candidate_pool_size=len(candidates),
            score_components={
                item.product_id: {
                    "ranker": round(item.score, 5),
                    **{
                        source: round(candidates.score_from(source, item.product_id), 5)
                        for source in candidates.sources_for(item.product_id)
                    },
                }
                for item in items
            },
            sources={item.product_id: candidates.sources_for(item.product_id) for item in items},
        )

    def _hybrid_result(
        self,
        context: RecommendationContext,
        candidates: CandidateSet,
        k: int,
        timings: dict[str, float],
    ) -> RankedResult:
        if self.hybrid is None:
            return self._fallback_result(context, k, timings, candidates)
        combined, breakdown = self.hybrid.blend(context, pool=400)
        restricted = {
            pid: combined.get(pid, 0.0) for pid in candidates.product_ids
        }
        ranked = top_k(restricted, max(k * 4, 40), source="hybrid")
        items = self.hybrid.diversify(ranked, k) if self.diversity.enabled else ranked[:k]
        return RankedResult(
            items=items,
            strategy="hybrid",
            candidate_pool_size=len(candidates),
            score_components={item.product_id: breakdown.get(item.product_id, {}) for item in items},
            sources={item.product_id: candidates.sources_for(item.product_id) for item in items},
            stage_timings_ms=timings,
        )

    def _fallback_result(
        self,
        context: RecommendationContext,
        k: int,
        timings: dict[str, float],
        candidates: CandidateSet | None = None,
    ) -> RankedResult:
        items = [
            Scored(pid, 1.0 - i * 0.001, "static_fallback")
            for i, pid in enumerate(p for p in self.fallback if p not in context.exclude)
        ][:k]
        return RankedResult(
            items=items,
            strategy="static_fallback",
            candidate_pool_size=len(candidates) if candidates else 0,
            stage_timings_ms=timings,
        )

    # -- Recommender interface --------------------------------------------

    def recommend(self, context: RecommendationContext, k: int = 10) -> list[Scored]:
        return self.rank(context, k).items


__all__ = ["RankedResult", "TwoStageRecommender"]
