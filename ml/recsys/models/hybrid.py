"""Hybrid recommender: adaptive weighted blend (ADR-014).

The brief suggests fixed weights:

    0.45 x collaborative + 0.25 x content + 0.15 x trending
    + 0.10 x affinity + 0.05 x popularity

Those are a reasonable *starting point* and are close to the `rich` profile
below. Shipping them as constants would be the mistake, for one concrete
reason: a user with two clicks would be scored 45% by a collaborative model
that has essentially no information about them, while the signals that do work
for that user - content similarity to what they just looked at, and what is
popular in their category - get 30% between them.

So the weights are a function of how much history the user has. Four regimes,
selected by interaction count, each summing to 1. As history grows the
collaborative term rises and the popularity term falls, which is exactly the
direction the evidence moves.

**Score normalisation is not optional.** An ALS dot product, a cosine
similarity and a log-popularity value have completely different scales. Summing
them with weights without normalising first makes the weights decorative - the
component with the largest raw magnitude wins regardless of what weight it was
given. Every component is min-max normalised over the candidate set first.
"""

from __future__ import annotations

import logging

import numpy as np

from recsys.config.settings import DiversityConfig, HybridConfig
from recsys.models.base import (
    RecommendationContext,
    Recommender,
    Scored,
    normalise_scores,
    top_k,
)

logger = logging.getLogger(__name__)

COMPONENTS: tuple[str, ...] = (
    "collaborative",
    "content",
    "trending",
    "affinity",
    "popularity",
)


class HybridRecommender(Recommender):
    """Blends component recommenders with history-adaptive weights."""

    name = "hybrid"
    supports_cold_users = True

    def __init__(
        self,
        *,
        collaborative: Recommender | None = None,
        content: Recommender | None = None,
        trending: Recommender | None = None,
        popularity: Recommender | None = None,
        covisitation: Recommender | None = None,
        config: HybridConfig | None = None,
        diversity: DiversityConfig | None = None,
        product_categories: dict[int, int] | None = None,
        product_brands: dict[int, int] | None = None,
        novelty: dict[int, float] | None = None,
    ) -> None:
        self.collaborative = collaborative
        self.content = content
        self.trending = trending
        self.popularity = popularity
        # Co-visitation feeds the `affinity` slot: it is the short-term,
        # session-scoped signal, which is what "what is this user into right
        # now" means in practice.
        self.covisitation = covisitation

        self.config = config or HybridConfig()
        self.diversity = diversity or DiversityConfig()
        self.product_categories = product_categories or {}
        self.product_brands = product_brands or {}
        self.novelty = novelty or {}

    def fit(self, *_: object, **__: object) -> HybridRecommender:
        """No-op: components are fitted individually and injected.

        Deliberate. The hybrid owns the blending policy, not the training of
        its parts, so each component can be retrained, versioned and evaluated
        on its own.
        """
        return self

    # -- scoring ----------------------------------------------------------

    def component_scores(
        self, context: RecommendationContext, pool: int
    ) -> dict[str, dict[int, float]]:
        """Per-component normalised scores over each component's own top-N."""
        scores: dict[str, dict[int, float]] = {}

        if self.collaborative is not None:
            scores["collaborative"] = normalise_scores(
                {s.product_id: s.score for s in self.collaborative.recommend(context, pool)}
            )
        if self.content is not None:
            scores["content"] = normalise_scores(
                {s.product_id: s.score for s in self.content.recommend(context, pool)}
            )
        if self.trending is not None:
            scores["trending"] = normalise_scores(
                {s.product_id: s.score for s in self.trending.recommend(context, pool)}
            )
        if self.covisitation is not None:
            scores["affinity"] = normalise_scores(
                {s.product_id: s.score for s in self.covisitation.recommend(context, pool)}
            )
        if self.popularity is not None:
            scores["popularity"] = normalise_scores(
                {s.product_id: s.score for s in self.popularity.recommend(context, pool)}
            )
        return scores

    def blend(
        self, context: RecommendationContext, pool: int = 400
    ) -> tuple[dict[int, float], dict[int, dict[str, float]]]:
        """Weighted blend plus the per-component breakdown.

        The breakdown is returned, not discarded: it becomes
        `recommendations.score_components` in the serving log, which is what
        makes a ranking decision reconstructible months later and what the
        rule-based explanation reads to say *why* an item was chosen (FR-11).
        """
        weights = dict(
            zip(COMPONENTS, self.config.weights_for(context.interaction_count), strict=True)
        )
        components = self.component_scores(context, pool)

        combined: dict[int, float] = {}
        breakdown: dict[int, dict[str, float]] = {}

        for component, values in components.items():
            weight = weights.get(component, 0.0)
            if weight <= 0:
                continue
            for product_id, score in values.items():
                contribution = weight * score
                combined[product_id] = combined.get(product_id, 0.0) + contribution
                breakdown.setdefault(product_id, {})[component] = round(score, 5)

        if self.diversity.popularity_discount > 0 and self.novelty:
            # Inverse-popularity discount. Every component of this system
            # amplifies popularity - popular items get more impressions, so
            # more clicks, so more training signal (R-06). This is the one
            # place that pushes back, and it is applied before truncation so
            # it can actually change what survives.
            for product_id in combined:
                novelty = self.novelty.get(product_id, 0.5)
                combined[product_id] *= 1.0 + self.diversity.popularity_discount * novelty

        return combined, breakdown

    def recommend(self, context: RecommendationContext, k: int = 10) -> list[Scored]:
        combined, _ = self.blend(context, pool=max(k * 8, 200))
        for product_id in list(combined):
            if product_id in context.exclude:
                del combined[product_id]
        ranked = top_k(combined, k * 4, source=self.name)

        if not self.diversity.enabled:
            return ranked[:k]
        return self.diversify(ranked, k)

    def score(
        self, context: RecommendationContext, product_ids: list[int]
    ) -> dict[int, float]:
        combined, _ = self.blend(context, pool=2000)
        return {pid: combined.get(pid, 0.0) for pid in product_ids}

    # -- re-ranking -------------------------------------------------------

    def diversify(self, ranked: list[Scored], k: int) -> list[Scored]:
        """Greedy MMR with hard category and brand caps.

        MMR alone lets a strong category dominate when its items are also the
        most relevant, so hard caps sit on top of it. The caps are what stop a
        homepage rail from being five running shoes, which is the failure mode
        users actually complain about.
        """
        selected: list[Scored] = []
        category_counts: dict[int, int] = {}
        brand_counts: dict[int, int] = {}
        lam = self.diversity.mmr_lambda

        remaining = list(ranked)
        while remaining and len(selected) < k:
            best_index = 0
            best_value = -np.inf
            for index, candidate in enumerate(remaining):
                category = self.product_categories.get(candidate.product_id)
                brand = self.product_brands.get(candidate.product_id)
                if (
                    category is not None
                    and category_counts.get(category, 0) >= self.diversity.max_per_category
                ):
                    continue
                if (
                    brand is not None
                    and brand_counts.get(brand, 0) >= self.diversity.max_per_brand
                ):
                    continue

                # Redundancy proxy: how much of the already-selected list shares
                # this candidate's category. Cheap, and it captures the failure
                # mode that matters without needing a pairwise similarity matrix
                # on the hot path.
                redundancy = (
                    category_counts.get(category, 0) / max(len(selected), 1)
                    if category is not None and selected
                    else 0.0
                )
                value = lam * candidate.score - (1.0 - lam) * redundancy
                if value > best_value:
                    best_value = value
                    best_index = index

            chosen = remaining.pop(best_index)
            category = self.product_categories.get(chosen.product_id)
            brand = self.product_brands.get(chosen.product_id)
            if (
                category is not None
                and category_counts.get(category, 0) >= self.diversity.max_per_category
            ):
                continue
            if brand is not None and brand_counts.get(brand, 0) >= self.diversity.max_per_brand:
                continue

            selected.append(chosen)
            if category is not None:
                category_counts[category] = category_counts.get(category, 0) + 1
            if brand is not None:
                brand_counts[brand] = brand_counts.get(brand, 0) + 1

        # If the caps starved the list, top it up in relevance order rather
        # than returning fewer items than asked for. A short rail looks broken.
        if len(selected) < k:
            chosen_ids = {item.product_id for item in selected}
            for candidate in ranked:
                if candidate.product_id not in chosen_ids:
                    selected.append(candidate)
                    if len(selected) >= k:
                        break
        return selected[:k]


__all__ = ["COMPONENTS", "HybridRecommender"]
