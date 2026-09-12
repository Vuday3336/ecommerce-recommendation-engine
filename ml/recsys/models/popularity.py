"""Popularity and trending baselines.

This is the number every other model has to beat. It matters more than its
simplicity suggests: popularity is a genuinely strong recommender, it is what a
new user sees, and a "personalised" model that fails to beat it is not
personalising - it is just adding latency and complexity.

Three variants, because they answer different questions:

* **Global popularity** - what most people buy. The floor.
* **Category popularity** - what most people buy *in a category the user cares
  about*. Already a weak form of personalisation, and a surprisingly hard
  baseline to beat.
* **Trending** - what is being bought *now*, with exponential time decay. Picks
  up seasonality and new releases without retraining.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from recsys.config.settings import PopularityConfig
from recsys.models.base import (
    RecommendationContext,
    Recommender,
    Scored,
    normalise_scores,
    top_k,
)

#: Event weights for popularity accumulation. Matching the trending weights in
#: `app/cache/counters.py` and `features/product.py` keeps one definition of
#: "how much does this event count" across batch and online paths.
POPULARITY_EVENT_WEIGHTS: dict[str, float] = {
    "PRODUCT_VIEW": 1.0,
    "PRODUCT_CLICK": 1.5,
    "WISHLIST": 3.0,
    "ADD_TO_CART": 5.0,
    "PURCHASE": 12.0,
}


class PopularityRecommender(Recommender):
    """Global and per-category popularity."""

    name = "popularity"
    supports_cold_users = True

    def __init__(self, config: PopularityConfig | None = None) -> None:
        self.config = config or PopularityConfig()
        self.global_scores: dict[int, float] = {}
        self.category_scores: dict[int, dict[int, float]] = {}
        self.product_category: dict[int, int] = {}
        self._global_ranked: list[Scored] = []

    def fit(
        self,
        interactions: pd.DataFrame,
        products: pd.DataFrame,
        *,
        as_of: dt.datetime | None = None,
    ) -> PopularityRecommender:
        history = interactions
        if as_of is not None:
            history = history[history["occurred_at"] < pd.Timestamp(as_of)]

        self.product_category = products.set_index("id")["category_id"].to_dict()

        if history.empty:
            return self

        weighted = history.assign(
            w=history["event_type"].map(POPULARITY_EVENT_WEIGHTS).fillna(0.0)
        )
        # Log-scale the accumulated weight. Raw counts are power-law
        # distributed, so a linear score is decided entirely by the head and
        # every product outside the top hundred scores indistinguishably.
        totals = weighted.groupby("product_id")["w"].sum()
        self.global_scores = normalise_scores(np.log1p(totals).to_dict())
        self._global_ranked = top_k(self.global_scores, 5000, source=self.name)

        by_category = weighted.assign(
            category_id=weighted["product_id"].map(self.product_category)
        )
        grouped = by_category.groupby(["category_id", "product_id"])["w"].sum()
        for category_id, group in grouped.groupby(level=0):
            scores = np.log1p(group.droplevel(0)).to_dict()
            self.category_scores[int(category_id)] = normalise_scores(scores)

        return self

    def recommend(self, context: RecommendationContext, k: int = 10) -> list[Scored]:
        """Category-weighted popularity when affinity is known, global otherwise."""
        if not context.category_affinity:
            return [item for item in self._global_ranked if item.product_id not in context.exclude][:k]

        blended: dict[int, float] = {}
        for category_id, affinity in context.category_affinity.items():
            for product_id, score in self.category_scores.get(int(category_id), {}).items():
                if product_id in context.exclude:
                    continue
                blended[product_id] = blended.get(product_id, 0.0) + affinity * score

        # Mix in a little global popularity so a user with narrow affinity is
        # not trapped inside two categories forever.
        for product_id, score in self.global_scores.items():
            if product_id in context.exclude:
                continue
            blended[product_id] = blended.get(product_id, 0.0) + 0.15 * score

        return top_k(blended, k, source=self.name)

    def score(
        self, context: RecommendationContext, product_ids: list[int]
    ) -> dict[int, float]:
        return {pid: self.global_scores.get(pid, 0.0) for pid in product_ids}


class TrendingRecommender(Recommender):
    """Time-decayed velocity.

    The half-life is the whole design. Six hours means a burst is visible
    almost immediately and mostly gone by the next day, which is what
    "trending" should mean on a storefront. A long half-life turns this into a
    slow popularity chart, which `PopularityRecommender` already is - two
    models computing the same thing is worse than one.
    """

    name = "trending"
    supports_cold_users = True

    def __init__(self, config: PopularityConfig | None = None) -> None:
        self.config = config or PopularityConfig()
        self.scores: dict[int, float] = {}
        self.category_scores: dict[int, dict[int, float]] = {}
        self.product_category: dict[int, int] = {}
        self._ranked: list[Scored] = []

    def fit(
        self,
        interactions: pd.DataFrame,
        products: pd.DataFrame,
        *,
        as_of: dt.datetime,
        window_multiplier: float = 7.0,
    ) -> TrendingRecommender:
        """Fit trending as of a point in time.

        `window_multiplier` widens the online window (24 hours, 6-hour
        half-life) for offline evaluation, where the test fold spans weeks
        rather than hours. Scaling both window and half-life together keeps the
        *shape* of the decay identical - it is the same model observed over a
        longer horizon, not a different one.
        """
        cutoff = pd.Timestamp(as_of)
        self.product_category = products.set_index("id")["category_id"].to_dict()

        window = pd.Timedelta(
            hours=self.config.trending_window_hours * window_multiplier
        )
        recent = interactions[
            (interactions["occurred_at"] < cutoff)
            & (interactions["occurred_at"] >= cutoff - window)
        ]
        if recent.empty:
            return self

        half_life_hours = self.config.trending_half_life_hours * window_multiplier
        age_hours = (cutoff - recent["occurred_at"]).dt.total_seconds() / 3600.0
        weights = recent["event_type"].map(POPULARITY_EVENT_WEIGHTS).fillna(0.0)
        decayed = weights * np.power(0.5, age_hours / half_life_hours)

        totals = decayed.groupby(recent["product_id"]).sum()
        self.scores = normalise_scores(totals[totals > 0].to_dict())
        self._ranked = top_k(self.scores, 2000, source=self.name)

        frame = pd.DataFrame(
            {
                "product_id": recent["product_id"].to_numpy(),
                "category_id": recent["product_id"].map(self.product_category).to_numpy(),
                "decayed": decayed.to_numpy(),
            }
        )
        grouped = frame.groupby(["category_id", "product_id"])["decayed"].sum()
        for category_id, group in grouped.groupby(level=0):
            if pd.isna(category_id):
                continue
            self.category_scores[int(category_id)] = normalise_scores(
                group.droplevel(0).to_dict()
            )
        return self

    def recommend(self, context: RecommendationContext, k: int = 10) -> list[Scored]:
        if not context.category_affinity:
            return [
                item for item in self._ranked if item.product_id not in context.exclude
            ][:k]
        blended: dict[int, float] = {}
        for category_id, affinity in context.category_affinity.items():
            for product_id, score in self.category_scores.get(int(category_id), {}).items():
                if product_id in context.exclude:
                    continue
                blended[product_id] = blended.get(product_id, 0.0) + affinity * score
        for product_id, score in self.scores.items():
            if product_id in context.exclude:
                continue
            blended[product_id] = blended.get(product_id, 0.0) + 0.25 * score
        return top_k(blended, k, source=self.name)

    def score(
        self, context: RecommendationContext, product_ids: list[int]
    ) -> dict[int, float]:
        return {pid: self.scores.get(pid, 0.0) for pid in product_ids}


__all__ = [
    "POPULARITY_EVENT_WEIGHTS",
    "PopularityRecommender",
    "TrendingRecommender",
]
