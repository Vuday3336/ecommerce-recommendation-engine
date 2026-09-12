"""Co-visitation and frequently-bought-together.

Two item-item models built from co-occurrence rather than factorisation. They
matter because they answer questions the other models cannot:

* **Co-visitation** (FR-04, "customers who viewed this also viewed") captures
  *substitutes* - products a shopper compares against each other in one
  session. Session-scoped, so it works from the very first page view and is the
  strongest signal an anonymous visitor generates.
* **Frequently bought together** (FR-03) captures *complements* - products that
  appear in the same order. Basket-scoped, and deliberately not the same thing:
  two running shoes are substitutes and should never be sold as a pair, while
  shoes and socks are complements and should.

Conflating the two produces the classic broken rail: "customers who bought this
phone also bought... these four other phones".

**Normalisation.** Raw co-occurrence counts are dominated by popularity - the
bestseller co-occurs with everything. Both models therefore score by *lift*
(observed co-occurrence over what independence would predict) with a support
floor, so a pair needs both a real association and enough evidence.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import pandas as pd

from recsys.models.base import (
    RecommendationContext,
    Recommender,
    Scored,
    normalise_scores,
    top_k,
)

logger = logging.getLogger(__name__)


class CoVisitationRecommender(Recommender):
    """Session co-occurrence: substitutes and comparison sets."""

    name = "covisitation"
    supports_cold_users = True

    def __init__(
        self,
        *,
        max_session_length: int = 30,
        min_support: int = 3,
        top_neighbours: int = 60,
    ) -> None:
        self.max_session_length = max_session_length
        self.min_support = min_support
        self.top_neighbours = top_neighbours
        self.neighbours: dict[int, list[tuple[int, float]]] = {}

    def fit(
        self,
        interactions: pd.DataFrame,
        *,
        event_types: tuple[str, ...] = ("PRODUCT_VIEW", "PRODUCT_CLICK"),
        **_: object,
    ) -> CoVisitationRecommender:
        frame = interactions[interactions["event_type"].isin(event_types)]
        if frame.empty:
            return self

        sessions = frame.groupby("session_key")["product_id"].apply(
            lambda s: list(dict.fromkeys(s))
        )

        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        item_counts: dict[int, int] = defaultdict(int)
        total_sessions = 0

        for products in sessions:
            # A session that browsed fifty products is a crawler or a lost
            # user; either way it contributes O(n^2) noisy pairs and would
            # dominate the matrix.
            if len(products) < 2 or len(products) > self.max_session_length:
                continue
            total_sessions += 1
            unique = [int(p) for p in products]
            for item in unique:
                item_counts[item] += 1
            for i in range(len(unique)):
                for j in range(i + 1, len(unique)):
                    a, b = unique[i], unique[j]
                    key = (a, b) if a < b else (b, a)
                    pair_counts[key] += 1

        self.neighbours = self._score_pairs(
            pair_counts, item_counts, total_sessions, self.min_support
        )
        logger.info(
            "co-visitation: %d sessions, %d pairs, %d items with neighbours",
            total_sessions,
            len(pair_counts),
            len(self.neighbours),
        )
        return self

    def _score_pairs(
        self,
        pair_counts: dict[tuple[int, int], int],
        item_counts: dict[int, int],
        total: int,
        min_support: int,
    ) -> dict[int, list[tuple[int, float]]]:
        """Convert counts to lift, keeping the strongest neighbours per item."""
        if total == 0:
            return {}
        scored: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for (a, b), count in pair_counts.items():
            if count < min_support:
                continue
            expected = (item_counts[a] / total) * (item_counts[b] / total) * total
            if expected <= 0:
                continue
            lift = count / expected
            scored[a].append((b, lift))
            scored[b].append((a, lift))

        return {
            item: sorted(pairs, key=lambda p: -p[1])[: self.top_neighbours]
            for item, pairs in scored.items()
        }

    def also_viewed(self, product_id: int, k: int = 10) -> list[Scored]:
        pairs = self.neighbours.get(int(product_id), [])[:k]
        return [Scored(int(pid), float(score), self.name) for pid, score in pairs]

    def recommend(self, context: RecommendationContext, k: int = 10) -> list[Scored]:
        anchors = context.recent_products or context.seen_products
        if not anchors:
            return []

        scores: dict[int, float] = {}
        # Recent anchors count for more: the most recently viewed product is
        # the best statement of what the user is doing right now.
        for position, anchor in enumerate(anchors[:20]):
            decay = 0.85**position
            for product_id, lift in self.neighbours.get(int(anchor), []):
                if product_id in context.exclude:
                    continue
                scores[product_id] = scores.get(product_id, 0.0) + decay * lift
        return top_k(normalise_scores(scores), k, source=self.name)

    def score(
        self, context: RecommendationContext, product_ids: list[int]
    ) -> dict[int, float]:
        ranked = {item.product_id: item.score for item in self.recommend(context, k=5000)}
        return {pid: ranked.get(pid, 0.0) for pid in product_ids}


class FrequentlyBoughtTogether(Recommender):
    """Basket co-occurrence: complements.

    Fitted from `order_items` rather than from purchase events, because the
    order is the basket. Deriving baskets by bucketing purchase events into
    time windows would be an approximation of something we already store
    exactly.
    """

    name = "frequently_bought_together"
    supports_cold_users = True

    def __init__(self, *, min_support: int = 2, top_neighbours: int = 30) -> None:
        self.min_support = min_support
        self.top_neighbours = top_neighbours
        self.neighbours: dict[int, list[tuple[int, float]]] = {}
        self.pair_support: dict[tuple[int, int], int] = {}

    def fit(self, order_items: pd.DataFrame, **_: object) -> FrequentlyBoughtTogether:
        if order_items.empty:
            return self

        baskets = order_items.groupby("order_id")["product_id"].apply(
            lambda s: sorted({int(p) for p in s})
        )
        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        item_counts: dict[int, int] = defaultdict(int)
        total = 0

        for products in baskets:
            if len(products) < 2:
                # Single-item orders still count toward the marginal, which is
                # the denominator of the lift. Dropping them would inflate every
                # lift value by shrinking the base rate.
                total += 1
                for item in products:
                    item_counts[item] += 1
                continue
            total += 1
            for item in products:
                item_counts[item] += 1
            for i in range(len(products)):
                for j in range(i + 1, len(products)):
                    pair_counts[(products[i], products[j])] += 1

        self.pair_support = dict(pair_counts)
        scored: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for (a, b), count in pair_counts.items():
            if count < self.min_support:
                continue
            expected = (item_counts[a] / total) * (item_counts[b] / total) * total
            if expected <= 0:
                continue
            lift = count / expected
            scored[a].append((b, lift))
            scored[b].append((a, lift))

        self.neighbours = {
            item: sorted(pairs, key=lambda p: -p[1])[: self.top_neighbours]
            for item, pairs in scored.items()
        }
        logger.info(
            "FBT: %d baskets, %d qualifying pairs, %d items with complements",
            total,
            sum(1 for c in pair_counts.values() if c >= self.min_support),
            len(self.neighbours),
        )
        return self

    def bought_together(self, product_id: int, k: int = 5) -> list[Scored]:
        pairs = self.neighbours.get(int(product_id), [])[:k]
        return [Scored(int(pid), float(score), self.name) for pid, score in pairs]

    def recommend(self, context: RecommendationContext, k: int = 10) -> list[Scored]:
        anchors = context.recent_products or context.seen_products
        if not anchors:
            return []
        scores: dict[int, float] = {}
        for anchor in anchors[:20]:
            for product_id, lift in self.neighbours.get(int(anchor), []):
                if product_id in context.exclude:
                    continue
                scores[product_id] = max(scores.get(product_id, 0.0), lift)
        return top_k(normalise_scores(scores), k, source=self.name)


__all__ = ["CoVisitationRecommender", "FrequentlyBoughtTogether"]
