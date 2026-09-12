"""Stage-1 candidate sources (ADR-001).

Each source is cheap, recall-oriented, and answers one question. Together they
should contain almost everything the user might plausibly want; the ranker
then decides the order. Stage 1 is measured by **recall@N**, not by precision -
a source that contributes one relevant item buried at position 200 has done its
job, because stage 2 will lift it.

Every source implements the same protocol, which is the seam that lets a neural
two-tower retriever be added later without touching the ranker or the generator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from recsys.models.base import RecommendationContext, Recommender

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class Candidate:
    """One retrieved product, with provenance."""

    product_id: int
    score: float
    source: str
    rank: int


@runtime_checkable
class CandidateSource(Protocol):
    name: str

    def generate(self, context: RecommendationContext, limit: int) -> list[Candidate]: ...


class ModelSource:
    """Adapts any `Recommender` into a candidate source."""

    def __init__(self, model: Recommender, name: str | None = None) -> None:
        self.model = model
        self.name = name or model.name

    def generate(self, context: RecommendationContext, limit: int) -> list[Candidate]:
        return [
            Candidate(item.product_id, item.score, self.name, rank)
            for rank, item in enumerate(self.model.recommend(context, k=limit))
        ]


class ItemNeighbourSource:
    """Expands the user's recent products through an item-item neighbour map.

    Used for co-visitation and frequently-bought-together, where the natural
    query is "given these anchors, what else?" rather than "given this user".
    """

    def __init__(
        self,
        neighbours: dict[int, list[tuple[int, float]]],
        name: str,
        *,
        max_anchors: int = 10,
        anchor_decay: float = 0.85,
    ) -> None:
        self.neighbours = neighbours
        self.name = name
        self.max_anchors = max_anchors
        self.anchor_decay = anchor_decay

    def generate(self, context: RecommendationContext, limit: int) -> list[Candidate]:
        anchors = context.recent_products or context.seen_products
        if not anchors:
            return []
        scores: dict[int, float] = {}
        for position, anchor in enumerate(anchors[: self.max_anchors]):
            decay = self.anchor_decay**position
            for product_id, strength in self.neighbours.get(int(anchor), []):
                if product_id in context.exclude:
                    continue
                scores[product_id] = scores.get(product_id, 0.0) + decay * strength

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
        return [
            Candidate(int(pid), float(score), self.name, rank)
            for rank, (pid, score) in enumerate(ranked)
        ]


class CategoryAffinitySource:
    """Top products from the categories the user actually engages with.

    Simple and strong. On this dataset it is the single most productive source,
    which is worth stating plainly: a large part of "personalisation" in
    e-commerce is knowing which aisle someone shops in.
    """

    name = "category_affinity"

    def __init__(
        self,
        category_top_products: dict[int, list[tuple[int, float]]],
        *,
        max_categories: int = 6,
    ) -> None:
        self.category_top_products = category_top_products
        self.max_categories = max_categories

    def generate(self, context: RecommendationContext, limit: int) -> list[Candidate]:
        if not context.category_affinity:
            return []
        ordered = sorted(
            context.category_affinity.items(), key=lambda item: -item[1]
        )[: self.max_categories]

        scores: dict[int, float] = {}
        for category_id, affinity in ordered:
            for product_id, score in self.category_top_products.get(int(category_id), []):
                if product_id in context.exclude:
                    continue
                scores[product_id] = max(scores.get(product_id, 0.0), affinity * score)

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
        return [
            Candidate(int(pid), float(score), self.name, rank)
            for rank, (pid, score) in enumerate(ranked)
        ]


class BrandAffinitySource:
    """Top products from brands the user favours."""

    name = "brand_affinity"

    def __init__(
        self,
        brand_top_products: dict[int, list[tuple[int, float]]],
        *,
        max_brands: int = 5,
    ) -> None:
        self.brand_top_products = brand_top_products
        self.max_brands = max_brands

    def generate(self, context: RecommendationContext, limit: int) -> list[Candidate]:
        if not context.brand_affinity:
            return []
        ordered = sorted(context.brand_affinity.items(), key=lambda item: -item[1])[
            : self.max_brands
        ]
        scores: dict[int, float] = {}
        for brand_id, affinity in ordered:
            for product_id, score in self.brand_top_products.get(int(brand_id), []):
                if product_id in context.exclude:
                    continue
                scores[product_id] = max(scores.get(product_id, 0.0), affinity * score)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
        return [
            Candidate(int(pid), float(score), self.name, rank)
            for rank, (pid, score) in enumerate(ranked)
        ]


class StaticListSource:
    """A fixed ranked list - global popularity or trending.

    The last line of the degradation ladder: it needs no user context, no
    model artefact and no database, so it can always produce something.
    """

    def __init__(self, ranked: list[tuple[int, float]], name: str) -> None:
        self.ranked = ranked
        self.name = name

    def generate(self, context: RecommendationContext, limit: int) -> list[Candidate]:
        out: list[Candidate] = []
        for product_id, score in self.ranked:
            if product_id in context.exclude:
                continue
            out.append(Candidate(int(product_id), float(score), self.name, len(out)))
            if len(out) >= limit:
                break
        return out


class RecentlyViewedSource:
    """The user's own recent products.

    Included as a candidate source rather than a separate rail because the
    ranker should be free to decide that returning to something is the best
    next action - which for considered purchases it often is.
    """

    name = "recently_viewed"

    def generate(self, context: RecommendationContext, limit: int) -> list[Candidate]:
        return [
            Candidate(int(pid), 1.0 - (rank * 0.01), self.name, rank)
            for rank, pid in enumerate(context.recent_products[:limit])
            if pid not in context.exclude
        ]


class ExplorationSource:
    """A small budget of cold items, sampled deterministically.

    This is the counter-measure to the popularity feedback loop (R-06). Without
    a guaranteed slice of impressions, a new product can never accumulate the
    interactions it needs to be recommended, and the catalogue ossifies around
    whatever was popular when the system launched.

    Sampling is seeded per user so the same user sees a stable exploration set
    rather than a different random product on every page load.
    """

    name = "exploration"

    def __init__(self, cold_products: np.ndarray, *, seed: int = 42) -> None:
        self.cold_products = cold_products
        self.seed = seed

    def generate(self, context: RecommendationContext, limit: int) -> list[Candidate]:
        if limit <= 0 or len(self.cold_products) == 0:
            return []
        rng = np.random.default_rng(self.seed + (context.user_id or 0))
        size = min(limit, len(self.cold_products))
        chosen = rng.choice(self.cold_products, size=size, replace=False)
        return [
            Candidate(int(pid), 0.01, self.name, rank)
            for rank, pid in enumerate(chosen)
            if int(pid) not in context.exclude
        ]


__all__ = [
    "BrandAffinitySource",
    "Candidate",
    "CandidateSource",
    "CategoryAffinitySource",
    "ExplorationSource",
    "ItemNeighbourSource",
    "ModelSource",
    "RecentlyViewedSource",
    "StaticListSource",
]
