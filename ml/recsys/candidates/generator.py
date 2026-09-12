"""Candidate generation: merge, dedupe, filter (ADR-001 stage 1).

Runs every source, unions the results, keeps per-source provenance, applies
business rules, and hands roughly 300 products to the ranker.

**Provenance survives the merge.** A product retrieved by three sources keeps
all three names and all three scores. That matters twice: the ranker uses
"which sources found this" as a feature (agreement between independent sources
is strong evidence), and the serving log records it, so the dashboard can
report recall and conversion contribution per source - which is how a source
earns or loses its slot in the budget.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from recsys.candidates.sources import CandidateSource
from recsys.config.settings import CandidateConfig
from recsys.models.base import RecommendationContext

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CandidateSet:
    """The output of stage 1."""

    product_ids: list[int]
    scores_by_source: dict[str, dict[int, float]] = field(default_factory=dict)
    ranks_by_source: dict[str, dict[int, int]] = field(default_factory=dict)
    source_counts: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.product_ids)

    def sources_for(self, product_id: int) -> list[str]:
        return [
            source
            for source, scores in self.scores_by_source.items()
            if product_id in scores
        ]

    def score_from(self, source: str, product_id: int, default: float = 0.0) -> float:
        return self.scores_by_source.get(source, {}).get(product_id, default)

    def source_agreement(self) -> dict[int, int]:
        """How many independent sources retrieved each product."""
        counts: dict[int, int] = {}
        for scores in self.scores_by_source.values():
            for product_id in scores:
                counts[product_id] = counts.get(product_id, 0) + 1
        return counts


#: A business rule receives (product_id, context) and returns True to keep.
BusinessRule = Callable[[int, RecommendationContext], bool]


class CandidateGenerator:
    """Runs the configured sources and assembles the candidate pool."""

    def __init__(
        self,
        sources: dict[str, CandidateSource],
        *,
        config: CandidateConfig | None = None,
        budgets: dict[str, int] | None = None,
        business_rules: list[BusinessRule] | None = None,
    ) -> None:
        self.sources = sources
        self.config = config or CandidateConfig()
        self.budgets = budgets or self._default_budgets()
        self.business_rules = business_rules or []

    def _default_budgets(self) -> dict[str, int]:
        config = self.config
        return {
            "collaborative_als": config.collaborative,
            "content": config.content,
            "covisitation": config.covisitation,
            "frequently_bought_together": config.frequently_bought,
            "category_affinity": config.category_affinity,
            "brand_affinity": config.brand_affinity,
            "trending": config.trending,
            "popularity": config.popularity,
            "recently_viewed": config.recently_viewed,
            "exploration": max(
                int(config.total * config.exploration_fraction), 1
            ),
        }

    def generate(self, context: RecommendationContext) -> CandidateSet:
        scores_by_source: dict[str, dict[int, float]] = {}
        ranks_by_source: dict[str, dict[int, int]] = {}
        source_counts: dict[str, int] = {}
        union: dict[int, float] = {}

        for name, source in self.sources.items():
            limit = self.budgets.get(name, 50)
            try:
                candidates = source.generate(context, limit)
            except Exception:
                # One broken source must not take down retrieval. The pool
                # shrinks, the ranker still runs, and the failure is logged -
                # which is the whole point of the degradation ladder.
                logger.exception("candidate source %s failed", name)
                continue

            if not candidates:
                continue

            scores_by_source[name] = {c.product_id: c.score for c in candidates}
            ranks_by_source[name] = {c.product_id: c.rank for c in candidates}
            source_counts[name] = len(candidates)

            for candidate in candidates:
                # Reciprocal-rank fusion for the union ordering. Raw scores are
                # not comparable across sources (a cosine similarity and a lift
                # value live on different scales), but *ranks* always are - so
                # fusing on rank is the one merge that does not need every
                # source to be calibrated against every other.
                union[candidate.product_id] = union.get(candidate.product_id, 0.0) + 1.0 / (
                    60.0 + candidate.rank
                )

        filtered = [
            product_id
            for product_id in union
            if product_id not in context.exclude and self._passes_rules(product_id, context)
        ]
        ordered = sorted(filtered, key=lambda pid: (-union[pid], pid))[: self.config.total]

        return CandidateSet(
            product_ids=ordered,
            scores_by_source=scores_by_source,
            ranks_by_source=ranks_by_source,
            source_counts=source_counts,
        )

    def _passes_rules(self, product_id: int, context: RecommendationContext) -> bool:
        return all(rule(product_id, context) for rule in self.business_rules)


# ---------------------------------------------------------------------------
# Business rules
# ---------------------------------------------------------------------------


def in_stock_rule(stock: dict[int, int]) -> BusinessRule:
    """Never recommend something that cannot be bought."""

    def rule(product_id: int, _: RecommendationContext) -> bool:
        return stock.get(product_id, 0) > 0

    return rule


def active_product_rule(active: set[int]) -> BusinessRule:
    def rule(product_id: int, _: RecommendationContext) -> bool:
        return product_id in active

    return rule


def not_recently_purchased_rule(
    purchased: dict[int, set[int]], consumable: set[int]
) -> BusinessRule:
    """Suppress repeat recommendations of durable goods.

    Someone who bought a laptop last week does not want another laptop; someone
    who bought coffee beans does. The distinction is the product's repeat rate,
    so consumables are exempt from the rule rather than the rule being applied
    uniformly - a uniform rule would delete the entire replenishment use case,
    which is some of the most valuable recommendation traffic there is.
    """

    def rule(product_id: int, context: RecommendationContext) -> bool:
        if product_id in consumable:
            return True
        if context.user_id is None:
            return True
        return product_id not in purchased.get(context.user_id, set())

    return rule


def price_band_rule(
    percentiles: dict[int, float], *, max_distance: float = 0.55
) -> BusinessRule:
    """Drop candidates far outside the user's observed price range.

    Deliberately loose. A tight band would trap users in whatever they bought
    first and eliminate any chance of an upsell; this only removes the extremes,
    where a recommendation reads as a mistake rather than a suggestion.
    """

    def rule(product_id: int, context: RecommendationContext) -> bool:
        percentile = percentiles.get(product_id)
        if percentile is None:
            return True
        return abs(percentile - context.price_percentile) <= max_distance

    return rule


def recall_at_n(
    candidate_sets: dict[int, CandidateSet], targets: dict[int, set[int]]
) -> float:
    """Stage-1 recall: the metric this stage is actually optimised for.

    If an item never enters the candidate pool, no ranker can recover it. This
    is the ceiling on the whole system's performance, and it is measured
    separately from NDCG for exactly that reason.
    """
    scores = []
    for user_id, relevant in targets.items():
        if not relevant:
            continue
        candidates = candidate_sets.get(user_id)
        if candidates is None:
            scores.append(0.0)
            continue
        retrieved = set(candidates.product_ids)
        scores.append(len(retrieved & relevant) / len(relevant))
    return float(np.mean(scores)) if scores else 0.0


def source_contribution(
    candidate_sets: dict[int, CandidateSet], targets: dict[int, set[int]]
) -> dict[str, float]:
    """Per-source recall contribution - which sources earn their budget."""
    hits: dict[str, int] = {}
    total_relevant = 0
    for user_id, relevant in targets.items():
        candidates = candidate_sets.get(user_id)
        if candidates is None or not relevant:
            continue
        total_relevant += len(relevant)
        for source, scores in candidates.scores_by_source.items():
            hits[source] = hits.get(source, 0) + len(set(scores) & relevant)
    if total_relevant == 0:
        return {}
    return {
        source: count / total_relevant
        for source, count in sorted(hits.items(), key=lambda item: -item[1])
    }


__all__ = [
    "BusinessRule",
    "CandidateGenerator",
    "CandidateSet",
    "active_product_rule",
    "in_stock_rule",
    "not_recently_purchased_rule",
    "price_band_rule",
    "recall_at_n",
    "source_contribution",
]
