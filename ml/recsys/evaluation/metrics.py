"""Recommendation metrics.

Accuracy metrics answer "did we rank the right things highly?". Beyond-accuracy
metrics answer "is the result any good as a product?" - and a system optimised
only for the first reliably degrades on the second: the easiest way to raise
NDCG is to recommend the same popular items to everyone, which is exactly the
popularity bias R-06 warns about. Both sets are therefore gate metrics.

Every function takes a ranked list of product ids and a set of relevant ones,
so they compose over any model without knowing anything about it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np


def precision_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """Fraction of the top-k that were relevant."""
    if k <= 0:
        return 0.0
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for item in top if item in relevant) / k


def recall_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """Fraction of relevant items that appeared in the top-k."""
    if not relevant:
        return 0.0
    top = ranked[:k]
    return sum(1 for item in top if item in relevant) / len(relevant)


def average_precision_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """AP@K - precision averaged at each relevant hit.

    Rewards putting relevant items *early*, not merely inside the window, which
    is what distinguishes it from precision@k.
    """
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for index, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / index
    return score / min(len(relevant), k)


def dcg_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    return sum(
        1.0 / math.log2(index + 1)
        for index, item in enumerate(ranked[:k], start=1)
        if item in relevant
    )


def ndcg_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """Normalised DCG - the primary ranking metric.

    Chosen as primary because it is position-weighted (a hit at rank 1 counts
    far more than one at rank 10, which matches how a rail is actually
    consumed) and because it is what the LambdaRank objective optimises, so
    training and reporting agree.
    """
    if not relevant:
        return 0.0
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    if ideal == 0:
        return 0.0
    return dcg_at_k(ranked, relevant, k) / ideal


def graded_ndcg_at_k(
    ranked: Sequence[int], gains: dict[int, float], k: int
) -> float:
    """NDCG with graded relevance (purchase > cart > click)."""
    if not gains:
        return 0.0
    dcg = sum(
        (2 ** gains.get(item, 0.0) - 1) / math.log2(index + 1)
        for index, item in enumerate(ranked[:k], start=1)
    )
    ideal_gains = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(
        (2**gain - 1) / math.log2(index + 1)
        for index, gain in enumerate(ideal_gains, start=1)
    )
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """Did we get at least one right? The most user-facing metric here."""
    return 1.0 if any(item in relevant for item in ranked[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    for index, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


# ---------------------------------------------------------------------------
# Beyond accuracy
# ---------------------------------------------------------------------------


def catalogue_coverage(
    all_recommendations: Iterable[Sequence[int]], catalogue_size: int
) -> float:
    """Share of the catalogue that ever gets recommended.

    A model with 3% coverage is a merchandising rule with extra steps: it has
    decided ninety-seven percent of the catalogue is unsellable.
    """
    if catalogue_size <= 0:
        return 0.0
    seen: set[int] = set()
    for ranked in all_recommendations:
        seen.update(ranked)
    return len(seen) / catalogue_size


def gini_coefficient(counts: Iterable[float]) -> float:
    """Concentration of recommendation exposure across products.

    Coverage says how many products are shown at all; Gini says whether
    exposure is spread or hoarded by a handful. A model can cover most of the
    catalogue and still send 90% of impressions to fifty items.
    """
    values = np.sort(np.asarray(list(counts), dtype=float))
    n = len(values)
    if n == 0:
        return 0.0
    total = values.sum()
    if total <= 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * (index * values).sum()) / (n * total) - (n + 1.0) / n)


def intra_list_diversity(
    ranked: Sequence[int], similarity: dict[tuple[int, int], float] | None = None,
    *, categories: dict[int, int] | None = None,
) -> float:
    """1 - mean pairwise similarity inside one recommendation list.

    When an embedding similarity matrix is unavailable, category distinctness
    is used instead. It is coarser but has the property that matters: a rail of
    five near-identical running shoes scores badly.
    """
    items = list(ranked)
    if len(items) < 2:
        return 0.0

    if categories is not None:
        cats = [categories.get(item) for item in items]
        pairs = [
            0.0 if cats[i] != cats[j] else 1.0
            for i in range(len(items))
            for j in range(i + 1, len(items))
        ]
        return 1.0 - (sum(pairs) / len(pairs))

    if not similarity:
        return 0.0
    pairs = [
        similarity.get((items[i], items[j]), similarity.get((items[j], items[i]), 0.0))
        for i in range(len(items))
        for j in range(i + 1, len(items))
    ]
    return 1.0 - (sum(pairs) / len(pairs)) if pairs else 0.0


def novelty(ranked: Sequence[int], popularity: dict[int, float]) -> float:
    """Mean self-information of the recommended items.

    Recommending the bestseller to everyone is correct and useless. Novelty
    measures how much the list tells the user something they would not have
    found on the homepage.
    """
    if not ranked:
        return 0.0
    scores = []
    for item in ranked:
        share = popularity.get(item, 0.0)
        scores.append(-math.log2(share) if share > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def serendipity(
    ranked: Sequence[int],
    relevant: set[int],
    baseline: Sequence[int],
    k: int,
) -> float:
    """Relevant hits the popularity baseline would *not* have produced.

    This is the metric that captures the actual value of personalisation. A
    model that only re-finds what popularity already surfaces adds nothing,
    however good its NDCG looks.
    """
    baseline_set = set(baseline[:k])
    top = ranked[:k]
    if not top:
        return 0.0
    unexpected_hits = sum(
        1 for item in top if item in relevant and item not in baseline_set
    )
    return unexpected_hits / k


def personalisation(all_recommendations: Sequence[Sequence[int]], k: int = 10) -> float:
    """Mean pairwise dissimilarity between different users' lists.

    Zero means every user sees the same thing - the system is not personalised,
    whatever the NDCG says. Sampled rather than exhaustive: the exhaustive
    computation is O(users^2).
    """
    lists = [set(ranked[:k]) for ranked in all_recommendations if ranked]
    if len(lists) < 2:
        return 0.0
    rng = np.random.default_rng(42)
    sample_size = min(len(lists), 400)
    indices = rng.choice(len(lists), size=sample_size, replace=False)
    sampled = [lists[int(i)] for i in indices]

    scores = []
    for i in range(len(sampled)):
        for j in range(i + 1, len(sampled)):
            union = sampled[i] | sampled[j]
            if not union:
                continue
            overlap = len(sampled[i] & sampled[j]) / len(union)
            scores.append(1.0 - overlap)
    return float(np.mean(scores)) if scores else 0.0


__all__ = [
    "average_precision_at_k",
    "catalogue_coverage",
    "dcg_at_k",
    "gini_coefficient",
    "graded_ndcg_at_k",
    "hit_rate_at_k",
    "intra_list_diversity",
    "ndcg_at_k",
    "novelty",
    "personalisation",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "serendipity",
]
