"""Common interfaces for recommendation models.

Every model implements `Recommender`, so the evaluation harness compares them
without special-casing, and the hybrid blends them without knowing what they
are. This is also the seam ADR-001 depends on: a neural retriever added later
implements the same two methods and drops into the candidate generator with no
other change.

`score` and `recommend` are separate because they answer different questions.
`recommend` produces a ranked list (retrieval). `score` evaluates a specific set
of items (ranking), which is what the hybrid and the stage-2 ranker need.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True, frozen=True)
class Scored:
    """One scored product."""

    product_id: int
    score: float
    source: str = "unknown"

    def __iter__(self):
        yield self.product_id
        yield self.score


@dataclass(slots=True)
class RecommendationContext:
    """Everything a model may need about the request.

    Passed by value so models stay pure functions of their inputs, which is
    what makes them testable without a database or a running application.
    """

    user_id: int | None = None
    session_key: str | None = None
    seen_products: tuple[int, ...] = ()
    recent_products: tuple[int, ...] = ()
    category_affinity: dict[int, float] = field(default_factory=dict)
    brand_affinity: dict[int, float] = field(default_factory=dict)
    price_percentile: float = 0.5
    interaction_count: int = 0
    exclude: frozenset[int] = frozenset()

    @property
    def is_cold(self) -> bool:
        return self.interaction_count == 0


class Recommender(ABC):
    """Base class for every recommendation strategy."""

    name: str = "recommender"
    #: Whether the model can produce anything for a user it has never seen.
    supports_cold_users: bool = False

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> Recommender:
        """Train from data. Returns self so calls can chain."""

    @abstractmethod
    def recommend(
        self, context: RecommendationContext, k: int = 10
    ) -> list[Scored]:
        """Return the top-k products for this context."""

    def score(
        self, context: RecommendationContext, product_ids: list[int]
    ) -> dict[int, float]:
        """Score a specific candidate set.

        The default implementation asks for a long recommendation list and
        reads scores out of it. Models that can score directly should override
        this - for a candidate set of 300 the default wastes work.
        """
        ranked = {item.product_id: item.score for item in self.recommend(context, k=10_000)}
        return {pid: ranked.get(pid, 0.0) for pid in product_ids}

    # -- persistence ------------------------------------------------------

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, path: Path) -> Recommender:
        with Path(path).open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError(f"{path} contains {type(model).__name__}, not {cls.__name__}")
        return model


def normalise_scores(scores: dict[int, float]) -> dict[int, float]:
    """Min-max normalise to [0, 1].

    Necessary before blending: an ALS dot product, a cosine similarity and a
    log-popularity value live on completely different scales, and summing them
    with weights would make the weights meaningless - the largest-magnitude
    component would dominate regardless of its weight (ADR-014).
    """
    if not scores:
        return {}
    values = np.fromiter(scores.values(), dtype=float, count=len(scores))
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return dict.fromkeys(scores, 0.0 if high == 0.0 else 1.0)
    span = high - low
    return {pid: (value - low) / span for pid, value in scores.items()}


def top_k(scores: dict[int, float], k: int, *, source: str = "unknown") -> list[Scored]:
    """Sort and truncate. Ties break on product id so results are deterministic."""
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [Scored(int(pid), float(score), source) for pid, score in ranked[:k]]


__all__ = [
    "RecommendationContext",
    "Recommender",
    "Scored",
    "normalise_scores",
    "top_k",
]
