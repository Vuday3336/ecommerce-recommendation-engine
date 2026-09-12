"""Offline evaluation harness.

Runs a model over the held-out future window and reports accuracy metrics
alongside beyond-accuracy metrics, segmented by how much history each user had
(R-05). One aggregate number hides the thing that matters most: a model can
look excellent overall because it does well for the 20% of users with rich
history, while being useless for the 80% who are sparse - which is most real
traffic.

The harness never sees the test fold except as ground truth. Models are fitted
on train (and tuned on validation) by the caller; passing a model that was
fitted on test data is a leak this class cannot detect, which is why the split
itself is guarded in `preprocessing/splitting.py`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from recsys.config.settings import EvaluationConfig
from recsys.evaluation import metrics as rank_metrics
from recsys.models.base import RecommendationContext, Recommender
from recsys.preprocessing.splitting import TemporalSplit

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EvaluationResult:
    """Metrics for one model."""

    model: str
    metrics: dict[str, float]
    per_segment: dict[str, dict[str, float]] = field(default_factory=dict)
    n_users: int = 0
    elapsed_seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)
    #: Per-user scores for the primary metric, keyed by user id. Retained so
    #: two models can be compared with a *paired* bootstrap: user-to-user
    #: variance dwarfs the model-to-model difference, and pairing is what makes
    #: a 0.002 NDCG gap testable at this sample size.
    per_user_scores: dict[int, float] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {"model": self.model, "users": self.n_users}
        row.update({k: round(v, 5) for k, v in self.metrics.items()})
        row["seconds"] = round(self.elapsed_seconds, 2)
        return row


class Evaluator:
    """Scores recommenders against a temporal split."""

    def __init__(
        self,
        split: TemporalSplit,
        *,
        products: pd.DataFrame,
        config: EvaluationConfig | None = None,
        positive_events: tuple[str, ...] = ("PURCHASE", "ADD_TO_CART", "PRODUCT_CLICK"),
        graded_gains: dict[str, float] | None = None,
        target_fold: str = "test",
    ) -> None:
        self.split = split
        self.config = config or EvaluationConfig()
        self.products = products
        self.positive_events = positive_events
        # Hyperparameters are tuned against `validation` and reported against
        # `test`. Selecting on the test fold would make every reported number a
        # best-of-N maximum rather than an estimate of future performance -
        # the same leak as a random split, one level up.
        if target_fold not in {"validation", "test"}:
            raise ValueError("target_fold must be 'validation' or 'test'")
        self.target_fold = target_fold
        self.graded_gains = graded_gains or {
            "PURCHASE": 3.0,
            "ADD_TO_CART": 2.0,
            "PRODUCT_CLICK": 1.0,
        }

        self.catalogue_size = len(products)
        self.categories = products.set_index("id")["category_id"].to_dict()

        self._targets = self._build_targets()
        self._graded = self._build_graded_targets()
        self._train_counts = split.train.groupby("user_id").size().to_dict()
        self._seen = (
            split.train.groupby("user_id")["product_id"].apply(set).to_dict()
        )
        self._popularity_share = self._build_popularity_share()
        self._users = self._select_users()

        self._baseline_lists: dict[int, list[int]] = {}

    # -- setup ------------------------------------------------------------

    def _build_targets(self) -> dict[int, set[int]]:
        frame = getattr(self.split, self.target_fold)
        positives = frame[frame["event_type"].isin(self.positive_events)]
        if positives.empty:
            return {}
        return {
            int(user): {int(p) for p in products}
            for user, products in positives.groupby("user_id")["product_id"].apply(set).items()
        }

    def _build_graded_targets(self) -> dict[int, dict[int, float]]:
        frame = getattr(self.split, self.target_fold)
        positives = frame[frame["event_type"].isin(self.positive_events)].copy()
        if positives.empty:
            return {}
        positives["gain"] = positives["event_type"].map(self.graded_gains).fillna(0.0)
        best = positives.groupby(["user_id", "product_id"])["gain"].max()
        result: dict[int, dict[int, float]] = {}
        for (user_id, product_id), gain in best.items():
            result.setdefault(int(user_id), {})[int(product_id)] = float(gain)
        return result

    def _build_popularity_share(self) -> dict[int, float]:
        """Training-window popularity share, used for the novelty metric."""
        counts = self.split.train["product_id"].value_counts()
        total = float(counts.sum())
        if total <= 0:
            return {}
        return {int(pid): float(count / total) for pid, count in counts.items()}

    def _select_users(self) -> list[int]:
        eligible = sorted(self._targets)
        limit = self.config.max_eval_users
        if limit is None or len(eligible) <= limit:
            return eligible
        # Sampled, not truncated. Taking the first N by id would bias toward
        # early-registered users, who have systematically more history.
        rng = np.random.default_rng(self.config.random_state)
        chosen = rng.choice(len(eligible), size=limit, replace=False)
        return sorted(eligible[int(i)] for i in chosen)

    def _segment_for(self, user_id: int) -> str:
        count = self._train_counts.get(user_id, 0)
        for name, low, high in self.config.history_segments:
            if low <= count < high:
                return name
        return "rich"

    def context_for(
        self,
        user_id: int,
        *,
        category_affinity: dict[int, dict[int, float]] | None = None,
        brand_affinity: dict[int, dict[int, float]] | None = None,
        price_percentiles: dict[int, float] | None = None,
    ) -> RecommendationContext:
        seen = self._seen.get(user_id, set())
        return RecommendationContext(
            user_id=int(user_id),
            seen_products=tuple(sorted(seen)),
            category_affinity=(category_affinity or {}).get(user_id, {}),
            brand_affinity=(brand_affinity or {}).get(user_id, {}),
            price_percentile=(price_percentiles or {}).get(user_id, 0.5),
            interaction_count=self._train_counts.get(user_id, 0),
            # Already-seen products are excluded from retrieval. Re-recommending
            # something the user already interacted with in training is
            # trivially "correct" for repeat-purchase items and would flatter
            # every model; the interesting question is what it finds next.
            exclude=frozenset(seen),
        )

    # -- evaluation -------------------------------------------------------

    def evaluate(
        self,
        model: Recommender,
        *,
        k: int | None = None,
        category_affinity: dict[int, dict[int, float]] | None = None,
        brand_affinity: dict[int, dict[int, float]] | None = None,
        price_percentiles: dict[int, float] | None = None,
        record_baseline: bool = False,
        name: str | None = None,
    ) -> EvaluationResult:
        k = k or self.config.primary_k
        max_k = max(self.config.k_values)
        started = time.perf_counter()

        per_user: list[dict[str, float]] = []
        per_user_scores: dict[int, float] = {}
        segments: dict[str, list[dict[str, float]]] = {}
        all_lists: list[list[int]] = []
        exposure: dict[int, int] = {}

        for user_id in self._users:
            context = self.context_for(
                user_id,
                category_affinity=category_affinity,
                brand_affinity=brand_affinity,
                price_percentiles=price_percentiles,
            )
            ranked = [item.product_id for item in model.recommend(context, k=max_k)]
            if record_baseline:
                self._baseline_lists[user_id] = ranked

            relevant = self._targets.get(user_id, set())
            gains = self._graded.get(user_id, {})
            row = self._score_user(user_id, ranked, relevant, gains, k, max_k)

            per_user.append(row)
            per_user_scores[int(user_id)] = row.get(f"ndcg@{k}", 0.0)
            segments.setdefault(self._segment_for(user_id), []).append(row)
            all_lists.append(ranked[:k])
            for product_id in ranked[:k]:
                exposure[product_id] = exposure.get(product_id, 0) + 1

        aggregate = self._aggregate(per_user)
        aggregate["coverage"] = rank_metrics.catalogue_coverage(all_lists, self.catalogue_size)
        aggregate["gini_exposure"] = rank_metrics.gini_coefficient(exposure.values())
        aggregate["personalisation"] = rank_metrics.personalisation(all_lists, k=k)

        return EvaluationResult(
            model=name or model.name,
            metrics=aggregate,
            per_segment={
                segment: self._aggregate(rows) for segment, rows in sorted(segments.items())
            },
            n_users=len(per_user),
            elapsed_seconds=time.perf_counter() - started,
            per_user_scores=per_user_scores,
        )

    def _score_user(
        self,
        user_id: int,
        ranked: list[int],
        relevant: set[int],
        gains: dict[int, float],
        k: int,
        max_k: int,
    ) -> dict[str, float]:
        row: dict[str, float] = {}
        for kk in self.config.k_values:
            row[f"precision@{kk}"] = rank_metrics.precision_at_k(ranked, relevant, kk)
            row[f"recall@{kk}"] = rank_metrics.recall_at_k(ranked, relevant, kk)
            row[f"ndcg@{kk}"] = rank_metrics.ndcg_at_k(ranked, relevant, kk)
            row[f"map@{kk}"] = rank_metrics.average_precision_at_k(ranked, relevant, kk)
            row[f"hit_rate@{kk}"] = rank_metrics.hit_rate_at_k(ranked, relevant, kk)
        row["mrr"] = rank_metrics.reciprocal_rank(ranked, relevant, max_k)
        row[f"graded_ndcg@{k}"] = rank_metrics.graded_ndcg_at_k(ranked, gains, k)
        row["diversity"] = rank_metrics.intra_list_diversity(ranked[:k], categories=self.categories)
        row["novelty"] = rank_metrics.novelty(ranked[:k], self._popularity_share)
        baseline = self._baseline_lists.get(user_id, [])
        row["serendipity"] = (
            rank_metrics.serendipity(ranked, relevant, baseline, k) if baseline else 0.0
        )
        return row

    @staticmethod
    def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
        if not rows:
            return {}
        frame = pd.DataFrame(rows)
        return {column: float(frame[column].mean()) for column in frame.columns}


def comparison_table(results: list[EvaluationResult]) -> pd.DataFrame:
    """Side-by-side comparison, sorted by the primary metric."""
    frame = pd.DataFrame([result.to_row() for result in results])
    if "ndcg@10" in frame.columns:
        frame = frame.sort_values("ndcg@10", ascending=False)
    return frame.reset_index(drop=True)


def segment_table(results: list[EvaluationResult], metric: str = "ndcg@10") -> pd.DataFrame:
    """One metric across models and history-depth segments."""
    rows = []
    for result in results:
        row = {"model": result.model}
        for segment, values in result.per_segment.items():
            row[segment] = round(values.get(metric, float("nan")), 5)
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = ["EvaluationResult", "Evaluator", "comparison_table", "segment_table"]
