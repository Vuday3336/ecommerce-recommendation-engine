"""Collaborative filtering: ALS in production, BPR for comparison (ADR-003).

**Why ALS.** It is built for exactly this data shape - implicit,
confidence-weighted, sparse. The alternating solve is closed-form, so it is
fast, deterministic given a seed (NFR-05 needs that), and parallel across
cores. A 7k x 5k matrix factorises in seconds, which means nightly retraining
is cheap rather than an event. It also yields item factors directly usable for
item-item similarity, which serves FR-02 and FR-04 for free.

**The confidence formulation.** ALS with implicit feedback minimises

    sum over all (u, i) of  c_ui * (p_ui - x_u . y_i)^2  +  lambda * (||x||^2 + ||y||^2)

where `p_ui` is 1 if the pair was observed and 0 otherwise, and
`c_ui = 1 + alpha * r_ui`. Crucially the sum runs over *every* cell, not just
observed ones - unobserved pairs are weak negatives with confidence 1. That is
what makes it appropriate here: we never see "user dislikes item", only
absence, and absence is weak evidence rather than no evidence.

`r_ui` is the calibrated weight from ADR-002, which is why the weighting work
belongs upstream of this model rather than inside it.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import sparse

from recsys.config.settings import CollaborativeConfig
from recsys.models.base import (
    RecommendationContext,
    Recommender,
    Scored,
    normalise_scores,
    top_k,
)

logger = logging.getLogger(__name__)


class _MatrixIndex:
    """Maps between database ids and contiguous matrix row/column indices."""

    def __init__(self, user_ids: np.ndarray, product_ids: np.ndarray) -> None:
        self.user_ids = user_ids
        self.product_ids = product_ids
        self.user_index = {int(u): i for i, u in enumerate(user_ids)}
        self.product_index = {int(p): i for i, p in enumerate(product_ids)}

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.user_ids), len(self.product_ids)


def build_sparse_matrix(
    weights: pd.DataFrame,
    *,
    min_item_interactions: int = 2,
) -> tuple[sparse.csr_matrix, _MatrixIndex]:
    """Build the user-item confidence matrix from the weighted interaction frame.

    Items below `min_item_interactions` are dropped from the *factorisation*
    only. They remain fully recommendable through the content path - excluding
    them here is about not fitting a 96-dimensional factor to a single
    observation, which produces a vector made entirely of regularisation noise.
    """
    frame = weights
    if min_item_interactions > 1:
        counts = frame.groupby("product_id").size()
        keep = counts[counts >= min_item_interactions].index
        frame = frame[frame["product_id"].isin(keep)]

    if frame.empty:
        empty = sparse.csr_matrix((0, 0), dtype=np.float32)
        return empty, _MatrixIndex(np.array([]), np.array([]))

    user_ids = np.sort(frame["user_id"].unique())
    product_ids = np.sort(frame["product_id"].unique())
    index = _MatrixIndex(user_ids, product_ids)

    rows = frame["user_id"].map(index.user_index).to_numpy()
    cols = frame["product_id"].map(index.product_index).to_numpy()
    values = frame["weight"].to_numpy(dtype=np.float32)

    matrix = sparse.csr_matrix(
        (values, (rows, cols)), shape=index.shape, dtype=np.float32
    )
    return matrix, index


class ALSRecommender(Recommender):
    """Alternating Least Squares over implicit confidence weights."""

    name = "collaborative_als"
    supports_cold_users = False

    def __init__(self, config: CollaborativeConfig | None = None) -> None:
        self.config = config or CollaborativeConfig()
        self.model = None
        self.index: _MatrixIndex | None = None
        self.matrix: sparse.csr_matrix | None = None
        self._item_norms: np.ndarray | None = None

    def fit(self, weights: pd.DataFrame, *, min_item_interactions: int = 2) -> ALSRecommender:
        from implicit.als import AlternatingLeastSquares

        matrix, index = build_sparse_matrix(
            weights, min_item_interactions=min_item_interactions
        )
        if matrix.shape[0] == 0:
            logger.warning("no interactions to factorise")
            return self

        self.matrix = matrix
        self.index = index

        self.model = AlternatingLeastSquares(
            factors=self.config.factors,
            regularization=self.config.regularization,
            iterations=self.config.iterations,
            alpha=self.config.alpha,
            use_native=self.config.use_native,
            calculate_training_loss=self.config.calculate_training_loss,
            random_state=self.config.random_state,
        )
        self.model.fit(matrix, show_progress=False)

        item_factors = np.asarray(self.model.item_factors)
        norms = np.linalg.norm(item_factors, axis=1)
        self._item_norms = np.where(norms > 0, norms, 1.0)

        logger.info(
            "ALS fitted: %d users x %d items, %d factors, density %.4f%%",
            matrix.shape[0],
            matrix.shape[1],
            self.config.factors,
            100.0 * matrix.nnz / (matrix.shape[0] * matrix.shape[1]),
        )
        return self

    # -- inference --------------------------------------------------------

    def _user_vector(self, context: RecommendationContext) -> np.ndarray | None:
        if self.model is None or self.index is None:
            return None
        if context.user_id is not None:
            row = self.index.user_index.get(int(context.user_id))
            if row is not None:
                return np.asarray(self.model.user_factors[row])

        # Fold-in for a user absent from the training matrix: approximate their
        # vector as the mean of the item factors they have touched. This is not
        # a full fold-in solve, but it costs microseconds and is the difference
        # between serving a session-based recommendation and serving nothing.
        indices = [
            self.index.product_index[pid]
            for pid in context.recent_products or context.seen_products
            if pid in self.index.product_index
        ]
        if not indices:
            return None
        return np.asarray(self.model.item_factors[indices]).mean(axis=0)

    def recommend(self, context: RecommendationContext, k: int = 10) -> list[Scored]:
        vector = self._user_vector(context)
        if vector is None or self.model is None or self.index is None:
            return []

        scores = np.asarray(self.model.item_factors) @ vector
        order = np.argsort(-scores)

        results: dict[int, float] = {}
        for position in order:
            product_id = int(self.index.product_ids[position])
            if product_id in context.exclude:
                continue
            results[product_id] = float(scores[position])
            if len(results) >= k:
                break
        return top_k(normalise_scores(results), k, source=self.name)

    def score(
        self, context: RecommendationContext, product_ids: list[int]
    ) -> dict[int, float]:
        vector = self._user_vector(context)
        if vector is None or self.model is None or self.index is None:
            return dict.fromkeys(product_ids, 0.0)
        item_factors = np.asarray(self.model.item_factors)
        return {
            pid: (
                float(item_factors[idx] @ vector)
                if (idx := self.index.product_index.get(pid)) is not None
                else 0.0
            )
            for pid in product_ids
        }

    def similar_products(self, product_id: int, k: int = 10) -> list[Scored]:
        """Item-item neighbours in factor space.

        Cosine rather than raw dot product: the dot product is dominated by
        factor magnitude, which encodes popularity, so it would return "popular
        items" rather than "similar items" - a subtle and very common bug.
        """
        if self.model is None or self.index is None or self._item_norms is None:
            return []
        index = self.index.product_index.get(int(product_id))
        if index is None:
            return []

        item_factors = np.asarray(self.model.item_factors)
        similarity = (item_factors @ item_factors[index]) / (
            self._item_norms * self._item_norms[index]
        )
        similarity[index] = -np.inf
        top = np.argpartition(-similarity, min(k, len(similarity) - 1))[:k]
        top = top[np.argsort(-similarity[top])]
        return [
            Scored(int(self.index.product_ids[i]), float(similarity[i]), "collaborative")
            for i in top
        ]


class BPRRecommender(Recommender):
    """Bayesian Personalised Ranking - the measured competitor to ALS.

    Trained so the evaluation report can state *with numbers* why ALS is the
    production choice rather than asserting it. BPR optimises a pairwise
    ranking loss, which is theoretically better matched to a ranking metric,
    but it is sensitive to negative sampling, non-deterministic across runs,
    and slower to converge. The decisive argument (ADR-003) is that the
    ranking objective belongs in stage 2, where the rich features live.
    """

    name = "collaborative_bpr"
    supports_cold_users = False

    def __init__(self, config: CollaborativeConfig | None = None) -> None:
        self.config = config or CollaborativeConfig()
        self.model = None
        self.index: _MatrixIndex | None = None

    def fit(self, weights: pd.DataFrame, *, min_item_interactions: int = 2) -> BPRRecommender:
        from implicit.bpr import BayesianPersonalizedRanking

        matrix, index = build_sparse_matrix(
            weights, min_item_interactions=min_item_interactions
        )
        if matrix.shape[0] == 0:
            return self

        self.index = index
        self.model = BayesianPersonalizedRanking(
            factors=self.config.bpr_factors,
            learning_rate=self.config.bpr_learning_rate,
            regularization=self.config.bpr_regularization,
            iterations=self.config.bpr_iterations,
            random_state=self.config.random_state,
        )
        # BPR treats the matrix as binary preference; the confidence weights
        # that ALS uses are not part of its objective.
        self.model.fit(matrix, show_progress=False)
        return self

    def recommend(self, context: RecommendationContext, k: int = 10) -> list[Scored]:
        if self.model is None or self.index is None or context.user_id is None:
            return []
        row = self.index.user_index.get(int(context.user_id))
        if row is None:
            return []

        scores = np.asarray(self.model.item_factors) @ np.asarray(
            self.model.user_factors[row]
        )
        results: dict[int, float] = {}
        for position in np.argsort(-scores):
            product_id = int(self.index.product_ids[position])
            if product_id in context.exclude:
                continue
            results[product_id] = float(scores[position])
            if len(results) >= k:
                break
        return top_k(normalise_scores(results), k, source=self.name)


__all__ = ["ALSRecommender", "BPRRecommender", "build_sparse_matrix"]
