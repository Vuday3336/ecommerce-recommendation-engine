"""Training-data construction for the stage-2 ranker.

**Where the labels come from, and why it is not obvious.**

The honest source of ranking labels is logged impressions: the system showed
these ten items, the user clicked the third. Before the system has served any
traffic there are no such logs, so the bootstrap ranker is trained on
*retrieved candidates labelled by what the user actually did next*:

    fit component models on TRAIN
        -> generate candidates for each user
            -> label each candidate from the VALIDATION window
                purchase = 3, add-to-cart = 2, click = 1, otherwise 0
                    -> train LambdaRank, evaluate on TEST

This is a genuine forward-looking task with no leakage: features come strictly
from train, labels strictly from the window after it, and the test window is
never touched.

**The bias it carries, stated plainly.** These labels describe what the user
did while browsing on their own, not what they would have done had we shown
them this list. Items the user never encountered are labelled 0 regardless of
whether they would have liked them, so absence of a label conflates
"not interested" with "never saw it". That is the standard missing-not-at-random
problem, and it is why the exploration budget in candidate generation exists: it
produces the unbiased impressions that a later inverse-propensity-weighted
retrain would need. `docs/evaluation.md` records this as a known limitation
rather than pretending the bootstrap labels are unbiased.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from recsys.candidates.generator import CandidateGenerator, CandidateSet
from recsys.config.settings import RankerConfig
from recsys.features.interaction import PairFeatureAssembler
from recsys.features.product import PRODUCT_FEATURE_COLUMNS
from recsys.features.user import USER_FEATURE_COLUMNS
from recsys.models.base import RecommendationContext

logger = logging.getLogger(__name__)

#: Per-source columns appended to every candidate row. The ranker learning that
#: "retrieved by both collaborative filtering and co-visitation" is a strong
#: signal is the main reason provenance is carried through the merge.
SOURCE_COLUMNS: tuple[str, ...] = (
    "src_collaborative_als_score",
    "src_collaborative_als_rank",
    "src_content_score",
    "src_content_rank",
    "src_covisitation_score",
    "src_covisitation_rank",
    "src_category_affinity_score",
    "src_category_affinity_rank",
    "src_brand_affinity_score",
    "src_brand_affinity_rank",
    "src_trending_score",
    "src_popularity_score",
    "src_frequently_bought_together_score",
    "src_recently_viewed_score",
    "source_agreement",
    "candidate_rank",
)

MISSING_RANK = 999.0


@dataclass(slots=True)
class RankingDataset:
    """Feature matrix, labels and query groups for LambdaRank."""

    features: pd.DataFrame
    labels: np.ndarray
    groups: np.ndarray
    user_ids: np.ndarray
    product_ids: np.ndarray

    def __len__(self) -> int:
        return len(self.features)

    @property
    def positive_rate(self) -> float:
        return float((self.labels > 0).mean()) if len(self.labels) else 0.0

    def summary(self) -> dict[str, float]:
        return {
            "rows": len(self.features),
            "queries": len(self.groups),
            "features": self.features.shape[1],
            "positive_rate": round(self.positive_rate, 5),
            "mean_group_size": round(float(self.groups.mean()), 1) if len(self.groups) else 0.0,
        }


def build_labels(
    events: pd.DataFrame, config: RankerConfig | None = None
) -> dict[int, dict[int, int]]:
    """Graded relevance per user from a future window.

    Grades are 3/2/1/0 rather than binary because LambdaRank uses gains of
    `2^label - 1`. That spacing (7 / 3 / 1 / 0) tells the model a purchase is
    more than twice as valuable as a cart addition, which matches the business
    reality and is the whole reason for using graded relevance at all.
    """
    config = config or RankerConfig()
    mapping = {
        "PURCHASE": config.label_purchase,
        "ADD_TO_CART": config.label_cart,
        "PRODUCT_CLICK": config.label_click,
    }
    frame = events[events["event_type"].isin(mapping)]
    if frame.empty:
        return {}

    labelled = frame.assign(label=frame["event_type"].map(mapping))
    best = labelled.groupby(["user_id", "product_id"])["label"].max()

    out: dict[int, dict[int, int]] = {}
    for (user_id, product_id), label in best.items():
        out.setdefault(int(user_id), {})[int(product_id)] = int(label)
    return out


def _source_row(
    candidates: CandidateSet, product_id: int, agreement: int, position: int
) -> dict[str, float]:
    """Per-source scores, ranks, and the fused retrieval rank.

    `candidate_rank` is the position in the *fused candidate pool*, not a
    display position. The distinction matters: the fused rank is computed
    identically at training and at serving time, so it is a legitimate feature
    and is used unchanged at inference. Display position - where an item
    actually appeared on screen - is the bias-prone one, and it only exists
    once the system has served traffic. When logged impressions replace these
    bootstrap labels, display position becomes a feature that *must* be held
    constant at inference; `candidate_rank` never does.
    """
    def score(source: str) -> float:
        return candidates.score_from(source, product_id, 0.0)

    def rank(source: str) -> float:
        return float(candidates.ranks_by_source.get(source, {}).get(product_id, MISSING_RANK))

    return {
        "src_collaborative_als_score": score("collaborative_als"),
        "src_collaborative_als_rank": rank("collaborative_als"),
        "src_content_score": score("content"),
        "src_content_rank": rank("content"),
        "src_covisitation_score": score("covisitation"),
        "src_covisitation_rank": rank("covisitation"),
        "src_category_affinity_score": score("category_affinity"),
        "src_category_affinity_rank": rank("category_affinity"),
        "src_brand_affinity_score": score("brand_affinity"),
        "src_brand_affinity_rank": rank("brand_affinity"),
        "src_trending_score": score("trending"),
        "src_popularity_score": score("popularity"),
        "src_frequently_bought_together_score": score("frequently_bought_together"),
        "src_recently_viewed_score": score("recently_viewed"),
        "source_agreement": float(agreement),
        "candidate_rank": float(position),
    }


class RankingDatasetBuilder:
    """Assembles the (features, label, group) triples LightGBM needs."""

    def __init__(
        self,
        *,
        generator: CandidateGenerator,
        user_features: pd.DataFrame,
        product_features: pd.DataFrame,
        pair_assembler: PairFeatureAssembler,
        config: RankerConfig | None = None,
    ) -> None:
        self.generator = generator
        self.user_features = user_features
        self.product_features = product_features
        self.pair_assembler = pair_assembler
        self.config = config or RankerConfig()

        self.feature_columns: list[str] = (
            [f"user_{c}" for c in USER_FEATURE_COLUMNS]
            + [f"product_{c}" for c in PRODUCT_FEATURE_COLUMNS]
            + list(self.pair_assembler.assemble(0, [], user_price_percentile=0.5).columns)
            + list(SOURCE_COLUMNS)
        )

    def build_for_user(
        self,
        context: RecommendationContext,
        candidates: CandidateSet,
        *,
        content_similarity: dict[int, float] | None = None,
        collaborative_score: dict[int, float] | None = None,
    ) -> pd.DataFrame:
        """Feature matrix for one user's candidate set.

        Vectorised deliberately: user features are broadcast once, product
        features are a single reindex, and pair features are one call. Building
        300 rows in a Python loop over a DataFrame would cost tens of
        milliseconds per request and put NFR-01 out of reach (R-09).
        """
        product_ids = candidates.product_ids
        if not product_ids:
            return pd.DataFrame(columns=self.feature_columns)

        user_id = context.user_id or -1
        if user_id in self.user_features.index:
            user_row = self.user_features.loc[user_id]
        else:
            user_row = pd.Series(0.0, index=self.user_features.columns)

        user_block = pd.DataFrame(
            np.repeat(user_row.to_numpy(dtype=float)[None, :], len(product_ids), axis=0),
            columns=[f"user_{c}" for c in self.user_features.columns],
            index=product_ids,
        )

        product_block = (
            self.product_features.reindex(product_ids)
            .fillna(0.0)
            .rename(columns=lambda c: f"product_{c}")
        )
        product_block.index = pd.Index(product_ids, name="product_id")

        pair_block = self.pair_assembler.assemble(
            user_id,
            product_ids,
            user_price_percentile=context.price_percentile,
            content_similarity=content_similarity,
            collaborative_score=collaborative_score,
        )

        agreement = candidates.source_agreement()
        source_block = pd.DataFrame(
            [
                _source_row(candidates, pid, agreement.get(pid, 0), position)
                for position, pid in enumerate(product_ids)
            ],
            index=pd.Index(product_ids, name="product_id"),
        )

        frame = pd.concat([user_block, product_block, pair_block, source_block], axis=1)
        return frame.reindex(columns=self.feature_columns).fillna(0.0)

    def build(
        self,
        contexts: dict[int, RecommendationContext],
        labels: dict[int, dict[int, int]],
        *,
        content_scores: dict[int, dict[int, float]] | None = None,
        collaborative_scores: dict[int, dict[int, float]] | None = None,
        require_positive: bool = True,
        negative_samples: int | None = 120,
        hard_negative_bias: bool = False,
        seed: int = 42,
    ) -> RankingDataset:
        """Build the full dataset over many users.

        `negative_samples` downsamples negatives per query, keeping every
        positive. This is a training-time change only - inference always ranks
        the full pool.

        It matters more than it sounds. With 293 candidates and typically two
        positives, the positive rate is about 0.7% and LambdaRank's early
        stopping halted after **9 of 400 trees**: nearly every pairwise
        comparison it sampled was negative-versus-negative and carried no
        gradient.

        **`hard_negative_bias` defaults to False, and that was measured, not
        assumed.** Sampling negatives preferentially from the top of the
        retrieved pool is the textbook "hard negatives" trick, and here it was
        actively harmful - test NDCG@10 collapsed from 0.050 to 0.022. The
        reason is a train/serve distribution mismatch rather than anything
        wrong with hard negatives in principle: `candidate_rank` is the
        strongest single feature, biased sampling gives the model almost no
        training examples above rank ~60, and at inference every deep-pool item
        then falls into one under-determined leaf and gets surfaced
        arbitrarily. Catalogue coverage jumping from 0.27 to 0.52 while
        accuracy halved is that failure made visible. Uniform sampling
        preserves the rank distribution and works:

            full pool (no sampling)      ndcg@10 0.05023   9 trees
            uniform, 60 negatives        ndcg@10 0.05106  17 trees
            uniform, 120 negatives       ndcg@10 0.05301  34 trees   <- default
            top-biased, 60 negatives     ndcg@10 0.02153   9 trees

        The remaining trade is that predicted scores are no longer calibrated
        probabilities, since the negative class is under-represented by
        construction. For a ranking model that is irrelevant - only the
        ordering is used. It would matter if these scores were displayed or
        thresholded, and they are not.
        """
        rng = np.random.default_rng(seed)
        blocks: list[pd.DataFrame] = []
        all_labels: list[np.ndarray] = []
        groups: list[int] = []
        user_ids: list[np.ndarray] = []
        product_ids: list[np.ndarray] = []

        for user_id, context in contexts.items():
            user_labels = labels.get(user_id, {})
            if require_positive and not user_labels:
                # A query with no positive at any rank contributes nothing to a
                # pairwise/listwise loss - there is no swap that changes NDCG -
                # so it is pure cost. Dropping these is standard and is why the
                # positive rate below looks high for a recommendation task.
                continue

            candidates = self.generator.generate(context)
            if not candidates.product_ids:
                continue

            frame = self.build_for_user(
                context,
                candidates,
                content_similarity=(content_scores or {}).get(user_id),
                collaborative_score=(collaborative_scores or {}).get(user_id),
            )
            if frame.empty:
                continue

            row_labels = np.array(
                [user_labels.get(pid, 0) for pid in candidates.product_ids], dtype=np.int32
            )
            if require_positive and row_labels.max() == 0:
                continue

            selected = np.arange(len(row_labels))
            if negative_samples is not None:
                positive_idx = np.flatnonzero(row_labels > 0)
                negative_idx = np.flatnonzero(row_labels == 0)
                if len(negative_idx) > negative_samples:
                    if hard_negative_bias:
                        weights = 1.0 / (1.0 + negative_idx.astype(float))
                        weights /= weights.sum()
                    else:
                        weights = None
                    negative_idx = rng.choice(
                        negative_idx, size=negative_samples, replace=False, p=weights
                    )
                selected = np.sort(np.concatenate([positive_idx, negative_idx]))

            frame = frame.iloc[selected]
            row_labels = row_labels[selected]
            sampled_products = np.asarray(candidates.product_ids, dtype=np.int64)[selected]

            blocks.append(frame)
            all_labels.append(row_labels)
            groups.append(len(frame))
            user_ids.append(np.full(len(frame), user_id, dtype=np.int64))
            product_ids.append(sampled_products)

        if not blocks:
            return RankingDataset(
                features=pd.DataFrame(columns=self.feature_columns),
                labels=np.array([], dtype=np.int32),
                groups=np.array([], dtype=np.int32),
                user_ids=np.array([], dtype=np.int64),
                product_ids=np.array([], dtype=np.int64),
            )

        return RankingDataset(
            features=pd.concat(blocks, ignore_index=True),
            labels=np.concatenate(all_labels),
            groups=np.asarray(groups, dtype=np.int32),
            user_ids=np.concatenate(user_ids),
            product_ids=np.concatenate(product_ids),
        )


__all__ = [
    "MISSING_RANK",
    "SOURCE_COLUMNS",
    "RankingDataset",
    "RankingDatasetBuilder",
    "build_labels",
]
