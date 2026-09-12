"""Interaction (user x product pair) features for the ranking stage.

These are the features that only exist for a *pair*: how often this user has
touched this product, how well its price fits their band, whether it matches a
category or brand they favour. They are what let the ranker separate two
products the collaborative model scores identically.

Assembly is vectorised over the whole candidate set rather than computed per
candidate. With 300 candidates per request and a p99 budget of 150 ms, a Python
loop doing 300 dictionary lookups and arithmetic operations is the difference
between comfortably inside the budget and comfortably outside it (R-09).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

INTERACTION_FEATURE_COLUMNS: tuple[str, ...] = (
    "pair_view_count",
    "pair_cart_count",
    "pair_purchase_count",
    "pair_weight",
    "days_since_pair_interaction",
    "has_interacted",
    "has_purchased",
    "category_affinity",
    "brand_affinity",
    "category_match",
    "brand_match",
    "price_distance",
    "price_band_match",
    "content_similarity",
    "collaborative_score",
)


def build_pair_history(
    interactions: pd.DataFrame,
    *,
    as_of: dt.datetime,
) -> pd.DataFrame:
    """Per-(user, product) counts and recency, bounded by `as_of`."""
    cutoff = pd.Timestamp(as_of)
    history = interactions[interactions["occurred_at"] < cutoff]
    if history.empty:
        return pd.DataFrame(
            columns=[
                "user_id",
                "product_id",
                "pair_view_count",
                "pair_cart_count",
                "pair_purchase_count",
                "days_since_pair_interaction",
            ]
        )

    grouped = history.groupby(["user_id", "product_id"])
    frame = pd.DataFrame(
        {
            "pair_view_count": grouped.apply(
                lambda g: int((g["event_type"] == "PRODUCT_VIEW").sum()),
                include_groups=False,
            ),
            "last_at": grouped["occurred_at"].max(),
        }
    )
    carts = history[history["event_type"] == "ADD_TO_CART"].groupby(
        ["user_id", "product_id"]
    ).size()
    purchases = history[history["event_type"] == "PURCHASE"].groupby(
        ["user_id", "product_id"]
    ).size()

    frame["pair_cart_count"] = carts
    frame["pair_purchase_count"] = purchases
    frame = frame.fillna({"pair_cart_count": 0, "pair_purchase_count": 0})
    frame["days_since_pair_interaction"] = (
        cutoff - frame["last_at"]
    ).dt.total_seconds() / 86_400.0

    return frame.drop(columns=["last_at"]).reset_index()


class PairFeatureAssembler:
    """Builds the ranking feature matrix for one user's candidate set.

    Constructed once per process from precomputed lookups, then called per
    request. Everything expensive - index building, array extraction - happens
    in `__init__`, so the per-request path is array indexing and arithmetic.
    """

    def __init__(
        self,
        *,
        product_features: pd.DataFrame,
        pair_history: pd.DataFrame | None = None,
        interaction_weights: pd.DataFrame | None = None,
        category_affinity: dict[int, dict[int, float]] | None = None,
        brand_affinity: dict[int, dict[int, float]] | None = None,
    ) -> None:
        self.product_features = product_features
        self.category_of = product_features["category_id"].to_dict()
        self.brand_of = product_features["brand_id"].to_dict()
        self.price_percentile_of = product_features["price_percentile"].to_dict()

        self.category_affinity = category_affinity or {}
        self.brand_affinity = brand_affinity or {}

        self._pair_index: dict[tuple[int, int], tuple[float, float, float, float]] = {}
        if pair_history is not None and not pair_history.empty:
            for row in pair_history.itertuples(index=False):
                self._pair_index[(int(row.user_id), int(row.product_id))] = (
                    float(row.pair_view_count),
                    float(row.pair_cart_count),
                    float(row.pair_purchase_count),
                    float(row.days_since_pair_interaction),
                )

        self._weight_index: dict[tuple[int, int], float] = {}
        if interaction_weights is not None and not interaction_weights.empty:
            for row in interaction_weights.itertuples(index=False):
                self._weight_index[(int(row.user_id), int(row.product_id))] = float(
                    row.weight
                )

    def assemble(
        self,
        user_id: int,
        product_ids: list[int],
        *,
        user_price_percentile: float,
        content_similarity: dict[int, float] | None = None,
        collaborative_score: dict[int, float] | None = None,
    ) -> pd.DataFrame:
        """Feature matrix with one row per candidate, columns in fixed order."""
        content_similarity = content_similarity or {}
        collaborative_score = collaborative_score or {}
        user_categories = self.category_affinity.get(user_id, {})
        user_brands = self.brand_affinity.get(user_id, {})

        n = len(product_ids)
        views = np.zeros(n)
        carts = np.zeros(n)
        purchases = np.zeros(n)
        recency = np.full(n, 999.0)
        weights = np.zeros(n)
        cat_affinity = np.zeros(n)
        brand_affinity_values = np.zeros(n)
        cat_match = np.zeros(n)
        brand_match = np.zeros(n)
        price_distance = np.zeros(n)
        band_match = np.zeros(n)
        content = np.zeros(n)
        collaborative = np.zeros(n)

        for i, product_id in enumerate(product_ids):
            key = (user_id, product_id)
            pair = self._pair_index.get(key)
            if pair is not None:
                views[i], carts[i], purchases[i], recency[i] = pair
            weights[i] = self._weight_index.get(key, 0.0)

            category = self.category_of.get(product_id)
            brand = self.brand_of.get(product_id)
            if category is not None:
                affinity = user_categories.get(int(category), 0.0)
                cat_affinity[i] = affinity
                cat_match[i] = 1.0 if affinity > 0 else 0.0
            if brand is not None:
                affinity = user_brands.get(int(brand), 0.0)
                brand_affinity_values[i] = affinity
                brand_match[i] = 1.0 if affinity > 0 else 0.0

            percentile = self.price_percentile_of.get(product_id, 0.5)
            price_distance[i] = abs(float(percentile) - user_price_percentile)
            band_match[i] = 1.0 if price_distance[i] < 0.2 else 0.0

            content[i] = content_similarity.get(product_id, 0.0)
            collaborative[i] = collaborative_score.get(product_id, 0.0)

        return pd.DataFrame(
            {
                "pair_view_count": views,
                "pair_cart_count": carts,
                "pair_purchase_count": purchases,
                "pair_weight": weights,
                "days_since_pair_interaction": recency,
                "has_interacted": (views + carts + purchases > 0).astype(float),
                "has_purchased": (purchases > 0).astype(float),
                "category_affinity": cat_affinity,
                "brand_affinity": brand_affinity_values,
                "category_match": cat_match,
                "brand_match": brand_match,
                "price_distance": price_distance,
                "price_band_match": band_match,
                "content_similarity": content,
                "collaborative_score": collaborative,
            },
            index=pd.Index(product_ids, name="product_id"),
        )[list(INTERACTION_FEATURE_COLUMNS)]


__all__ = [
    "INTERACTION_FEATURE_COLUMNS",
    "PairFeatureAssembler",
    "build_pair_history",
]
