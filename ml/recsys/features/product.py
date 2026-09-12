"""Product feature builder.

Same `as_of` contract as the user builder. Popularity features are the most
dangerous kind of leakage in a recommender: computing "how popular is this
product?" over the full period encodes the test window into every training row,
and the resulting model looks excellent offline and mediocre online. Every
count here is bounded by `as_of`.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from recsys.config.settings import PopularityConfig
from recsys.preprocessing.splitting import assert_features_respect_cutoff

PRODUCT_FEATURE_COLUMNS: tuple[str, ...] = (
    "price",
    "log_price",
    "price_percentile",
    "category_id",
    "brand_id",
    "rating_average",
    "rating_count",
    "quality_score",
    "view_count_total",
    "view_count_30d",
    "view_count_7d",
    "cart_count_30d",
    "purchase_count_total",
    "purchase_count_30d",
    "distinct_buyers",
    "revenue_30d",
    "view_to_cart_rate",
    "conversion_rate",
    "repeat_purchase_rate",
    "popularity_score",
    "trending_score",
    "novelty_score",
    "days_since_release",
    "stock_quantity",
    "is_cold",
)

#: Below this many interactions a product has no usable collaborative signal
#: and must be served through the content path instead (FR-10).
COLD_ITEM_THRESHOLD = 5


def build_product_features(
    interactions: pd.DataFrame,
    products: pd.DataFrame,
    *,
    as_of: dt.datetime,
    config: PopularityConfig | None = None,
) -> pd.DataFrame:
    """Compute per-product features from behaviour strictly before `as_of`."""
    config = config or PopularityConfig()
    cutoff = pd.Timestamp(as_of)

    history = interactions[interactions["occurred_at"] < cutoff]
    assert_features_respect_cutoff(history, as_of)

    features = products.set_index("id")[
        ["category_id", "brand_id", "price", "stock_quantity", "rating_average", "rating_count"]
    ].copy()
    features.index.name = "product_id"

    features["log_price"] = np.log1p(features["price"])
    features["price_percentile"] = (
        products.set_index("id")["price_percentile"]
        if "price_percentile" in products.columns
        else features.groupby("category_id")["price"].rank(pct=True)
    )

    released = pd.to_datetime(products.set_index("id")["released_at"], utc=True)
    features["days_since_release"] = (
        (cutoff - released).dt.total_seconds() / 86_400.0
    ).clip(lower=0.0)

    window_30d = cutoff - pd.Timedelta(days=30)
    window_7d = cutoff - pd.Timedelta(days=7)

    def counts(event_type: str, since: pd.Timestamp | None = None) -> pd.Series:
        subset = history[history["event_type"] == event_type]
        if since is not None:
            subset = subset[subset["occurred_at"] >= since]
        return subset.groupby("product_id").size()

    features["view_count_total"] = counts("PRODUCT_VIEW")
    features["view_count_30d"] = counts("PRODUCT_VIEW", window_30d)
    features["view_count_7d"] = counts("PRODUCT_VIEW", window_7d)
    features["cart_count_30d"] = counts("ADD_TO_CART", window_30d)
    features["purchase_count_total"] = counts("PURCHASE")
    features["purchase_count_30d"] = counts("PURCHASE", window_30d)

    purchases = history[history["event_type"] == "PURCHASE"]
    if not purchases.empty:
        features["distinct_buyers"] = purchases.groupby("product_id")["user_id"].nunique()
        recent_purchases = purchases[purchases["occurred_at"] >= window_30d]
        features["revenue_30d"] = (
            recent_purchases.assign(
                price=recent_purchases["product_id"].map(features["price"])
            )
            .groupby("product_id")["price"]
            .sum()
        )
        per_pair = purchases.groupby(["product_id", "user_id"]).size()
        repeat = (per_pair > 1).groupby(level=0).mean()
        features["repeat_purchase_rate"] = repeat
    else:
        features["distinct_buyers"] = 0.0
        features["revenue_30d"] = 0.0
        features["repeat_purchase_rate"] = 0.0

    features = features.fillna(0.0)

    # --- rates, smoothed --------------------------------------------------
    # Bayesian shrinkage toward the catalogue mean. Without it, a product with
    # two views and one purchase shows a 50% conversion rate and tops every
    # ranking that uses the feature - the classic small-sample trap.
    prior = config.smoothing_prior
    global_conversion = (
        features["purchase_count_total"].sum() / max(features["view_count_total"].sum(), 1)
    )
    global_cart_rate = (
        features["cart_count_30d"].sum() / max(features["view_count_30d"].sum(), 1)
    )

    features["conversion_rate"] = (
        features["purchase_count_total"] + prior * global_conversion
    ) / (features["view_count_total"] + prior)
    features["view_to_cart_rate"] = (
        features["cart_count_30d"] + prior * global_cart_rate
    ) / (features["view_count_30d"] + prior)

    # Bayesian-smoothed rating for the same reason: one five-star review must
    # not outrank two hundred four-star ones.
    catalogue_mean_rating = float(features["rating_average"].mean())
    features["quality_score"] = (
        features["rating_average"] * features["rating_count"]
        + catalogue_mean_rating * prior
    ) / (features["rating_count"] + prior)

    # --- composite scores -------------------------------------------------
    # Popularity blends purchases, views and revenue on a log scale, because
    # raw counts are power-law distributed and a linear blend would be entirely
    # decided by the head of the distribution.
    popularity = (
        0.55 * np.log1p(features["purchase_count_total"])
        + 0.25 * np.log1p(features["view_count_total"])
        + 0.20 * np.log1p(features["revenue_30d"])
    )
    features["popularity_score"] = _normalise(popularity)

    features["trending_score"] = _trending_score(history, features, cutoff, config)

    # Novelty is the inverse of log popularity, used by the diversity re-ranker
    # to reward showing something the user is unlikely to have seen.
    interaction_totals = features["view_count_total"] + features["purchase_count_total"]
    features["novelty_score"] = _normalise(-np.log1p(interaction_totals))

    features["is_cold"] = (interaction_totals < COLD_ITEM_THRESHOLD).astype(float)

    return features[list(PRODUCT_FEATURE_COLUMNS)].fillna(0.0)


def _normalise(series: pd.Series) -> pd.Series:
    """Min-max to [0, 1]. Constant input maps to 0, not NaN."""
    low, high = float(series.min()), float(series.max())
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return pd.Series(0.0, index=series.index)
    return (series - low) / (high - low)


def _trending_score(
    history: pd.DataFrame,
    features: pd.DataFrame,
    cutoff: pd.Timestamp,
    config: PopularityConfig,
) -> pd.Series:
    """Time-decayed interaction velocity.

    Mirrors the online formulation in `app/cache/counters.py`: the same event
    weights and the same half-life, so the batch-computed trending list and the
    live Redis one agree rather than being two different notions of "trending"
    that quietly disagree on the homepage.
    """
    window_start = cutoff - pd.Timedelta(hours=config.trending_window_hours * 7)
    recent = history[history["occurred_at"] >= window_start]
    if recent.empty:
        return pd.Series(0.0, index=features.index)

    event_weights = {
        "PRODUCT_VIEW": 1.0,
        "PRODUCT_CLICK": 1.5,
        "WISHLIST": 3.0,
        "ADD_TO_CART": 5.0,
        "PURCHASE": 12.0,
    }
    weighted = recent.assign(
        w=recent["event_type"].map(event_weights).fillna(0.0),
        age_hours=(cutoff - recent["occurred_at"]).dt.total_seconds() / 3600.0,
    )
    weighted["decayed"] = weighted["w"] * np.power(
        0.5, weighted["age_hours"] / (config.trending_half_life_hours * 7)
    )
    scores = weighted.groupby("product_id")["decayed"].sum()
    return _normalise(scores.reindex(features.index).fillna(0.0))


__all__ = [
    "COLD_ITEM_THRESHOLD",
    "PRODUCT_FEATURE_COLUMNS",
    "build_product_features",
]
