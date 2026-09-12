"""User feature builder.

**The `as_of` contract.** Every builder takes an `as_of` timestamp and reads
only events strictly before it. This is the mechanical defence against the
leakage described in ADR-007: a feature attached to a training row at time `t`
must not have seen anything at or after `t`. The filter happens once, at the
top of `build`, and `assert_features_respect_cutoff` verifies it - so a future
edit that reaches around the filter fails a test rather than silently inflating
every offline metric.

Nothing here reads `users.segment` or the latent profile columns. Those are
ground truth used by the Phase 2 diagnostic; training on them would be training
on the answer key.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from recsys.preprocessing.splitting import assert_features_respect_cutoff

#: Feature columns produced, in a fixed order. Order is part of the contract:
#: the ranking model indexes features positionally, so a silent reordering
#: between training and serving would feed the model scrambled inputs with no
#: error anywhere (NFR-06).
USER_FEATURE_COLUMNS: tuple[str, ...] = (
    "total_events",
    "total_views",
    "total_clicks",
    "total_carts",
    "total_purchases",
    "distinct_products_viewed",
    "distinct_categories",
    "distinct_brands",
    "session_count",
    "active_days",
    "total_spending",
    "avg_order_value",
    "avg_purchase_price",
    "median_purchase_price",
    "price_percentile_mean",
    "price_sensitivity",
    "view_to_cart_rate",
    "cart_to_purchase_rate",
    "view_to_purchase_rate",
    "purchase_frequency_days",
    "days_since_last_event",
    "days_since_last_purchase",
    "days_since_first_seen",
    "events_per_active_day",
    "category_concentration",
    "brand_concentration",
    "customer_lifetime_value",
)


def _herfindahl(counts: np.ndarray) -> float:
    """Concentration of a user's attention, 0 (spread) to 1 (single category).

    The Herfindahl index rather than entropy because it is bounded, needs no
    normalisation by the number of categories, and is directly interpretable as
    "the probability two random interactions fall in the same category".
    """
    total = counts.sum()
    if total <= 0:
        return 0.0
    shares = counts / total
    return float((shares**2).sum())


def build_user_features(
    interactions: pd.DataFrame,
    *,
    as_of: dt.datetime,
    products: pd.DataFrame,
    orders: pd.DataFrame | None = None,
    sessions: pd.DataFrame | None = None,
    user_ids: pd.Index | None = None,
) -> pd.DataFrame:
    """Compute per-user features from behaviour strictly before `as_of`."""
    history = interactions[interactions["occurred_at"] < pd.Timestamp(as_of)]
    assert_features_respect_cutoff(history, as_of)

    if history.empty:
        index = user_ids if user_ids is not None else pd.Index([], name="user_id")
        return pd.DataFrame(
            0.0, index=index, columns=list(USER_FEATURE_COLUMNS)
        ).rename_axis("user_id")

    price_lookup = products.set_index("id")["price"]
    percentile_lookup = (
        products.set_index("id")["price_percentile"]
        if "price_percentile" in products.columns
        else products.set_index("id")["price"].rank(pct=True)
    )

    frame = history.assign(
        price=history["product_id"].map(price_lookup),
        price_percentile=history["product_id"].map(percentile_lookup),
        day=history["occurred_at"].dt.floor("D"),
    )

    grouped = frame.groupby("user_id")
    features = pd.DataFrame(index=grouped.size().index)
    features.index.name = "user_id"

    # --- volume ---------------------------------------------------------
    features["total_events"] = grouped.size()
    for label, event_type in (
        ("total_views", "PRODUCT_VIEW"),
        ("total_clicks", "PRODUCT_CLICK"),
        ("total_carts", "ADD_TO_CART"),
        ("total_purchases", "PURCHASE"),
    ):
        features[label] = (
            frame[frame["event_type"] == event_type].groupby("user_id").size()
        )
    features = features.fillna(0.0)

    views = frame[frame["event_type"] == "PRODUCT_VIEW"]
    features["distinct_products_viewed"] = views.groupby("user_id")["product_id"].nunique()
    features["distinct_categories"] = frame.groupby("user_id")["category_id"].nunique()
    features["distinct_brands"] = frame.groupby("user_id")["brand_id"].nunique()
    features["session_count"] = frame.groupby("user_id")["session_key"].nunique()
    features["active_days"] = frame.groupby("user_id")["day"].nunique()

    # --- monetary -------------------------------------------------------
    purchases = frame[frame["event_type"] == "PURCHASE"]
    if not purchases.empty:
        by_user = purchases.groupby("user_id")
        features["total_spending"] = by_user["price"].sum()
        features["avg_purchase_price"] = by_user["price"].mean()
        features["median_purchase_price"] = by_user["price"].median()
        features["price_percentile_mean"] = by_user["price_percentile"].mean()
    else:
        for column in (
            "total_spending",
            "avg_purchase_price",
            "median_purchase_price",
            "price_percentile_mean",
        ):
            features[column] = 0.0

    if orders is not None and not orders.empty:
        past_orders = orders[orders["placed_at"] < pd.Timestamp(as_of)]
        if not past_orders.empty:
            by_user = past_orders.groupby("user_id")["grand_total"]
            features["avg_order_value"] = by_user.mean()
            features["order_count"] = by_user.size()
        else:
            features["avg_order_value"] = 0.0
            features["order_count"] = 0.0
    else:
        features["avg_order_value"] = features.get("total_spending", 0.0)
        features["order_count"] = features["total_purchases"]

    features = features.fillna(0.0)

    # --- price sensitivity ----------------------------------------------
    # Defined as how far below the middle of a category's price range the user
    # actually buys. A user whose purchases sit at the 20th percentile is
    # price-sensitive; one at the 80th is not. Using the *realised* percentile
    # rather than absolute price is what makes this comparable across a user
    # who shops for books and one who shops for laptops.
    features["price_sensitivity"] = (1.0 - features["price_percentile_mean"]).clip(0.0, 1.0)
    features.loc[features["total_purchases"] == 0, "price_sensitivity"] = 0.5

    # --- funnel rates ---------------------------------------------------
    safe_views = features["total_views"].replace(0, np.nan)
    safe_carts = features["total_carts"].replace(0, np.nan)
    features["view_to_cart_rate"] = (features["total_carts"] / safe_views).fillna(0.0)
    features["cart_to_purchase_rate"] = (
        features["total_purchases"] / safe_carts
    ).fillna(0.0)
    features["view_to_purchase_rate"] = (
        features["total_purchases"] / safe_views
    ).fillna(0.0)

    # --- recency --------------------------------------------------------
    cutoff = pd.Timestamp(as_of)
    last_event = grouped["occurred_at"].max()
    first_event = grouped["occurred_at"].min()
    features["days_since_last_event"] = (cutoff - last_event).dt.total_seconds() / 86_400.0
    features["days_since_first_seen"] = (cutoff - first_event).dt.total_seconds() / 86_400.0

    if not purchases.empty:
        last_purchase = purchases.groupby("user_id")["occurred_at"].max()
        features["days_since_last_purchase"] = (
            (cutoff - last_purchase).dt.total_seconds() / 86_400.0
        )
        # Mean gap between purchases. NaN with fewer than two purchases is
        # correct and is filled with the observed lifetime instead of 0, which
        # would falsely imply "buys constantly".
        span = purchases.groupby("user_id")["occurred_at"].agg(["min", "max", "size"])
        gap = (span["max"] - span["min"]).dt.total_seconds() / 86_400.0
        features["purchase_frequency_days"] = (gap / (span["size"] - 1).clip(lower=1)).where(
            span["size"] > 1
        )
    else:
        features["days_since_last_purchase"] = np.nan
        features["purchase_frequency_days"] = np.nan

    features["days_since_last_purchase"] = features["days_since_last_purchase"].fillna(
        features["days_since_first_seen"]
    )
    features["purchase_frequency_days"] = features["purchase_frequency_days"].fillna(
        features["days_since_first_seen"]
    )

    features["events_per_active_day"] = (
        features["total_events"] / features["active_days"].clip(lower=1)
    )

    # --- concentration ---------------------------------------------------
    category_counts = frame.groupby(["user_id", "category_id"]).size()
    features["category_concentration"] = category_counts.groupby("user_id").apply(
        lambda s: _herfindahl(s.to_numpy())
    )
    brand_counts = frame.groupby(["user_id", "brand_id"]).size()
    features["brand_concentration"] = brand_counts.groupby("user_id").apply(
        lambda s: _herfindahl(s.to_numpy())
    )

    # --- lifetime value ---------------------------------------------------
    # Historical spend plus a crude forward projection: observed purchase rate
    # extended over a 12-month horizon, discounted by recency. Deliberately
    # simple - a survival model would be better but would need churn labels the
    # dataset does not carry, and an unexplainable CLV is worse than a rough one.
    tenure_days = features["days_since_first_seen"].clip(lower=1.0)
    daily_value = features["total_spending"] / tenure_days
    recency_discount = np.exp(-features["days_since_last_purchase"] / 90.0)
    features["customer_lifetime_value"] = (
        features["total_spending"] + daily_value * 365.0 * recency_discount
    )

    features = features.fillna(0.0)

    if user_ids is not None:
        features = features.reindex(user_ids).fillna(0.0)
        features.loc[features["total_events"] == 0, "price_sensitivity"] = 0.5
        features.index.name = "user_id"

    return features[list(USER_FEATURE_COLUMNS)]


def build_category_affinity(
    interactions: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    as_of: dt.datetime,
    level: str = "category_id",
    top_n: int = 10,
) -> dict[int, dict[int, float]]:
    """Normalised affinity map per user: {user_id: {category_id: share}}.

    Built from the *weighted* interaction frame rather than raw counts, so a
    purchase moves affinity far more than a view - which is the whole point of
    calibrating the weights in the first place (ADR-002).
    """
    history = interactions[interactions["occurred_at"] < pd.Timestamp(as_of)]
    if history.empty or weights.empty:
        return {}

    joined = weights.merge(
        history[["user_id", "product_id", level]].drop_duplicates(
            subset=["user_id", "product_id"]
        ),
        on=["user_id", "product_id"],
        how="inner",
    )
    if joined.empty:
        return {}

    totals = joined.groupby(["user_id", level])["weight"].sum()
    affinity: dict[int, dict[int, float]] = {}
    for user_id, group in totals.groupby(level=0):
        series = group.droplevel(0).nlargest(top_n)
        total = float(series.sum())
        if total <= 0:
            continue
        affinity[int(user_id)] = {
            int(key): float(value / total) for key, value in series.items()
        }
    return affinity


__all__ = ["USER_FEATURE_COLUMNS", "build_category_affinity", "build_user_features"]
