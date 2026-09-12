"""Implicit-feedback weighting, derived from data (ADR-002).

The question this module answers: how much should a view count relative to a
purchase when building the user-item matrix?

Asserting "purchase = 10, view = 1" is arbitrary and indefensible. Instead the
weight for each event type is derived from how strongly that event predicts a
purchase of that same product:

    w(e)  proportional to  log(1 + P(purchase | e) / P(purchase))

The framing that makes this coherent: in ALS the value is *confidence*, not
preference. The objective is

    sum over (u,i) of  c_ui * (p_ui - x_u . y_i)^2,   c_ui = 1 + alpha * r_ui

where `p_ui` is binary preference and `c_ui` controls how hard the model is
pushed to fit that cell. So the weight answers "how sure are we this user
likes this item?", which is exactly what a conversion-odds ratio measures.

Two properties are enforced:

* **Calibration reads training data only.** Computing conversion odds over the
  full period would leak the test window into the weights, which feed the
  matrix, which trains the model (ADR-007).
* **Recency decays and weights are capped.** A three-month-old view says less
  than yesterday's, and one obsessive user must not dominate an item's factor.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from recsys.config.settings import WeightingConfig

logger = logging.getLogger(__name__)

NEGATIVE_EVENTS: frozenset[str] = frozenset({"REMOVE_FROM_CART"})


@dataclass(slots=True)
class WeightCalibration:
    """Derived weights plus the evidence behind them."""

    weights: dict[str, float]
    conversion_rates: dict[str, float] = field(default_factory=dict)
    lifts: dict[str, float] = field(default_factory=dict)
    support: dict[str, int] = field(default_factory=dict)
    base_rate: float = 0.0
    calibrated: bool = False

    def as_report(self) -> pd.DataFrame:
        """Table for the evaluation report and the MLflow artefact."""
        rows = [
            {
                "event_type": event,
                "support": self.support.get(event, 0),
                "p_purchase_given_event": self.conversion_rates.get(event, float("nan")),
                "lift_over_base": self.lifts.get(event, float("nan")),
                "weight": weight,
            }
            for event, weight in sorted(
                self.weights.items(), key=lambda item: -item[1]
            )
        ]
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, float]:
        return dict(self.weights)


def calibrate_weights(
    train_interactions: pd.DataFrame,
    config: WeightingConfig | None = None,
    *,
    min_support: int = 200,
) -> WeightCalibration:
    """Derive per-event-type weights from observed conversion odds.

    `train_interactions` must be the training fold only.

    For each event type, the probability that the user purchases *that same
    product* within the conversion window. Same product, not any product: the
    latter would mostly measure how active the user is, not how much the event
    says about that item.
    """
    config = config or WeightingConfig()

    if train_interactions.empty:
        return WeightCalibration(weights=dict(config.prior_weights), calibrated=False)

    frame = train_interactions[["user_id", "product_id", "event_type", "occurred_at"]]
    purchases = frame[frame["event_type"] == "PURCHASE"][
        ["user_id", "product_id", "occurred_at"]
    ].rename(columns={"occurred_at": "purchased_at"})

    if purchases.empty:
        logger.warning("no purchases in the training fold; falling back to prior weights")
        return WeightCalibration(weights=dict(config.prior_weights), calibrated=False)

    # Earliest purchase per (user, product): a later repeat purchase should not
    # retroactively make an old view look predictive.
    first_purchase = (
        purchases.groupby(["user_id", "product_id"])["purchased_at"].min().rename("purchased_at")
    )

    window = pd.Timedelta(days=config.conversion_window_days)
    non_purchase = frame[frame["event_type"] != "PURCHASE"].join(
        first_purchase, on=["user_id", "product_id"]
    )

    # The window is symmetric around the event, not forward-only. This is a
    # correctness fix, not a convenience: the quantity being estimated is
    # *confidence that this pair represents a real preference*, not a forecast
    # of a future action. Several event types are inherently post-purchase - a
    # rating or a review can only follow the purchase it describes - so a
    # forward-only window scores them at exactly zero and would tell us a
    # five-star rating carries no information about preference, which is
    # obviously false. Directionality is simply not the relevant property here.
    delta = (non_purchase["purchased_at"] - non_purchase["occurred_at"]).abs()
    non_purchase = non_purchase.assign(converted=delta <= window)

    # Base rate: how often any (user, product) pair the user touched at all is
    # purchased. This is the denominator the lift is measured against.
    touched_pairs = frame.groupby(["user_id", "product_id"]).size()
    purchased_pairs = len(first_purchase)
    base_rate = purchased_pairs / max(len(touched_pairs), 1)

    grouped = non_purchase.groupby("event_type")["converted"]
    rates = grouped.mean().to_dict()
    support = grouped.size().to_dict()

    weights: dict[str, float] = {}
    lifts: dict[str, float] = {}
    conversion_rates: dict[str, float] = {}

    for event_type, rate in rates.items():
        count = int(support.get(event_type, 0))
        if count < min_support:
            # Too little evidence to derive a weight; keep the prior and say so.
            weights[event_type] = config.prior_weights.get(event_type, 1.0)
            logger.info(
                "event %s has support %d < %d; keeping prior weight",
                event_type,
                count,
                min_support,
            )
            continue
        lift = float(rate) / base_rate if base_rate > 0 else 0.0
        conversion_rates[event_type] = float(rate)
        lifts[event_type] = lift
        weights[event_type] = math.log1p(max(lift, 0.0))

    # A purchase is the definition of conversion, so its own conversion rate is
    # 1 by construction and carries no information. Anchoring it to the maximum
    # derived weight plus a margin keeps the ordering meaningful without
    # inventing a number from nothing.
    derived = [w for e, w in weights.items() if e not in NEGATIVE_EVENTS and w > 0]
    purchase_weight = (max(derived) * 1.6) if derived else config.prior_weights["PURCHASE"]
    weights["PURCHASE"] = purchase_weight
    conversion_rates["PURCHASE"] = 1.0
    lifts["PURCHASE"] = 1.0 / base_rate if base_rate > 0 else float("nan")
    support["PURCHASE"] = len(purchases)

    # Removing from the cart is evidence *against* preference. ALS has no place
    # for a negative class - everything unobserved is already weak-negative -
    # so this reduces the pair's confidence rather than forming its own signal.
    for event_type in NEGATIVE_EVENTS:
        if event_type in weights or event_type in rates:
            weights[event_type] = -abs(weights.get(event_type, 0.6)) or -0.6

    # Normalise so a plain product view is 1.0. The absolute scale is absorbed
    # by ALS's `alpha`, so only ratios matter - but the choice of anchor is not
    # arbitrary. Anchoring on the *weakest* signal makes the scale hostage to
    # the noisiest, lowest-support event type: when PRODUCT_SHARE has a near-
    # zero lift it becomes the divisor and inflates every other weight into the
    # hundreds. A view is the highest-volume, most stable event in the log, so
    # anchoring there keeps weights interpretable ("a purchase is worth ~5
    # views") and comparable across runs.
    anchor = weights.get("PRODUCT_VIEW")
    if not anchor or anchor <= 0:
        positives = [w for w in weights.values() if w > 0]
        anchor = min(positives) if positives else 1.0
    weights = {event: weight / anchor for event, weight in weights.items()}

    return WeightCalibration(
        weights=weights,
        conversion_rates=conversion_rates,
        lifts=lifts,
        support={k: int(v) for k, v in support.items()},
        base_rate=float(base_rate),
        calibrated=True,
    )


def build_interaction_matrix_frame(
    interactions: pd.DataFrame,
    calibration: WeightCalibration,
    config: WeightingConfig | None = None,
    *,
    as_of: dt.datetime | None = None,
) -> pd.DataFrame:
    """Aggregate events into one weighted row per (user, product).

    This is the frame that becomes the sparse matrix ALS factorises, and it is
    also what `user_product_interactions` stores in Postgres. Producing it here
    - in the shared package - is what keeps the offline matrix and the online
    affinity table byte-identical (NFR-06).
    """
    config = config or WeightingConfig()
    if interactions.empty:
        return pd.DataFrame(
            columns=["user_id", "product_id", "weight", "last_interaction_at", "n_events"]
        )

    as_of = as_of or interactions["occurred_at"].max().to_pydatetime()
    frame = interactions[["user_id", "product_id", "event_type", "occurred_at"]].copy()

    frame["base_weight"] = frame["event_type"].map(calibration.weights).fillna(1.0)

    # Exponential recency decay. Half-life rather than a hard window because a
    # cliff at N days makes an item's weight jump discontinuously as time
    # passes, which shows up as unexplained churn in recommendations.
    age_days = (
        (pd.Timestamp(as_of) - frame["occurred_at"]).dt.total_seconds() / 86_400.0
    ).clip(lower=0.0)
    frame["recency"] = np.power(0.5, age_days / config.recency_half_life_days)
    frame["weight"] = frame["base_weight"] * frame["recency"]

    aggregated = (
        frame.groupby(["user_id", "product_id"])
        .agg(
            weight=("weight", "sum"),
            last_interaction_at=("occurred_at", "max"),
            first_interaction_at=("occurred_at", "min"),
            n_events=("weight", "size"),
        )
        .reset_index()
    )

    # Cap after aggregation: the cap is about a single pair's total influence,
    # not about any one event.
    aggregated["weight"] = aggregated["weight"].clip(
        lower=0.0, upper=config.max_weight
    )
    return aggregated[aggregated["weight"] > 0].reset_index(drop=True)


def event_type_counts(interactions: pd.DataFrame) -> pd.Series:
    return interactions["event_type"].value_counts()


__all__ = [
    "NEGATIVE_EVENTS",
    "WeightCalibration",
    "build_interaction_matrix_frame",
    "calibrate_weights",
    "event_type_counts",
]
