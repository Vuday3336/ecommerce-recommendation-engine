"""Temporal splitting and the leakage guard (ADR-007).

A random split of interactions lets a model see a user's future while
predicting their past. A recommender that has already seen you buy the shoes
trivially "predicts" that you will view them, and every metric inflates. It
also leaks globally: item popularity computed over the full period encodes the
test window.

So the split is by time, and `assert_no_leakage` is run on every split rather
than trusted. The guard is cheap and the failure it catches is invisible - a
leaking pipeline produces beautiful numbers and no error.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

import pandas as pd

from recsys.config.settings import SplitConfig

logger = logging.getLogger(__name__)


class LeakageError(AssertionError):
    """Raised when a split would let the future inform the past."""


@dataclass(slots=True)
class TemporalSplit:
    """Train / validation / test frames and their boundaries."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_end: dt.datetime
    validation_end: dt.datetime

    @property
    def as_of(self) -> dt.datetime:
        """The cut-off features must respect when training the final model.

        Every feature used to predict a validation or test interaction must be
        computed from data strictly before this timestamp.
        """
        return self.train_end

    def summary(self) -> dict[str, object]:
        return {
            "train_rows": len(self.train),
            "validation_rows": len(self.validation),
            "test_rows": len(self.test),
            "train_end": self.train_end.isoformat(),
            "validation_end": self.validation_end.isoformat(),
            "train_users": int(self.train["user_id"].nunique()),
            "test_users": int(self.test["user_id"].nunique()),
            "train_items": int(self.train["product_id"].nunique()),
        }


def split_temporal(
    interactions: pd.DataFrame,
    config: SplitConfig | None = None,
    *,
    time_column: str = "occurred_at",
) -> TemporalSplit:
    """Split interactions into three contiguous time windows.

    Boundaries are quantiles of the *interaction* timeline rather than of the
    calendar. Calendar thirds would put very different volumes in each fold
    whenever traffic is seasonal, which makes fold-to-fold metric comparisons
    partly an artefact of volume.
    """
    config = config or SplitConfig()
    if interactions.empty:
        raise ValueError("cannot split an empty interaction frame")

    ordered = interactions.sort_values(time_column, kind="stable")
    times = ordered[time_column]

    train_end = times.quantile(config.train_fraction, interpolation="nearest")
    validation_end = times.quantile(
        config.train_fraction + config.validation_fraction, interpolation="nearest"
    )

    train = ordered[times < train_end]
    validation = ordered[(times >= train_end) & (times < validation_end)]
    test = ordered[times >= validation_end]

    split = TemporalSplit(
        train=train,
        validation=validation,
        test=test,
        train_end=pd.Timestamp(train_end).to_pydatetime(),
        validation_end=pd.Timestamp(validation_end).to_pydatetime(),
    )
    assert_no_leakage(split, time_column=time_column)
    return split


def assert_no_leakage(split: TemporalSplit, *, time_column: str = "occurred_at") -> None:
    """Verify the folds are ordered in time and do not overlap."""
    if not split.train.empty and not split.validation.empty:
        latest_train = split.train[time_column].max()
        earliest_validation = split.validation[time_column].min()
        if latest_train >= earliest_validation:
            raise LeakageError(
                f"train contains an interaction at {latest_train} which is at or "
                f"after the first validation interaction at {earliest_validation}"
            )

    if not split.validation.empty and not split.test.empty:
        latest_validation = split.validation[time_column].max()
        earliest_test = split.test[time_column].min()
        if latest_validation >= earliest_test:
            raise LeakageError(
                f"validation contains an interaction at {latest_validation} which "
                f"is at or after the first test interaction at {earliest_test}"
            )

    if (
        not split.train.empty
        and not split.test.empty
        and split.train[time_column].max() >= split.test[time_column].min()
    ):
        raise LeakageError("train and test windows overlap")


def assert_features_respect_cutoff(
    frame: pd.DataFrame,
    as_of: dt.datetime,
    *,
    time_column: str = "occurred_at",
) -> None:
    """Assert that no row used to build a feature is at or after the cut-off.

    Called by the feature builders. It is the mechanical enforcement of the
    rule in `architecture.md` section 6: a feature computed for a row at time
    `t` may only read events strictly before `t`.
    """
    if frame.empty:
        return
    latest = frame[time_column].max()
    if latest >= as_of:
        raise LeakageError(
            f"feature input contains data at {latest}, which is at or after the "
            f"as_of cut-off {as_of}. This would leak the label window into the "
            f"features."
        )


def build_evaluation_targets(
    split: TemporalSplit,
    *,
    positive_events: tuple[str, ...] = ("PURCHASE", "ADD_TO_CART", "PRODUCT_CLICK"),
    fold: str = "test",
) -> dict[int, set[int]]:
    """Ground truth for ranking metrics: user -> set of relevant products.

    Only genuinely positive events count. Including plain views would make the
    target "did the user look at this?", which popularity alone answers well
    and which is not the business question.
    """
    frame = getattr(split, fold)
    positives = frame[frame["event_type"].isin(positive_events)]
    if positives.empty:
        return {}
    grouped = positives.groupby("user_id")["product_id"].apply(set)
    return {int(user): {int(p) for p in products} for user, products in grouped.items()}


def eligible_users(
    split: TemporalSplit,
    targets: dict[int, set[int]],
    config: SplitConfig | None = None,
) -> list[int]:
    """Users with enough training history to be fairly scored.

    Users below the threshold are not discarded from the project - they are
    evaluated separately as the cold segment (R-05). Mixing them into the
    aggregate would report one number that describes neither group.
    """
    config = config or SplitConfig()
    counts = split.train.groupby("user_id").size()
    return sorted(
        user
        for user in targets
        if counts.get(user, 0) >= config.min_train_interactions
    )


__all__ = [
    "LeakageError",
    "TemporalSplit",
    "assert_features_respect_cutoff",
    "assert_no_leakage",
    "build_evaluation_targets",
    "eligible_users",
    "split_temporal",
]
