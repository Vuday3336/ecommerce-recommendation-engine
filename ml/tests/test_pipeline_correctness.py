"""Tests for the properties the whole evaluation depends on.

These are the tests that matter most in an ML project, because the failures
they catch are silent. A leaking split, a mis-scaled weight or a
training/serving skew produces no error - only better-looking numbers and a
model that underperforms in production.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from recsys.config.settings import SplitConfig, WeightingConfig
from recsys.drift.detector import (
    DriftDetector,
    DriftSeverity,
    classify,
    population_stability_index,
)
from recsys.preprocessing.splitting import (
    LeakageError,
    TemporalSplit,
    assert_features_respect_cutoff,
    assert_no_leakage,
    split_temporal,
)
from recsys.preprocessing.weighting import (
    build_interaction_matrix_frame,
    calibrate_weights,
)
from recsys.registry.promotion import GateOutcome, evaluate_gate


def make_interactions(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-01", tz="UTC")
    return pd.DataFrame(
        {
            "user_id": rng.integers(1, 200, size=n),
            "product_id": rng.integers(1, 400, size=n),
            "event_type": rng.choice(
                ["PRODUCT_VIEW", "PRODUCT_CLICK", "ADD_TO_CART", "PURCHASE"],
                size=n,
                p=[0.6, 0.25, 0.1, 0.05],
            ),
            "occurred_at": [
                start + pd.Timedelta(hours=int(h)) for h in rng.integers(0, 24 * 120, size=n)
            ],
        }
    ).sort_values("occurred_at")


class TestTemporalSplit:
    def test_folds_are_ordered_in_time(self):
        split = split_temporal(make_interactions())
        assert split.train["occurred_at"].max() < split.validation["occurred_at"].min()
        assert split.validation["occurred_at"].max() < split.test["occurred_at"].min()

    def test_fractions_are_approximately_respected(self):
        split = split_temporal(make_interactions(5000), SplitConfig(0.7, 0.15))
        total = len(split.train) + len(split.validation) + len(split.test)
        assert len(split.train) / total == pytest.approx(0.70, abs=0.03)
        assert len(split.validation) / total == pytest.approx(0.15, abs=0.03)

    def test_leakage_guard_rejects_an_overlapping_split(self):
        """The guard has to actually fire, or it is decoration."""
        frame = make_interactions(500)
        bad = TemporalSplit(
            train=frame,
            validation=frame,
            test=frame,
            train_end=frame["occurred_at"].max().to_pydatetime(),
            validation_end=frame["occurred_at"].max().to_pydatetime(),
        )
        with pytest.raises(LeakageError, match="at or after"):
            assert_no_leakage(bad)

    def test_feature_cutoff_guard_rejects_future_rows(self):
        frame = make_interactions(200)
        cutoff = frame["occurred_at"].quantile(0.5).to_pydatetime()
        with pytest.raises(LeakageError, match="as_of cut-off"):
            assert_features_respect_cutoff(frame, cutoff)

    def test_feature_cutoff_guard_accepts_past_rows(self):
        frame = make_interactions(200)
        cutoff = frame["occurred_at"].max().to_pydatetime() + dt.timedelta(days=1)
        assert_features_respect_cutoff(frame, cutoff)

    def test_empty_input_is_rejected_rather_than_silently_split(self):
        with pytest.raises(ValueError, match="empty"):
            split_temporal(pd.DataFrame(columns=["user_id", "product_id", "occurred_at"]))


def make_funnel_interactions(n_pairs: int = 6000, seed: int = 11) -> pd.DataFrame:
    """Interactions with a real funnel, for testing weight calibration.

    `make_interactions` samples event types independently of outcome, so no
    event predicts a purchase and the calibration correctly derives no
    ordering. That is the right behaviour but the wrong fixture for testing
    whether calibration *recovers* structure - so this one injects the funnel
    the calibration is supposed to find:

        every pair is viewed; 30% are clicked; 40% of those are carted;
        60% of carted pairs are purchased.

    Carting therefore genuinely predicts purchase far better than viewing, and
    a correct implementation has to discover that from the data alone.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-01", tz="UTC")
    rows: list[dict] = []

    for index in range(n_pairs):
        user_id = int(rng.integers(1, 400))
        product_id = int(rng.integers(1, 600))
        base = start + pd.Timedelta(hours=int(rng.integers(0, 24 * 120)))

        rows.append({"user_id": user_id, "product_id": product_id,
                     "event_type": "PRODUCT_VIEW", "occurred_at": base})

        if rng.random() >= 0.30:
            continue
        rows.append({"user_id": user_id, "product_id": product_id,
                     "event_type": "PRODUCT_CLICK",
                     "occurred_at": base + pd.Timedelta(minutes=2)})

        if rng.random() >= 0.40:
            continue
        rows.append({"user_id": user_id, "product_id": product_id,
                     "event_type": "ADD_TO_CART",
                     "occurred_at": base + pd.Timedelta(minutes=5)})

        if rng.random() >= 0.60:
            continue
        rows.append({"user_id": user_id, "product_id": product_id,
                     "event_type": "PURCHASE",
                     "occurred_at": base + pd.Timedelta(minutes=9)})
        _ = index

    return pd.DataFrame(rows).sort_values("occurred_at").reset_index(drop=True)


class TestWeightCalibration:
    def test_weights_recover_the_injected_funnel_ordering(self):
        """ADR-002's central claim, checked against derived numbers.

        The ordering is not asserted into the data - it is *discovered* from
        conversion odds the calibration was never told about.
        """
        split = split_temporal(make_funnel_interactions())
        calibration = calibrate_weights(split.train, min_support=50)
        weights = calibration.weights

        assert calibration.calibrated
        assert weights["PURCHASE"] > weights["ADD_TO_CART"] > weights["PRODUCT_CLICK"]
        assert weights["PRODUCT_CLICK"] > weights["PRODUCT_VIEW"]

    def test_no_ordering_is_invented_when_the_data_has_none(self):
        """The complement, and the more important half.

        On event types sampled independently of outcome, a correct calibration
        must report near-equal weights rather than reproducing a prior it was
        given. Asserting an ordering here would mean the implementation was
        ignoring the data.
        """
        split = split_temporal(make_interactions(20_000, seed=3))
        weights = calibrate_weights(split.train, min_support=50).weights

        # Measured for reference: on structured data these come out at 3.03
        # and 2.05, so a tolerance of 0.3 around 1.0 comfortably separates
        # "found real structure" from "found none" while absorbing the
        # chance co-occurrence that random sampling produces.
        assert weights["ADD_TO_CART"] == pytest.approx(1.0, abs=0.3)
        assert weights["PRODUCT_CLICK"] == pytest.approx(1.0, abs=0.3)

    def test_view_is_the_normalisation_anchor(self):
        split = split_temporal(make_funnel_interactions(seed=4))
        calibration = calibrate_weights(split.train, min_support=50)
        assert calibration.weights["PRODUCT_VIEW"] == pytest.approx(1.0)

    def test_low_support_events_keep_their_prior(self):
        split = split_temporal(make_funnel_interactions(1500, seed=5))
        calibration = calibrate_weights(split.train, min_support=10_000)
        # With an unreachable support threshold nothing is derived, so the
        # ordering must come from the priors rather than from noise.
        assert calibration.weights["PURCHASE"] > calibration.weights["PRODUCT_VIEW"]

    def test_falls_back_to_priors_without_purchases(self):
        frame = make_interactions(500)
        frame = frame[frame["event_type"] != "PURCHASE"]
        calibration = calibrate_weights(frame)
        assert not calibration.calibrated
        assert calibration.weights == WeightingConfig().prior_weights

    def test_recency_decay_reduces_the_weight_of_old_interactions(self):
        base = pd.Timestamp("2026-06-01", tz="UTC")
        frame = pd.DataFrame(
            {
                "user_id": [1, 2],
                "product_id": [10, 10],
                "event_type": ["PRODUCT_VIEW", "PRODUCT_VIEW"],
                "occurred_at": [base, base - pd.Timedelta(days=180)],
            }
        )
        calibration = calibrate_weights(make_funnel_interactions(seed=6), min_support=50)
        weights = build_interaction_matrix_frame(
            frame, calibration, as_of=base + dt.timedelta(days=1)
        ).set_index("user_id")["weight"]
        assert weights[1] > weights[2] * 2

    def test_weights_are_capped(self):
        base = pd.Timestamp("2026-06-01", tz="UTC")
        frame = pd.DataFrame(
            {
                "user_id": [1] * 500,
                "product_id": [10] * 500,
                "event_type": ["PURCHASE"] * 500,
                "occurred_at": [base] * 500,
            }
        )
        calibration = calibrate_weights(make_funnel_interactions(seed=7), min_support=50)
        config = WeightingConfig(max_weight=60.0)
        result = build_interaction_matrix_frame(
            frame, calibration, config, as_of=base + dt.timedelta(days=1)
        )
        assert result["weight"].max() <= 60.0


class TestDriftDetection:
    def test_identical_distributions_have_zero_psi(self):
        values = np.random.default_rng(0).normal(size=5000)
        assert population_stability_index(values, values) == pytest.approx(0.0, abs=1e-6)

    def test_a_shifted_distribution_raises_psi(self):
        rng = np.random.default_rng(1)
        reference = rng.normal(0, 1, 5000)
        shifted = rng.normal(2.5, 1, 5000)
        assert population_stability_index(reference, shifted) > 0.25

    def test_psi_is_finite_when_a_category_disappears(self):
        """Without smoothing this is infinite, and one missing category alerts."""
        reference = np.array(["a"] * 100 + ["b"] * 100)
        current = np.array(["a"] * 200)
        value = population_stability_index(reference, current, categorical=True)
        assert np.isfinite(value)
        assert value > 0

    def test_severity_thresholds(self):
        assert classify(0.05) is DriftSeverity.NONE
        assert classify(0.15) is DriftSeverity.WARNING
        assert classify(0.40) is DriftSeverity.ALERT

    def test_retrain_needs_more_than_one_alerting_feature(self):
        """One feature can move for a benign reason; two is a population shift."""
        rng = np.random.default_rng(2)
        reference = pd.DataFrame(
            {"a": rng.normal(size=3000), "b": rng.normal(size=3000), "c": rng.normal(size=3000)}
        )
        detector = DriftDetector(reference, numeric_features=("a", "b", "c"))

        one_moved = reference.copy()
        one_moved["a"] = one_moved["a"] + 3.0
        assert not detector.compare(one_moved).should_retrain

        two_moved = one_moved.copy()
        two_moved["b"] = two_moved["b"] + 3.0
        assert detector.compare(two_moved).should_retrain


class TestPromotionGate:
    def test_a_better_model_is_promoted(self):
        decision = evaluate_gate(
            {"ndcg@10": 0.062, "hit_rate@10": 0.24, "coverage": 0.30},
            {"ndcg@10": 0.058, "hit_rate@10": 0.23, "coverage": 0.28},
        )
        assert decision.outcome is GateOutcome.PROMOTE
        assert decision.should_promote

    def test_a_worse_model_is_rejected(self):
        decision = evaluate_gate(
            {"ndcg@10": 0.050, "hit_rate@10": 0.20, "coverage": 0.28},
            {"ndcg@10": 0.058, "hit_rate@10": 0.23, "coverage": 0.28},
        )
        assert decision.outcome is GateOutcome.REJECT
        assert any("ndcg@10" in line for line in decision.failed)

    def test_a_coverage_collapse_is_rejected_even_when_ndcg_improves(self):
        """The degenerate solution: recommend the same popular items to everyone."""
        decision = evaluate_gate(
            {"ndcg@10": 0.090, "hit_rate@10": 0.30, "coverage": 0.02},
            {"ndcg@10": 0.058, "hit_rate@10": 0.23, "coverage": 0.28},
        )
        assert not decision.should_promote
        assert any("coverage" in line for line in decision.failed)

    def test_a_noise_sized_improvement_is_rejected(self):
        """Promoting on noise makes the production model random-walk."""
        decision = evaluate_gate(
            {"ndcg@10": 0.05805, "hit_rate@10": 0.23, "coverage": 0.28},
            {"ndcg@10": 0.05800, "hit_rate@10": 0.23, "coverage": 0.28},
        )
        assert not decision.should_promote

    def test_the_first_model_is_promoted_when_it_clears_the_floors(self):
        decision = evaluate_gate(
            {"ndcg@10": 0.04, "hit_rate@10": 0.15, "coverage": 0.20}, None
        )
        assert decision.outcome is GateOutcome.PROMOTE_FIRST

    def test_the_first_model_is_rejected_below_the_coverage_floor(self):
        decision = evaluate_gate(
            {"ndcg@10": 0.04, "hit_rate@10": 0.15, "coverage": 0.01}, None
        )
        assert decision.outcome is GateOutcome.REJECT
