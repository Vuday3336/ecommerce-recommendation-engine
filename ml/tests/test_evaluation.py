"""Tests for the evaluation metrics and significance testing.

Metric implementations are worth testing precisely because they are the thing
every conclusion rests on. A subtly wrong NDCG does not error - it just makes
the wrong model look best, and nothing downstream notices.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from recsys.evaluation.metrics import (
    average_precision_at_k,
    catalogue_coverage,
    gini_coefficient,
    graded_ndcg_at_k,
    hit_rate_at_k,
    intra_list_diversity,
    ndcg_at_k,
    novelty,
    personalisation,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    serendipity,
)
from recsys.evaluation.significance import (
    paired_bootstrap,
    sample_ratio_mismatch,
    two_proportion_test,
)


class TestAccuracyMetrics:
    def test_precision_counts_hits_in_the_window(self):
        assert precision_at_k([1, 2, 3, 4], {2, 4}, 4) == 0.5
        assert precision_at_k([1, 2, 3, 4], {2, 4}, 2) == 0.5
        assert precision_at_k([2, 1, 3, 4], {2, 4}, 2) == 0.5

    def test_precision_divides_by_k_not_by_list_length(self):
        """A short list must not be rewarded for having fewer chances to miss."""
        assert precision_at_k([2], {2, 4}, 10) == pytest.approx(0.1)

    def test_recall_divides_by_the_relevant_set(self):
        assert recall_at_k([1, 2, 3], {2, 4}, 3) == 0.5
        assert recall_at_k([1, 2, 3], set(), 3) == 0.0

    def test_ndcg_rewards_earlier_hits(self):
        early = ndcg_at_k([1, 9, 9, 9, 9], {1}, 5)
        late = ndcg_at_k([9, 9, 9, 9, 1], {1}, 5)
        assert early > late
        assert early == pytest.approx(1.0)

    def test_ndcg_is_one_for_a_perfect_ranking(self):
        assert ndcg_at_k([1, 2, 3], {1, 2, 3}, 3) == pytest.approx(1.0)

    def test_ndcg_is_zero_when_nothing_relevant_is_found(self):
        assert ndcg_at_k([7, 8, 9], {1, 2}, 3) == 0.0

    def test_ndcg_discount_matches_the_definition(self):
        # A single hit at rank 2 gives DCG = 1/log2(3), IDCG = 1/log2(2) = 1.
        assert ndcg_at_k([9, 1, 9], {1}, 3) == pytest.approx(1.0 / math.log2(3))

    def test_graded_ndcg_prefers_the_higher_grade_first(self):
        gains = {1: 3.0, 2: 1.0}
        assert graded_ndcg_at_k([1, 2], gains, 2) > graded_ndcg_at_k([2, 1], gains, 2)

    def test_average_precision_rewards_early_clustering(self):
        clustered = average_precision_at_k([1, 2, 9, 9], {1, 2}, 4)
        spread = average_precision_at_k([1, 9, 9, 2], {1, 2}, 4)
        assert clustered > spread
        assert clustered == pytest.approx(1.0)

    def test_hit_rate_is_binary(self):
        assert hit_rate_at_k([1, 2, 3], {3}, 3) == 1.0
        assert hit_rate_at_k([1, 2, 3], {4}, 3) == 0.0

    def test_reciprocal_rank_is_the_inverse_position(self):
        assert reciprocal_rank([9, 9, 1], {1}, 5) == pytest.approx(1 / 3)
        assert reciprocal_rank([9, 9, 9], {1}, 5) == 0.0


class TestBeyondAccuracyMetrics:
    def test_coverage_counts_distinct_recommended_products(self):
        assert catalogue_coverage([[1, 2], [2, 3]], 10) == pytest.approx(0.3)
        assert catalogue_coverage([], 10) == 0.0

    def test_gini_is_zero_for_uniform_exposure(self):
        assert gini_coefficient([5, 5, 5, 5]) == pytest.approx(0.0, abs=1e-9)

    def test_gini_approaches_one_when_one_item_takes_everything(self):
        assert gini_coefficient([0, 0, 0, 100]) > 0.7

    def test_diversity_is_higher_across_categories(self):
        categories = {1: 10, 2: 10, 3: 20, 4: 30}
        same = intra_list_diversity([1, 2], categories=categories)
        mixed = intra_list_diversity([1, 3, 4], categories=categories)
        assert same == 0.0
        assert mixed == pytest.approx(1.0)

    def test_novelty_rewards_rare_items(self):
        popularity = {1: 0.5, 2: 0.001}
        assert novelty([2], popularity) > novelty([1], popularity)

    def test_serendipity_ignores_hits_the_baseline_already_found(self):
        """The whole point: re-finding what popularity surfaces adds nothing."""
        relevant = {5}
        assert serendipity([5], relevant, [5, 6, 7], 3) == 0.0
        assert serendipity([5], relevant, [6, 7, 8], 3) > 0.0

    def test_personalisation_is_zero_when_everyone_sees_the_same_list(self):
        assert personalisation([[1, 2, 3]] * 20, k=3) == pytest.approx(0.0)

    def test_personalisation_is_one_for_disjoint_lists(self):
        lists = [[i, i + 100, i + 200] for i in range(20)]
        assert personalisation(lists, k=3) == pytest.approx(1.0)


class TestSignificance:
    def test_paired_bootstrap_detects_a_consistent_improvement(self):
        rng = np.random.default_rng(1)
        baseline = {i: float(rng.random()) for i in range(600)}
        candidate = {i: v + 0.12 for i, v in baseline.items()}

        result = paired_bootstrap(baseline, candidate)
        assert result.significant
        assert result.difference == pytest.approx(0.12, abs=0.01)
        assert result.ci_low > 0
        assert "better" in result.verdict()

    def test_paired_bootstrap_reports_no_difference_for_noise(self):
        rng = np.random.default_rng(2)
        baseline = {i: float(rng.random()) for i in range(400)}
        candidate = {i: float(rng.random()) for i in range(400)}

        result = paired_bootstrap(baseline, candidate)
        assert not result.significant
        assert result.p_value > 0.05

    def test_paired_bootstrap_never_reports_zero_probability(self):
        """A p-value of exactly 0 would overstate what resampling established."""
        baseline = dict.fromkeys(range(300), 0.0)
        candidate = dict.fromkeys(range(300), 1.0)
        assert paired_bootstrap(baseline, candidate).p_value > 0

    def test_paired_bootstrap_requires_shared_users(self):
        with pytest.raises(ValueError, match="share no evaluated users"):
            paired_bootstrap({1: 0.5}, {2: 0.5})

    def test_two_proportion_test_detects_a_real_lift(self):
        result = two_proportion_test(500, 10_000, 650, 10_000)
        assert result.significant
        assert result.relative_lift == pytest.approx(0.30, abs=0.01)

    def test_two_proportion_test_is_quiet_on_a_small_difference(self):
        result = two_proportion_test(500, 10_000, 510, 10_000)
        assert not result.significant

    def test_sample_ratio_mismatch_passes_a_fair_split(self):
        p_value, mismatched = sample_ratio_mismatch(
            {"control": 4980, "treatment": 5020}, {"control": 0.5, "treatment": 0.5}
        )
        assert not mismatched
        assert p_value > 0.05

    def test_sample_ratio_mismatch_catches_a_skewed_split(self):
        """The guardrail that stops a broken experiment from shipping."""
        p_value, mismatched = sample_ratio_mismatch(
            {"control": 6500, "treatment": 3500}, {"control": 0.5, "treatment": 0.5}
        )
        assert mismatched
        assert p_value < 0.001
