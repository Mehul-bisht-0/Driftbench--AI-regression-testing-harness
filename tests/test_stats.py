"""Tests for driftbench.stats"""

from __future__ import annotations

import math

from driftbench import stats


class TestWilson:
    def test_perfect(self):
        iv = stats.wilson(20, 20)
        assert iv.lo > 0.8
        assert iv.hi == 1.0

    def test_zero(self):
        iv = stats.wilson(0, 20)
        assert iv.lo == 0.0
        assert iv.hi < 0.2

    def test_half(self):
        iv = stats.wilson(10, 20)
        assert 0.2 < iv.lo < 0.6
        assert 0.4 < iv.hi < 0.8

    def test_empty(self):
        iv = stats.wilson(0, 0)
        assert iv.lo == 0.0
        assert iv.hi == 1.0

    def test_contains_true_rate(self):
        iv = stats.wilson(15, 20, confidence=0.95)
        assert iv.contains(0.75)

    def test_width_decreases_with_n(self):
        narrow = stats.wilson(80, 100)
        wide = stats.wilson(8, 10)
        assert narrow.width < wide.width


class TestFisherExact:
    def test_identical(self):
        p = stats.fisher_exact(5, 0, 5, 0)
        assert p == 1.0  # no difference

    def test_clear_difference(self):
        # 10/10 vs 0/10
        p = stats.fisher_exact(10, 0, 0, 10)
        assert p < 0.01

    def test_symmetry(self):
        p1 = stats.fisher_exact(8, 2, 3, 7)
        p2 = stats.fisher_exact(3, 7, 8, 2)
        assert abs(p1 - p2) < 1e-9

    def test_one_sided_extreme(self):
        p = stats.fisher_exact(5, 0, 0, 5)
        assert p < 0.05


class TestBenjaminiHochberg:
    def test_all_significant(self):
        # All p-values very small
        result = stats.benjamini_hochberg([0.001, 0.002, 0.003], fdr=0.10)
        assert all(result)

    def test_none_significant(self):
        result = stats.benjamini_hochberg([0.5, 0.6, 0.7], fdr=0.10)
        assert not any(result)

    def test_mixed(self):
        result = stats.benjamini_hochberg([0.01, 0.5, 0.9], fdr=0.10)
        assert result[0] is True  # smallest
        # Others may or may not pass depending on rank

    def test_empty(self):
        assert stats.benjamini_hochberg([], fdr=0.10) == []

    def test_single(self):
        assert stats.benjamini_hochberg([0.05], fdr=0.10) == [True]
        assert stats.benjamini_hochberg([0.20], fdr=0.10) == [False]


class TestShannonEntropy:
    def test_uniform(self):
        e = stats.shannon_entropy([1, 1, 1, 1])
        assert abs(e - 2.0) < 0.01  # log2(4) = 2

    def test_all_same(self):
        e = stats.shannon_entropy([10])
        assert e == 0.0

    def test_empty(self):
        assert stats.shannon_entropy([]) == 0.0

    def test_two_classes(self):
        e = stats.shannon_entropy([1, 1])
        assert abs(e - 1.0) < 0.01  # log2(2) = 1


class TestNormalizedEntropy:
    def test_one_class(self):
        assert stats.normalized_entropy([10]) == 0.0

    def test_max_one_class_each(self):
        # 10 classes, each with 1 observation
        ne = stats.normalized_entropy([1] * 10)
        assert ne > 0.9  # close to 1.0

    def test_empty(self):
        assert stats.normalized_entropy([]) == 0.0


class TestOutcomeFlake:
    def test_perfect(self):
        assert stats.outcome_flake(20, 20) == 0.0
        assert stats.outcome_flake(0, 20) == 0.0

    def test_coin_flip(self):
        f = stats.outcome_flake(10, 20)
        assert abs(f - 1.0) < 0.01

    def test_empty(self):
        assert stats.outcome_flake(0, 0) == 0.0


class TestCohensKappa:
    def test_perfect_agreement(self):
        pairs = [("good", "good")] * 20
        a = stats.cohens_kappa(pairs)
        # When expected agreement is 1.0 (all same class), kappa is 0.0
        # by design: kappa is undefined when one class has 100% prevalence.
        # The raw agreement is 1.0, but kappa cannot distinguish this from
        # a trivial all-good judge.
        assert a.raw == 1.0
        assert a.kappa == 0.0  # undefined, returned as 0.0

    def test_no_agreement(self):
        # 50% good, 50% bad, judge always says opposite
        pairs = [("good", "bad")] * 10 + [("bad", "good")] * 10
        a = stats.cohens_kappa(pairs)
        assert a.kappa < 0.0

    def test_chance_agreement(self):
        # Judge always says "good" on a 50/50 sample
        pairs = [("good", "good")] * 10 + [("bad", "good")] * 10
        a = stats.cohens_kappa(pairs)
        assert a.kappa < 0.1  # near zero after correction

    def test_empty(self):
        a = stats.cohens_kappa([])
        assert a.n == 0
        assert a.kappa == 0.0


class TestBootstrapCI:
    def test_narrow_for_consistent(self):
        vals = [5.0] * 20
        iv = stats.bootstrap_ci(vals)
        assert iv.width < 0.1

    def test_contains_mean(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        iv = stats.bootstrap_ci(vals)
        assert iv.contains(3.0)


class TestProportionDelta:
    def test_no_change(self):
        d = stats.proportion_delta(10, 20, 10, 20)
        assert abs(d["delta"]) < 1e-9
        assert d["p_value"] == 1.0

    def test_regression(self):
        d = stats.proportion_delta(20, 20, 15, 20)
        assert d["delta"] < 0
        assert d["p_value"] < 1.0
