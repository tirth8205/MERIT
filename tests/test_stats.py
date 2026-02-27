"""Tests for statistical utilities."""
import pytest
import numpy as np
from merit.utils.stats import bootstrap_ci, cohens_d, spearman_with_ci, aggregate_runs


class TestBootstrapCI:
    def test_basic_ci(self):
        rng = np.random.default_rng(42)
        data = rng.normal(5.0, 1.0, size=100)
        low, high = bootstrap_ci(data, confidence=0.95, seed=42)
        assert low < 5.0 < high
        assert low > 4.0  # Shouldn't be too wide
        assert high < 6.0

    def test_narrow_ci_with_low_variance(self):
        data = np.array([5.0, 5.01, 4.99, 5.0, 5.0])
        low, high = bootstrap_ci(data, seed=42)
        assert high - low < 0.1  # Very tight CI

    def test_ci_contains_mean(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        low, high = bootstrap_ci(data, seed=42)
        assert low <= np.mean(data) <= high

    def test_median_statistic(self):
        # Use larger dataset so bootstrap median is meaningfully tighter than mean
        rng = np.random.default_rng(99)
        base = rng.normal(3.0, 1.0, size=50)
        data = np.append(base, [100.0, 200.0, 150.0])  # Add outliers
        low_mean, high_mean = bootstrap_ci(data, statistic="mean", seed=42)
        low_med, high_med = bootstrap_ci(data, statistic="median", seed=42)
        # Median CI should be tighter (less affected by outliers)
        assert (high_med - low_med) < (high_mean - low_mean)


class TestCohensD:
    def test_large_effect(self):
        a = np.array([10.0, 11.0, 12.0, 10.5, 11.5])
        b = np.array([5.0, 6.0, 5.5, 6.5, 5.0])
        d = cohens_d(a, b)
        assert d > 0.8  # Large effect

    def test_no_effect(self):
        a = np.array([5.0, 5.0, 5.0])
        b = np.array([5.0, 5.0, 5.0])
        d = cohens_d(a, b)
        assert d == 0.0

    def test_negative_effect(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([10.0, 11.0, 12.0])
        d = cohens_d(a, b)
        assert d < -0.8  # Large negative effect

    def test_small_effect(self):
        rng = np.random.default_rng(42)
        a = rng.normal(5.0, 1.0, size=100)
        b = rng.normal(4.9, 1.0, size=100)  # Very slight difference
        d = cohens_d(a, b)
        assert abs(d) < 0.5  # Small effect


class TestSpearmanWithCI:
    def test_perfect_correlation(self):
        x = list(range(10))
        y = list(range(10))
        rho, p, (low, high) = spearman_with_ci(x, y, seed=42)
        assert rho == pytest.approx(1.0, abs=1e-9)
        assert p < 0.001
        assert low > 0.8

    def test_negative_correlation(self):
        x = list(range(10))
        y = list(range(9, -1, -1))
        rho, p, (low, high) = spearman_with_ci(x, y, seed=42)
        assert rho == pytest.approx(-1.0, abs=1e-9)
        assert high < -0.8

    def test_no_correlation(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 50).tolist()
        y = rng.normal(0, 1, 50).tolist()
        rho, p, (low, high) = spearman_with_ci(x, y, seed=42)
        assert abs(rho) < 0.4  # Should be near zero


class TestAggregateRuns:
    def test_basic_aggregation(self):
        runs = [[0.8, 0.7, 0.9], [0.85, 0.72, 0.88], [0.82, 0.71, 0.91]]
        stats = aggregate_runs(runs)
        assert "mean" in stats
        assert "std" in stats
        assert "ci_low" in stats
        assert "ci_high" in stats
        assert "n_runs" in stats
        assert stats["n_runs"] == 3
        assert stats["n_samples"] == 3
        assert 0.7 < stats["mean"] < 0.9

    def test_single_run(self):
        runs = [[0.8, 0.7, 0.9]]
        stats = aggregate_runs(runs)
        assert stats["std"] == 0.0
        assert stats["n_runs"] == 1

    def test_ci_contains_mean(self):
        runs = [[0.8, 0.7, 0.9], [0.85, 0.72, 0.88], [0.82, 0.71, 0.91], [0.79, 0.68, 0.87], [0.83, 0.73, 0.89]]
        stats = aggregate_runs(runs)
        assert stats["ci_low"] <= stats["mean"] <= stats["ci_high"]
