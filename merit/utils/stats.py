"""Statistical analysis utilities for MERIT experiments.

Provides bootstrap confidence intervals, effect sizes, correlation analysis,
and multi-run aggregation for rigorous experiment reporting.
"""
import numpy as np
from scipy.stats import spearmanr
from typing import List, Tuple, Dict, Optional


def bootstrap_ci(
    data: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    statistic: str = "mean",
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for a statistic.

    Args:
        data: 1D array of observations
        confidence: Confidence level (default 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap resamples
        statistic: Which statistic to compute ("mean" or "median")
        seed: Random seed for reproducibility

    Returns:
        (lower_bound, upper_bound) tuple
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    stat_fn = np.mean if statistic == "mean" else np.median

    boot_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_stats[i] = stat_fn(sample)

    alpha = 1 - confidence
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups.

    Uses pooled standard deviation. Positive d means group1 > group2.

    Interpretation:
        |d| < 0.2: negligible
        0.2 <= |d| < 0.5: small
        0.5 <= |d| < 0.8: medium
        |d| >= 0.8: large
    """
    group1 = np.asarray(group1, dtype=float)
    group2 = np.asarray(group2, dtype=float)

    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def spearman_with_ci(
    x: list,
    y: list,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: Optional[int] = None,
) -> Tuple[float, float, Tuple[float, float]]:
    """Spearman rank correlation with bootstrap confidence interval.

    Args:
        x, y: Two sequences of the same length
        confidence: Confidence level
        n_bootstrap: Number of bootstrap resamples
        seed: Random seed

    Returns:
        (rho, p_value, (ci_lower, ci_upper))
    """
    x = np.asarray(x)
    y = np.asarray(y)

    rho, p_value = spearmanr(x, y)

    # Bootstrap CI for the correlation
    rng = np.random.default_rng(seed)
    boot_rhos = np.empty(n_bootstrap)
    n = len(x)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        r, _ = spearmanr(x[idx], y[idx])
        boot_rhos[i] = r

    alpha = 1 - confidence
    ci_lower = float(np.percentile(boot_rhos, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_rhos, 100 * (1 - alpha / 2)))

    return float(rho), float(p_value), (ci_lower, ci_upper)


def aggregate_runs(runs: List[List[float]]) -> Dict[str, float]:
    """Aggregate multiple experiment runs with comprehensive statistics.

    Args:
        runs: List of runs, each run is a list of per-sample scores.
              e.g., [[0.8, 0.7, 0.9], [0.85, 0.72, 0.88], [0.82, 0.71, 0.91]]

    Returns:
        Dict with mean, std, ci_low, ci_high, min, max, n_runs, n_samples
    """
    # Compute per-run means
    run_means = [float(np.mean(run)) for run in runs]

    overall_mean = float(np.mean(run_means))
    overall_std = float(np.std(run_means, ddof=1)) if len(run_means) > 1 else 0.0

    ci_low, ci_high = bootstrap_ci(np.array(run_means), seed=42)

    return {
        "mean": overall_mean,
        "std": overall_std,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "min": float(min(run_means)),
        "max": float(max(run_means)),
        "n_runs": len(runs),
        "n_samples": len(runs[0]) if runs else 0,
    }
