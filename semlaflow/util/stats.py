"""Statistics for arm-vs-arm comparison, with the pairing assumption corrected.

Why this exists. compare_arms.py matches arms on the **size sequence** drawn from the test set:
both arms use the same seed, so slot i in each arm is sized to match the same real test molecule.
That is a legitimate blocking variable and worth keeping -- it reduces variance -- but it is *not*
true pairing. Arm A's molecule i and arm B's molecule i are different molecules that merely have
the same atom count, and Wilcoxon signed-rank assumes genuinely paired observations. So:

  - The size-matched comparison is reported as a variance-reduced **descriptive**.
  - The formal claim uses **unpaired** tests: Mann-Whitney U, plus a bootstrap CI on the
    difference of medians.
  - Every test is reported with an **effect size**. A p-value of 1e-195 at n=2000 says there is a
    systematic difference of unknown size; it says nothing about whether the difference matters.

Across seeds, formal testing has almost no power with three of them. Show the three seed-level
values per arm instead: if the worst seed of one arm beats the best seed of the other, that is
persuasive without a p-value.
"""

import numpy as np
from scipy.stats import mannwhitneyu

DEFAULT_N_BOOTSTRAP = 10000
DEFAULT_ALPHA = 0.05


def _clean(values):
    return np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)


def cliffs_delta(values_a, values_b) -> float:
    """Effect size for a Mann-Whitney comparison: P(a > b) - P(a < b), in [-1, 1].

    Chosen over a mean-difference effect size because it is rank-based, so it makes the same
    distributional assumptions as the test it accompanies, and it is unitless -- comparable across
    metrics measured in kcal/mol, Angstrom and dimensionless ratios alike.

    Rough convention: |d| < 0.147 negligible, < 0.33 small, < 0.474 medium, else large.
    """

    a, b = _clean(values_a), _clean(values_b)
    if a.size == 0 or b.size == 0:
        return float("nan")

    # Rank-based identity for the pairwise dominance count, so this stays O(n log n) rather than
    # materialising the n_a x n_b comparison matrix
    combined = np.concatenate([a, b])
    order = combined.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, combined.size + 1)

    # Average ranks over ties so the identity holds with duplicate values present
    _, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    tie_sums = np.zeros(counts.size)
    np.add.at(tie_sums, inverse, ranks)
    ranks = (tie_sums / counts)[inverse]

    rank_sum_a = ranks[: a.size].sum()
    u_a = rank_sum_a - a.size * (a.size + 1) / 2.0
    return float(2.0 * u_a / (a.size * b.size) - 1.0)


def mann_whitney(values_a, values_b) -> dict:
    """Unpaired two-sided Mann-Whitney U, with its effect size."""

    a, b = _clean(values_a), _clean(values_b)
    result = {"n_a": int(a.size), "n_b": int(b.size), "u": None, "p": None, "cliffs_delta": None}

    if a.size < 2 or b.size < 2:
        return result

    try:
        u_stat, p_value = mannwhitneyu(a, b, alternative="two-sided")
    except ValueError:
        return result

    result["u"] = float(u_stat)
    result["p"] = float(p_value)
    result["cliffs_delta"] = cliffs_delta(a, b)
    return result


def bootstrap_median_difference(
    values_a, values_b, n_bootstrap: int = DEFAULT_N_BOOTSTRAP, alpha: float = DEFAULT_ALPHA, seed: int = 0
) -> dict:
    """Percentile bootstrap CI on median(a) - median(b), resampling each arm independently.

    Independent resampling is the point: it matches the unpaired structure of the data. A paired
    bootstrap would reimport exactly the pairing assumption this module exists to remove.
    """

    a, b = _clean(values_a), _clean(values_b)
    result = {"diff": None, "ci_low": None, "ci_high": None, "n_bootstrap": n_bootstrap}

    if a.size < 2 or b.size < 2:
        return result

    rng = np.random.default_rng(seed)
    result["diff"] = float(np.median(a) - np.median(b))

    draws_a = rng.choice(a, size=(n_bootstrap, a.size), replace=True)
    draws_b = rng.choice(b, size=(n_bootstrap, b.size), replace=True)
    diffs = np.median(draws_a, axis=1) - np.median(draws_b, axis=1)

    result["ci_low"] = float(np.quantile(diffs, alpha / 2.0))
    result["ci_high"] = float(np.quantile(diffs, 1.0 - alpha / 2.0))
    return result


def describe(values) -> dict:
    """Median AND mean, because the metrics here are heavily right-skewed."""

    clean = _clean(values)
    if clean.size == 0:
        return {"n": 0, "median": None, "mean": None, "std": None}

    return {
        "n": int(clean.size),
        "median": float(np.median(clean)),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)) if clean.size > 1 else 0.0,
    }


def compare_metric(values_a, values_b, n_bootstrap: int = DEFAULT_N_BOOTSTRAP, seed: int = 0) -> dict:
    """Everything that should be reported for one metric on one pair of arms."""

    return {
        "a": describe(values_a),
        "b": describe(values_b),
        "test": mann_whitney(values_a, values_b),
        "bootstrap": bootstrap_median_difference(values_a, values_b, n_bootstrap=n_bootstrap, seed=seed),
    }


def size_matched_difference(values_a, values_b) -> dict:
    """The slot-matched mean difference, kept as a variance-reduced DESCRIPTIVE only.

    Matching on size is real blocking and does reduce variance, so the number is informative. It
    just does not license a paired significance test, so none is computed here.
    """

    diffs = [
        a - b
        for a, b in zip(values_a, values_b)
        if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)
    ]

    if not diffs:
        return {"n_slots": 0, "mean_difference": None, "median_difference": None}

    return {
        "n_slots": len(diffs),
        "mean_difference": float(np.mean(diffs)),
        "median_difference": float(np.median(diffs)),
    }
