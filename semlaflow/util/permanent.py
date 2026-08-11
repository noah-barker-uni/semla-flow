"""Exact permanent marginals via Ryser's formula, for small n.

The whole project rests on one claim: the hard Hungarian assignment is a biased point estimate of
a posterior mean over permutations, and Sinkhorn is a mean-field approximation to that posterior
which is "systematically more diffuse than the truth". Both halves of that are assertions until
someone computes the truth. For n <= 12 the truth is computable, so it should be computed.

The exact object is the matrix of permanent marginals of the weight matrix W = exp(-cost / eps):

    M_ij = P(sigma(i) = j) = W_ij * perm(W with row i and column j deleted) / perm(W)

M is doubly stochastic, because sigma is a bijection. Sinkhorn's plan is the mean-field
approximation to M and is not required to match it.

Cost: perm() is #P-hard in general, but Ryser's formula evaluates it in O(2^n * n), and every
marginal needs one (n-1)-minor, so the whole matrix is O(n^2 * 2^n * n). At n=12 that is a few
million floating point operations -- CPU-seconds, which is why this comparison is worth doing.
"""

import numpy as np

# Beyond this the 2^n subset enumeration stops being cheap and the float64 alternating sum starts
# losing precision to cancellation.
MAX_EXACT_N = 14


def permanent(matrix: np.ndarray) -> float:
    """Permanent of a square matrix by Ryser's formula.

        perm(A) = (-1)^n * sum_{S subset of [n]} (-1)^{|S|} * prod_i sum_{j in S} A_ij

    Evaluated over all 2^n column subsets at once. Note the alternating sum means catastrophic
    cancellation is possible for badly scaled matrices -- scale before calling (permanent_marginals
    does).
    """

    matrix = np.asarray(matrix, dtype=np.float64)
    n = matrix.shape[0]

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square, got {matrix.shape}.")

    if n == 0:
        return 1.0

    if n > MAX_EXACT_N:
        raise ValueError(f"n={n} exceeds MAX_EXACT_N={MAX_EXACT_N}; the exact permanent is intractable.")

    # membership[s, j] = 1 if column j is in subset s
    subsets = np.arange(1 << n, dtype=np.int64)
    membership = ((subsets[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)

    row_sums = membership @ matrix.T            # [2^n, n]
    products = row_sums.prod(axis=1)            # [2^n]
    signs = np.where(membership.sum(axis=1) % 2 == 0, 1.0, -1.0)

    return float(((-1) ** n) * (signs * products).sum())


def permanent_marginals(weights: np.ndarray) -> np.ndarray:
    """Exact P(sigma(i) = j) for the distribution p(sigma) proportional to prod_i W_{i,sigma(i)}.

    Args:
        weights (np.ndarray): Non-negative weight matrix [n, n], eg. exp(-cost / eps).

    Returns:
        np.ndarray: Doubly stochastic marginal matrix [n, n].
    """

    weights = np.asarray(weights, dtype=np.float64)
    n = weights.shape[0]

    if weights.shape[0] != weights.shape[1]:
        raise ValueError(f"weights must be square, got {weights.shape}.")

    if n == 1:
        return np.ones((1, 1))

    # A global scale cancels in the M_ij ratio, so normalise to keep the products near 1 and the
    # alternating sum well conditioned
    scale = weights.max()
    if scale <= 0:
        raise ValueError("weights must contain at least one positive entry.")
    weights = weights / scale

    total = permanent(weights)
    if total <= 0:
        raise ValueError("permanent is non-positive; weights are degenerate or badly conditioned.")

    marginals = np.zeros((n, n))
    all_rows = np.arange(n)
    for i in range(n):
        rows = all_rows[all_rows != i]
        for j in range(n):
            cols = all_rows[all_rows != j]
            minor = weights[np.ix_(rows, cols)]
            marginals[i, j] = weights[i, j] * permanent(minor)

    return marginals / total


def weights_from_cost(cost: np.ndarray, eps: float) -> np.ndarray:
    """exp(-cost / eps), shifted so the largest weight is 1 and nothing underflows."""

    cost = np.asarray(cost, dtype=np.float64)
    return np.exp(-(cost - cost.min()) / eps)


def normalised_entropy(plan: np.ndarray) -> float:
    """Mean row entropy of a row-stochastic matrix, divided by log n so n cancels."""

    plan = np.asarray(plan, dtype=np.float64)
    n = plan.shape[0]
    if n < 2:
        return 0.0

    rows = plan / plan.sum(axis=1, keepdims=True).clip(min=1e-300)
    entropy = -(rows * np.log(rows.clip(min=1e-300))).sum(axis=1)
    return float(entropy.mean() / np.log(n))
