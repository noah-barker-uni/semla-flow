import math
import unittest
from itertools import permutations

import numpy as np

import semlaflow.util.permanent as permanent


def _brute_force_permanent(matrix):
    n = matrix.shape[0]
    return sum(np.prod([matrix[i, p[i]] for i in range(n)]) for p in permutations(range(n)))


def _brute_force_marginals(weights):
    n = weights.shape[0]
    marginals = np.zeros((n, n))
    total = 0.0
    for perm in permutations(range(n)):
        weight = np.prod([weights[i, perm[i]] for i in range(n)])
        total += weight
        for i in range(n):
            marginals[i, perm[i]] += weight
    return marginals / total


class RyserPermanentTests(unittest.TestCase):
    def test_matches_brute_force_enumeration(self):
        rng = np.random.default_rng(0)
        for n in [1, 2, 3, 5, 7]:
            matrix = rng.random((n, n)) + 0.1
            self.assertAlmostEqual(
                _brute_force_permanent(matrix), permanent.permanent(matrix), places=8, msg=f"n={n}"
            )

    def test_known_answers(self):
        for n in [3, 5, 8]:
            self.assertAlmostEqual(1.0, permanent.permanent(np.eye(n)), places=8)
            self.assertAlmostEqual(
                float(math.factorial(n)), permanent.permanent(np.ones((n, n))), places=4
            )

    def test_empty_matrix_is_one(self):
        self.assertEqual(1.0, permanent.permanent(np.zeros((0, 0))))

    def test_rejects_non_square(self):
        with self.assertRaises(ValueError):
            permanent.permanent(np.ones((3, 4)))

    def test_rejects_intractably_large(self):
        with self.assertRaises(ValueError):
            permanent.permanent(np.ones((permanent.MAX_EXACT_N + 1,) * 2))


class PermanentMarginalsTests(unittest.TestCase):
    def test_matches_brute_force_enumeration(self):
        rng = np.random.default_rng(1)
        for n in [2, 3, 5, 7]:
            weights = rng.random((n, n)) + 0.05
            expected = _brute_force_marginals(weights)
            np.testing.assert_almost_equal(
                permanent.permanent_marginals(weights), expected, decimal=8, err_msg=f"n={n}"
            )

    def test_marginals_are_doubly_stochastic(self):
        """sigma is a bijection, so the exact marginal matrix must be doubly stochastic.

        This is the property Sinkhorn's plan only approximates, and it is what makes the exact
        marginals the right reference to measure the mean-field bias against.
        """

        rng = np.random.default_rng(2)
        weights = rng.random((8, 8)) + 0.05

        marginals = permanent.permanent_marginals(weights)

        np.testing.assert_almost_equal(marginals.sum(axis=1), np.ones(8), decimal=9)
        np.testing.assert_almost_equal(marginals.sum(axis=0), np.ones(8), decimal=9)
        self.assertTrue((marginals >= 0).all())

    def test_near_deterministic_weights_give_a_permutation_matrix(self):
        """As eps -> 0 the posterior collapses onto the argmin, so the marginals become 0/1."""

        cost = np.array([[0.0, 5.0, 5.0], [5.0, 0.0, 5.0], [5.0, 5.0, 0.0]])
        weights = permanent.weights_from_cost(cost, eps=0.01)

        np.testing.assert_almost_equal(permanent.permanent_marginals(weights), np.eye(3), decimal=6)

    def test_uniform_weights_give_uniform_marginals(self):
        """As eps -> inf every permutation is equally likely, so every marginal is 1/n."""

        marginals = permanent.permanent_marginals(np.ones((5, 5)))

        np.testing.assert_almost_equal(marginals, np.full((5, 5), 0.2), decimal=9)

    def test_scale_invariance(self):
        """A global rescaling of the weights cancels in the marginal ratio."""

        rng = np.random.default_rng(3)
        weights = rng.random((6, 6)) + 0.05

        np.testing.assert_almost_equal(
            permanent.permanent_marginals(weights),
            permanent.permanent_marginals(weights * 1e6),
            decimal=9,
        )

    def test_single_atom(self):
        np.testing.assert_almost_equal(np.ones((1, 1)), permanent.permanent_marginals(np.ones((1, 1))))

    def test_rejects_all_zero_weights(self):
        with self.assertRaises(ValueError):
            permanent.permanent_marginals(np.zeros((3, 3)))


class WeightsAndEntropyTests(unittest.TestCase):
    def test_weights_from_cost_does_not_underflow(self):
        """Costs are shifted before exponentiation, so a large offset cannot flush to zero."""

        cost = np.array([[1000.0, 1001.0], [1002.0, 1000.5]])

        weights = permanent.weights_from_cost(cost, eps=0.5)

        self.assertEqual(1.0, weights.max())
        self.assertTrue((weights > 0).all())

    def test_normalised_entropy_bounds(self):
        n = 6
        self.assertAlmostEqual(1.0, permanent.normalised_entropy(np.full((n, n), 1.0 / n)), places=9)
        self.assertAlmostEqual(0.0, permanent.normalised_entropy(np.eye(n)), places=9)

    def test_normalised_entropy_of_trivial_matrix(self):
        self.assertEqual(0.0, permanent.normalised_entropy(np.ones((1, 1))))


if __name__ == "__main__":
    unittest.main()
