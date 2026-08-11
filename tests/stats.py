import unittest

import numpy as np

import semlaflow.util.stats as stats


class CliffsDeltaTests(unittest.TestCase):
    def test_matches_the_pairwise_definition(self):
        """The fast rank-based form must equal mean(sign(a_i - b_j)) over all pairs."""

        rng = np.random.default_rng(0)
        for shift in [0.0, 0.3, 1.5]:
            a = rng.normal(0.0, 1.0, 200)
            b = rng.normal(shift, 1.0, 200)
            brute = float(np.mean(np.sign(a[:, None] - b[None, :])))

            self.assertAlmostEqual(brute, stats.cliffs_delta(a, b), places=9, msg=f"shift={shift}")

    def test_handles_ties(self):
        """Ties get averaged ranks, without which the rank identity silently breaks."""

        a = [1.0, 2.0, 2.0, 3.0]
        b = [2.0, 2.0, 3.0, 4.0]
        brute = float(np.mean(np.sign(np.array(a)[:, None] - np.array(b)[None, :])))

        self.assertAlmostEqual(brute, stats.cliffs_delta(a, b), places=9)

    def test_bounds_and_sign(self):
        self.assertAlmostEqual(1.0, stats.cliffs_delta([10, 11, 12], [1, 2, 3]), places=9)
        self.assertAlmostEqual(-1.0, stats.cliffs_delta([1, 2, 3], [10, 11, 12]), places=9)
        self.assertAlmostEqual(0.0, stats.cliffs_delta([1, 2, 3], [1, 2, 3]), places=9)

    def test_empty_input_is_nan_not_a_crash(self):
        self.assertTrue(np.isnan(stats.cliffs_delta([], [1, 2, 3])))


class MannWhitneyTests(unittest.TestCase):
    def test_detects_a_shift_and_reports_an_effect_size(self):
        rng = np.random.default_rng(1)
        result = stats.mann_whitney(rng.normal(0, 1, 500), rng.normal(1.0, 1, 500))

        self.assertLess(result["p"], 1e-10)
        self.assertLess(result["cliffs_delta"], -0.4)

    def test_identical_distributions_are_not_significant(self):
        rng = np.random.default_rng(2)
        result = stats.mann_whitney(rng.normal(0, 1, 500), rng.normal(0, 1, 500))

        self.assertGreater(result["p"], 0.01)
        self.assertLess(abs(result["cliffs_delta"]), 0.15)

    def test_too_few_values_returns_none_rather_than_raising(self):
        result = stats.mann_whitney([1.0], [2.0])

        self.assertIsNone(result["p"])
        self.assertIsNone(result["cliffs_delta"])

    def test_none_and_nan_are_dropped(self):
        result = stats.mann_whitney([1.0, None, float("nan"), 2.0, 3.0], [4.0, 5.0, 6.0])

        self.assertEqual(3, result["n_a"])
        self.assertEqual(3, result["n_b"])


class BootstrapTests(unittest.TestCase):
    def test_ci_contains_the_true_median_difference(self):
        rng = np.random.default_rng(3)
        a = rng.normal(5.0, 1.0, 400)
        b = rng.normal(3.0, 1.0, 400)

        result = stats.bootstrap_median_difference(a, b, n_bootstrap=2000, seed=0)

        self.assertLess(result["ci_low"], 2.0)
        self.assertGreater(result["ci_high"], 2.0)
        self.assertAlmostEqual(float(np.median(a) - np.median(b)), result["diff"], places=9)

    def test_ci_for_identical_distributions_straddles_zero(self):
        rng = np.random.default_rng(4)
        a = rng.normal(0, 1, 400)
        b = rng.normal(0, 1, 400)

        result = stats.bootstrap_median_difference(a, b, n_bootstrap=2000, seed=0)

        self.assertLess(result["ci_low"], 0.0)
        self.assertGreater(result["ci_high"], 0.0)

    def test_is_reproducible_for_a_fixed_seed(self):
        rng = np.random.default_rng(5)
        a, b = rng.normal(0, 1, 100), rng.normal(0.5, 1, 100)

        first = stats.bootstrap_median_difference(a, b, n_bootstrap=500, seed=7)
        second = stats.bootstrap_median_difference(a, b, n_bootstrap=500, seed=7)

        self.assertEqual(first, second)

    def test_too_few_values_returns_none(self):
        self.assertIsNone(stats.bootstrap_median_difference([1.0], [2.0])["diff"])


class DescriptiveTests(unittest.TestCase):
    def test_reports_median_and_mean(self):
        """Both, because these metrics are right-skewed and the two say different things."""

        result = stats.describe([1.0, 2.0, 3.0, 100.0])

        self.assertAlmostEqual(2.5, result["median"], places=9)
        self.assertAlmostEqual(26.5, result["mean"], places=9)
        self.assertEqual(4, result["n"])

    def test_empty_is_none_not_a_crash(self):
        result = stats.describe([None, float("nan")])

        self.assertEqual(0, result["n"])
        self.assertIsNone(result["median"])


class SizeMatchedTests(unittest.TestCase):
    def test_slot_matched_difference_is_descriptive_only(self):
        """No p-value is produced here -- blocking on size does not license a paired test."""

        result = stats.size_matched_difference([1.0, 2.0, 3.0], [0.5, 1.0, 1.5])

        self.assertEqual(3, result["n_slots"])
        self.assertAlmostEqual(1.0, result["mean_difference"], places=9)
        self.assertNotIn("p", result)

    def test_slots_with_a_missing_side_are_dropped(self):
        result = stats.size_matched_difference([1.0, None, 3.0], [0.5, 1.0, None])

        self.assertEqual(1, result["n_slots"])

    def test_no_usable_slots(self):
        result = stats.size_matched_difference([None, None], [None, None])

        self.assertEqual(0, result["n_slots"])
        self.assertIsNone(result["mean_difference"])


class CompareMetricTests(unittest.TestCase):
    def test_bundles_descriptives_test_and_bootstrap(self):
        rng = np.random.default_rng(6)
        result = stats.compare_metric(rng.normal(0, 1, 200), rng.normal(0.8, 1, 200), n_bootstrap=500)

        self.assertIn("median", result["a"])
        self.assertIsNotNone(result["test"]["p"])
        self.assertIsNotNone(result["test"]["cliffs_delta"])
        self.assertIsNotNone(result["bootstrap"]["ci_low"])


if __name__ == "__main__":
    unittest.main()
