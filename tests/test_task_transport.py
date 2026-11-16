"""Independent transport solutions, paired-information checks, and input contracts."""
import unittest
import numpy as np
from core.task_transport import minimum_transport, paired_movement, DISTANCE


class TaskTransport(unittest.TestCase):
    def test_point_mass_has_exact_hamming_cost(self):
        p, q = np.eye(16)[3], np.eye(16)[0]
        result = minimum_transport(p, q)
        self.assertAlmostEqual(result["minimum"], 2)
        self.assertAlmostEqual(result["coupling"][3, 0], 1)
        self.assertLess(max(result["certificate"].values()), 1e-10)

    def test_opposing_observed_changes_can_have_zero_distribution_transport(self):
        observed = np.zeros((16, 16))
        observed[1, 2] = observed[2, 1] = 1
        result = paired_movement(observed)
        self.assertAlmostEqual(result["minimum"], 0)
        self.assertAlmostEqual(result["gross_turnover"], 2)
        self.assertAlmostEqual(result["net_count_change"], 0)
        self.assertAlmostEqual(result["same_count_changed_tasks_share"], 1)
        self.assertAlmostEqual(result["share_beyond_minimum"], 1)
        self.assertAlmostEqual(result["mutual_information_bits"], 1)

    def test_independent_coupling_has_zero_mutual_information(self):
        rng = np.random.default_rng(516)
        p, q = rng.dirichlet(np.ones(16), size=2)
        result = paired_movement(np.outer(p, q))
        self.assertAlmostEqual(result["mutual_information_bits"], 0, places=12)
        self.assertAlmostEqual(result["conditional_entropy_bits"], result["target_entropy_bits"], places=12)
        self.assertLessEqual(result["minimum"], np.sum(np.outer(p, q) * DISTANCE) + 1e-10)

    def test_random_feasible_pairs_retain_marginals_and_cost_inequality(self):
        rng = np.random.default_rng(517)
        for _ in range(8):
            result = paired_movement(rng.integers(0, 30, size=(16, 16)))
            np.testing.assert_allclose(result["coupling"].sum(1), result["source_distribution"], atol=1e-10)
            np.testing.assert_allclose(result["coupling"].sum(0), result["target_distribution"], atol=1e-10)
            self.assertGreaterEqual(result["excess_paired_movement"], -1e-10)

    def test_invalid_distributions_and_empty_pairs_are_rejected(self):
        p = np.repeat(1 / 16, 16)
        for invalid in (np.zeros(16), np.repeat(np.nan, 16), -p, np.ones(15)):
            with self.assertRaises(ValueError):
                minimum_transport(invalid, p)
        for invalid in (np.zeros((16, 16)), -np.ones((16, 16)), np.ones((15, 16))):
            with self.assertRaises(ValueError):
                paired_movement(invalid)
