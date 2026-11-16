"""Closed-form and constrained restoration identities for RIFT."""
import unittest
import numpy as np
from core.recovery_frontier import COUNT, frontier, equal_cost_solution


class RecoveryFrontierTests(unittest.TestCase):
    def test_linear_program_matches_severity_specific_closed_form(self):
        rng = np.random.default_rng(20260923)
        for _ in range(12):
            probability = rng.dirichlet(np.ones(16))
            budget = rng.uniform(0, 2)
            for threshold in [1, 2, 3]:
                result = frontier(probability, budget, threshold=threshold)
                self.assertAlmostEqual(result['risk'], equal_cost_solution(probability, budget, threshold), places=9)
                self.assertLess(result['constraint_error'], 1e-9)
                self.assertLessEqual(result['spent'], budget+1e-9)

    def test_capacities_and_single_task_policy(self):
        probability = np.ones(16)/16
        zero = frontier(probability, 10, capacities=np.zeros(4))
        self.assertAlmostEqual(zero['risk'], probability[COUNT >= 2].sum(), places=10)
        self.assertAlmostEqual(frontier(probability, 4)['risk'], 0, places=10)
        joint = frontier(probability, .4)
        single = frontier(probability, .4, single_task=True)
        self.assertGreaterEqual(single['risk'], joint['risk']-1e-10)
        risks = [frontier(probability, budget)['risk'] for budget in np.linspace(0, 1.5, 16)]
        self.assertTrue(np.all(np.diff(risks) <= 1e-10))


if __name__ == '__main__':
    unittest.main()
