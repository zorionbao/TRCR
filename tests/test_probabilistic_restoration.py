"""Independent boundary solutions and stochastic-allocation regression tests."""
import json
import unittest
import numpy as np
from core.recovery_frontier import frontier as deterministic_frontier
from core.probabilistic_restoration import (
    ACTIONS, SOURCE, ATTEMPT_MASK, ATTEMPTS, BITS, COUNT, ORDERS,
    independent_kernel, comonotonic_kernel, validate_kernel, success_marginals,
    frontier, minimum_budget, fixed_order_actions,
)

METRICS = {}


def point_mass(state):
    p = np.zeros(16); p[state] = 1
    return p


class ProbabilisticRestorationRegression(unittest.TestCase):
    def certificate(self, result):
        keys = ["primal_dual_gap", "source_mass_error", "inequality_violation",
                "negative_flow", "dual_sign_violation", "reduced_cost_violation",
                "flow_complementarity_error", "inequality_complementarity_error"]
        error = max(result["certificate"][key] for key in keys)
        METRICS["maximum_certificate_error"] = max(
            METRICS.get("maximum_certificate_error", 0), error)
        self.assertLess(error, 1e-8)
        self.assertAlmostEqual(result["final_distribution"].sum(), 1, places=10)
        self.assertTrue(np.all(result["expected_restored"] <= result["attempted"] + 1e-10))

    def test_kernel_normalization_support_and_matching_marginals(self):
        rng = np.random.default_rng(2026100201)
        q = rng.uniform(0, 1, (16, 4)); errors = []
        for builder in (independent_kernel, comonotonic_kernel):
            kernel = builder(q)
            errors.append(float(abs(kernel.sum(1)-1).max()))
            expected = BITS[ATTEMPT_MASK] * q[SOURCE]
            errors.append(float(abs(success_marginals(kernel)-expected).max()))
            for i, (source, action) in enumerate(ACTIONS):
                for target in np.flatnonzero(kernel[i] > 0):
                    self.assertEqual(int(source & target), int(target))
                    self.assertEqual(int((source ^ target) & action), int(source ^ target))
        # Mixtures retain the complete supplied dependent success law.
        mixed = .37*independent_kernel(q)+.63*comonotonic_kernel(q)
        np.testing.assert_allclose(validate_kernel(mixed), mixed, atol=1e-14)
        METRICS["kernel_marginal_max_error"] = max(errors)
        self.assertLess(max(errors), 1e-12)

    def test_invalid_kernel_cannot_restore_unattempted_tasks(self):
        kernel = independent_kernel(.5)
        idx = np.flatnonzero((SOURCE == 3) & (ATTEMPT_MASK == 1))[0]
        bad = kernel.copy(); bad[idx] = 0; bad[idx, 0] = 1
        with self.assertRaises(ValueError): validate_kernel(bad)
        bad = kernel.copy(); bad[0, 0] = .9
        with self.assertRaises(ValueError): validate_kernel(bad)
        with self.assertRaises(ValueError): independent_kernel(1.1)
        with self.assertRaises(ValueError): fixed_order_actions([0, 1, 1, 3])

    def test_zero_success_and_exact_deterministic_reduction(self):
        rng = np.random.default_rng(2026100202); errors = []
        for _ in range(24):
            p = rng.dirichlet(np.ones(16)); budget = rng.uniform(0, 2)
            costs = rng.uniform(.4, 2, 4); caps = rng.uniform(0, .4, 4)
            threshold = int(rng.integers(1, 5))
            old = deterministic_frontier(p, budget, costs=costs,
                                        capacities=caps, threshold=threshold)
            for builder in (independent_kernel, comonotonic_kernel):
                zero = frontier(p, builder(0), budget, costs, caps, threshold)
                one = frontier(p, builder(1), budget, costs, caps, threshold)
                errors.append(abs(one["risk"]-old["risk"]))
                self.assertAlmostEqual(zero["risk"], p[COUNT >= threshold].sum(), places=11)
                self.certificate(zero); self.certificate(one)
        METRICS["deterministic_reduction_comparisons"] = len(errors)
        METRICS["deterministic_reduction_max_error"] = max(errors)
        self.assertLess(max(errors), 1e-10)

    def test_two_task_independent_analytic_solution(self):
        p = point_mass(3); q_high = .7; q_low = .3
        kernel = independent_kernel([q_low, q_high, .2, .9]); errors = []
        for budget in np.linspace(0, 2.5, 21):
            if budget <= 1:
                truth = 1-q_high*budget
            elif budget <= 2:
                truth = (1-q_high)*(1-q_low*(budget-1))
            else:
                truth = (1-q_high)*(1-q_low)
            result = frontier(p, kernel, budget)
            errors.append(abs(result["risk"]-truth)); self.certificate(result)
        METRICS["two_task_analytic_max_error"] = max(errors)
        self.assertLess(max(errors), 1e-10)

    def test_same_marginal_success_has_different_joint_threshold_frontier(self):
        # Three difficulties need two successful restorations to cross threshold 2.
        # At q=.5, three independent attempts give .5 success at cost 3.
        # A common draw gives .5 success with two attempts at cost 2.
        p = point_mass(7); errors = []; difference = None
        for builder, cap, slope in [(independent_kernel, 3., 1/6),
                                    (comonotonic_kernel, 2., .25)]:
            for budget in np.linspace(0, 3.5, 15):
                result = frontier(p, builder(.5), budget)
                errors.append(abs(result["risk"]-(1-slope*min(budget, cap))))
                np.testing.assert_allclose(result["expected_restored"], .5*result["attempted"], atol=1e-12)
                self.assertAlmostEqual(result["expected_success_cost"], .5*result["spent"], places=11)
                self.certificate(result)
        independent = frontier(p, independent_kernel(.5), 2)
        correlated = frontier(p, comonotonic_kernel(.5), 2)
        difference = independent["risk"]-correlated["risk"]
        METRICS["same_q_joint_kernel_risk_difference_at_budget2"] = difference
        METRICS["three_task_analytic_max_error"] = max(errors)
        self.assertAlmostEqual(difference, 1/6, places=10)
        self.assertLess(max(errors), 1e-10)

    def test_minimum_budget_and_infeasible_target(self):
        p = point_mass(7)
        for builder, expected in [(independent_kernel, 3.), (comonotonic_kernel, 2.)]:
            result = minimum_budget(p, builder(.5), .5)
            self.assertTrue(result["attainable"])
            self.assertAlmostEqual(result["budget"], expected, places=10)
            self.assertLessEqual(result["risk"], .5+1e-10); self.certificate(result)
        self.assertFalse(minimum_budget(p, independent_kernel(0), .5)["attainable"])
        self.assertFalse(minimum_budget(p, independent_kernel(.5), .49)["attainable"])
        self.assertFalse(minimum_budget(p, independent_kernel(1), .5,
                                        capacities=np.zeros(4))["attainable"])

    def test_heterogeneous_cost_capacity_monotonicity_and_prefix_controls(self):
        rng = np.random.default_rng(2026100203)
        p = rng.dirichlet(np.ones(16)); q = rng.uniform(.2, .85, (16, 4))
        costs = np.array([.8, 1.3, 2., 1.]); caps = np.array([.15, .3, .1, .3])
        gaps = []
        for builder in (independent_kernel, comonotonic_kernel):
            kernel = builder(q)
            risks = []
            for budget in np.linspace(0, 1.5, 16):
                result = frontier(p, kernel, budget, costs, caps)
                self.certificate(result); risks.append(result["risk"])
                self.assertLessEqual(result["spent"], budget+1e-10)
                self.assertTrue(np.all(result["attempted"] <= caps+1e-10))
            self.assertTrue(np.all(np.diff(risks) <= 1e-10))
            increasing_q = [frontier(p, builder(q*fraction), .4, costs, caps)["risk"]
                            for fraction in [0., .2, .5, 1.]]
            self.assertTrue(np.all(np.diff(increasing_q) <= 1e-10))
            relaxed = frontier(p, kernel, .4, costs, caps+.05)
            joint = frontier(p, kernel, .4, costs, caps)
            self.assertLessEqual(relaxed["risk"], joint["risk"]+1e-10)
            ordered = []
            for order in ORDERS:
                result = frontier(p, kernel, .4, costs, caps, order=order)
                self.certificate(result); ordered.append(result["risk"])
                self.assertTrue(np.all(result["allocation"][~fixed_order_actions(order)] == 0))
            gap = min(ordered)-joint["risk"]; gaps.append(gap)
            self.assertGreaterEqual(gap, -1e-10)
        METRICS["heterogeneous_best_order_minus_joint_risk"] = gaps

    def test_minimum_budget_certificate_with_heterogeneous_inputs(self):
        rng = np.random.default_rng(2026100204); successes = 0
        for _ in range(12):
            p = rng.dirichlet(np.ones(16)); q = rng.uniform(.5, .95, (16, 4))
            costs = rng.uniform(.5, 2., 4); caps = rng.uniform(.2, .5, 4)
            result = minimum_budget(p, independent_kernel(q), .75*p[COUNT >= 2].sum(),
                                    costs, caps)
            if result["attainable"]:
                self.certificate(result); successes += 1
                # Solving risk minimization with the minimum cost gives the same target.
                check = frontier(p, independent_kernel(q), result["budget"], costs, caps)
                self.assertLessEqual(check["risk"], result["target_risk"]+1e-9)
        self.assertGreater(successes, 0)
        METRICS["heterogeneous_minimum_budget_certificates"] = successes


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ProbabilisticRestorationRegression))
    METRICS["tests_run"] = result.testsRun
    METRICS["passed"] = result.wasSuccessful()
    print(json.dumps(METRICS, sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
