"""Probability, Shapley, and derivative identities for MOSAIC."""
import unittest
import numpy as np
from scipy.special import softmax
from core.joint_decomposition import BITS, simplex, decompose, influence_operator, independent, information


class JointDecompositionTests(unittest.TestCase):
    def setUp(self):
        self.p0 = softmax(BITS @ np.array([-1.8, -2.2, -1.7, -2.1])
                          + .7*BITS[:, 0]*BITS[:, 2] + .4*BITS[:, 1]*BITS[:, 3])
        self.p1 = softmax(np.log(self.p0) + BITS @ np.array([.7, .3, .8, .2])
                          + .5*BITS[:, 0]*BITS[:, 1]*BITS[:, 2])

    def test_projection_probability_and_shapley_additivity(self):
        projected = simplex(np.linspace(-.2, .3, 16))
        self.assertTrue(np.all(projected >= 0))
        self.assertAlmostEqual(projected.sum(), 1, places=12)
        result = decompose(self.p0, self.p1)
        np.testing.assert_allclose(result['joint'].sum(1), 1, atol=1e-10)
        np.testing.assert_allclose(result['phi'].sum(0), result['value'][-1]-result['value'][0], atol=1e-9)
        self.assertLess(result['marginal_error'], 2e-11)

    def test_identity_reversal_and_task_permutation(self):
        original = decompose(self.p0, self.p1)['phi']
        np.testing.assert_allclose(decompose(self.p0, self.p0)['phi'], 0, atol=1e-9)
        np.testing.assert_allclose(decompose(self.p1, self.p0)['phi'], -original, atol=1e-9)
        permutation = np.array([2, 0, 3, 1])
        states = (BITS[:, permutation]*2**np.arange(4)).sum(1).astype(int)
        p0, p1 = np.zeros(16), np.zeros(16)
        p0[states], p1[states] = self.p0, self.p1
        changed = decompose(p0, p1)['phi']
        np.testing.assert_allclose(changed[:4], original[permutation], atol=1e-9)
        np.testing.assert_allclose(changed[4], original[4], atol=1e-9)

    def test_derivative_against_centered_probability_perturbations(self):
        derivative, condition = influence_operator(self.p0, self.p1)
        self.assertTrue(np.isfinite(condition))
        epsilon = 1e-6
        for arm in range(2):
            for state in [1, 7, 15]:
                direction = np.eye(16)[state]-np.eye(16)[0]
                plus, minus = [self.p0.copy(), self.p1.copy()], [self.p0.copy(), self.p1.copy()]
                plus[arm] += epsilon*direction
                minus[arm] -= epsilon*direction
                difference = (decompose(*plus)['phi']-decompose(*minus)['phi'])/(2*epsilon)
                np.testing.assert_allclose(difference, derivative[:, :, arm] @ direction,
                                           atol=1e-6, rtol=1e-6)

    def test_product_portfolio_preserves_marginals_and_has_zero_information(self):
        product = independent(self.p0)
        np.testing.assert_allclose(product @ BITS, self.p0 @ BITS, atol=1e-12)
        self.assertAlmostEqual(product.sum(), 1, places=12)
        self.assertAlmostEqual(information(product), 0, places=12)


if __name__ == '__main__':
    unittest.main()
