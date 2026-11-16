"""Fourteen synthetic portfolio checks of analytic RIFT conditions."""
import unittest
import numpy as np
from core.recovery_identities import half_risk_identity, verify_half_risk_identities


class RecoveryIdentities(unittest.TestCase):
    def test_fourteen_portfolios_match_allocation_programs(self):
        portfolios = [np.eye(16)[k] for k in (0, 1, 3, 7, 15)]
        boundary = np.zeros(16)
        boundary[3], boundary[15] = 3 / 7, 4 / 7
        portfolios.append(boundary)
        portfolios.extend(np.random.default_rng(518).dirichlet(np.ones(16), size=8))
        reports = verify_half_risk_identities(portfolios)
        self.assertEqual(len(reports), 14)
        self.assertLess(max(x["minimum_risk_error"] for x in reports), 1e-10)
        self.assertLess(max(x["shared_budget_error"] for x in reports), 1e-10)
        self.assertTrue(reports[5]["independent_half_risk_attainable"])

    def test_integer_counts_preserve_the_exact_feasibility_boundary(self):
        counts = np.zeros(16, dtype=int)
        counts[3], counts[15] = 3, 4
        self.assertTrue(half_risk_identity(counts / counts.sum(), counts)["independent_half_risk_attainable"])
        counts[15] = 5
        self.assertFalse(half_risk_identity(counts / counts.sum(), counts)["independent_half_risk_attainable"])

    def test_invalid_and_disagreeing_counts_are_rejected(self):
        p = np.repeat(1 / 16, 16)
        with self.assertRaises(ValueError):
            half_risk_identity(np.zeros(16))
        with self.assertRaises(ValueError):
            half_risk_identity(p, np.repeat(.5, 16))
        with self.assertRaises(ValueError):
            half_risk_identity(p, np.arange(16))
