"""Closed-form RIFT half-risk identities under 50% task success.

Assumptions: four binary tasks, threshold two, equal unit attempt costs,
unrestricted task capacity, and sufficient total budget. These are mathematical
planning scenarios. Their application uses measured task portfolios and inputs.
"""
from __future__ import annotations

import numpy as np
from .probabilistic_restoration import (
    COUNT, independent_kernel, comonotonic_kernel, frontier, minimum_budget,
)


def half_risk_identity(probabilities, case_counts=None):
    """Return analytic minimum risk, feasibility, and shared-success budget.

    Optional integer counts provide an exact 3*n4 <= 4*n2 feasibility test.
    Probabilities must agree with those counts if both are supplied.
    """
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (16,) or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("Portfolio needs 16 finite nonnegative probabilities")
    if not np.isclose(p.sum(), 1, atol=1e-10, rtol=0):
        raise ValueError("Portfolio probabilities must sum to one")
    p = p / p.sum()
    p2, p3, p4 = (float(p[COUNT == k].sum()) for k in (2, 3, 4))
    feasible = 3 * p4 <= 4 * p2 + 1e-12
    if case_counts is not None:
        counts = np.asarray(case_counts, dtype=float)
        if (counts.shape != (16,) or not np.isfinite(counts).all()
                or np.any(counts < 0) or not np.all(counts == np.floor(counts))
                or counts.sum() <= 0):
            raise ValueError("Case counts need 16 nonnegative integers with a positive total")
        if not np.allclose(p, counts / counts.sum(), atol=1e-10, rtol=0):
            raise ValueError("Case counts and portfolio probabilities disagree")
        n2 = sum(int(x) for x in counts[COUNT == 2])
        n4 = sum(int(x) for x in counts[COUNT == 4])
        feasible = 3 * n4 <= 4 * n2
    return {
        "baseline_risk": p2 + p3 + p4,
        "half_risk_target": .5 * (p2 + p3 + p4),
        "independent_minimum_risk": .25 * p2 + .5 * p3 + .6875 * p4,
        "independent_half_risk_attainable": bool(feasible),
        "shared_uniform_half_risk_budget": p2 + 2 * p3 + 3 * p4,
        "sufficient_all_attempt_budget": float(p @ COUNT),
    }


def verify_half_risk_identities(portfolios):
    """Independently compare the formulas with LPs for any supplied portfolios.

    The study verification used 14 portfolios. This portable function accepts
    any number; its tests use 14 synthetic portfolios and embed no study data.
    """
    independent, shared = independent_kernel(.5), comonotonic_kernel(.5)
    reports = []
    for probabilities in portfolios:
        p = np.asarray(probabilities, dtype=float)
        identity = half_risk_identity(p)
        low = frontier(p, independent, identity["sufficient_all_attempt_budget"])
        ind_target = minimum_budget(p, independent, identity["half_risk_target"])
        shared_target = minimum_budget(p, shared, identity["half_risk_target"])
        error = abs(low["risk"] - identity["independent_minimum_risk"])
        budget_error = abs(shared_target["budget"] - identity["shared_uniform_half_risk_budget"])
        if (error > 1e-8 or budget_error > 1e-8 or
                bool(ind_target["attainable"]) != identity["independent_half_risk_attainable"]):
            raise RuntimeError("A half-risk identity disagrees with the allocation program")
        reports.append({**identity, "minimum_risk_error": error, "shared_budget_error": budget_error})
    return reports
