"""RIFT with uncertain restoration and resources consumed by attempts.

Every action chooses a subset of a person's currently difficult tasks to
attempt. A complete transition kernel specifies which attempts succeed.
The kernel specifies a planning success law.
The objective retains the probability of the complete destination state.

All attempts are chosen together. A fixed task order permits only prefixes
of that order among the difficult tasks in each source state.
"""
from __future__ import annotations

import itertools
import numpy as np
from scipy.optimize import linprog

TASKS = ("Dressing", "Eating", "Bed transfer", "Toileting")
BITS = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(float)
COUNT = BITS.sum(1)
ACTIONS = np.array([(s, a) for s in range(16) for a in range(16)
                    if (s & a) == a], dtype=int)
SOURCE, ATTEMPT_MASK = ACTIONS.T
ATTEMPTS = BITS[ATTEMPT_MASK].T
MASS = (SOURCE[None, :] == np.arange(16)[:, None]).astype(float)
ORDERS = tuple(itertools.permutations(range(4)))


def _probabilities(probability):
    q = np.asarray(probability, float)
    if q.ndim == 0:
        q = np.full((16, 4), float(q))
    elif q.shape == (4,):
        q = np.broadcast_to(q, (16, 4)).copy()
    elif q.shape != (16, 4):
        raise ValueError("Success probability must be scalar, (4,), or (16,4)")
    if not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError("Success probabilities must be finite and in [0,1]")
    return q


def independent_kernel(probability):
    """Independent Bernoulli successes, conditional on source and action."""
    q = _probabilities(probability)
    kernel = np.zeros((len(ACTIONS), 16))
    for i, (source, action) in enumerate(ACTIONS):
        attempted = np.flatnonzero(BITS[action])
        for success in range(16):
            if (success & action) != success:
                continue
            probability_of_success = np.prod(np.where(
                BITS[success, attempted] == 1, q[source, attempted],
                1 - q[source, attempted]))
            kernel[i, source ^ success] = probability_of_success
    return validate_kernel(kernel)


def comonotonic_kernel(probability):
    """The same marginal probabilities with a common uniform success draw.

    Task j succeeds when U <= q[source,j]. With equal q, all attempted tasks
    succeed together with probability q and all fail with probability 1-q.
    With unequal q, successes are nested and retain every task's marginal q.
    """
    q = _probabilities(probability)
    kernel = np.zeros((len(ACTIONS), 16))
    for i, (source, action) in enumerate(ACTIONS):
        attempted = np.flatnonzero(BITS[action])
        breaks = np.unique(np.r_[0., q[source, attempted], 1.])
        for lower, upper in zip(breaks[:-1], breaks[1:]):
            middle = (lower + upper) / 2
            success = sum(1 << int(j) for j in attempted
                          if middle <= q[source, j])
            kernel[i, source ^ success] += upper - lower
    return validate_kernel(kernel)


def validate_kernel(kernel):
    """Validate the complete source/action-to-destination kernel.

    Unattempted difficult tasks cannot disappear, and an action cannot add
    difficulty. A caller can supply any joint success law satisfying these
    constraints, including dependence not represented by the two builders.
    """
    kernel = np.asarray(kernel, float)
    if kernel.shape != (len(ACTIONS), 16) or not np.isfinite(kernel).all():
        raise ValueError("The complete kernel must have shape (81,16) and be finite")
    if np.any(kernel < -1e-12):
        raise ValueError("Kernel probabilities must be nonnegative")
    kernel = np.maximum(kernel, 0).copy()
    if not np.allclose(kernel.sum(1), 1, atol=1e-10, rtol=0):
        raise ValueError("Every source/action kernel row must sum to one")
    for i, (source, action) in enumerate(ACTIONS):
        valid = np.array([(source & target) == target
                          and ((source ^ target) & action) == (source ^ target)
                          for target in range(16)])
        if np.any(kernel[i, ~valid] > 1e-12):
            raise ValueError("Kernel includes a new difficulty or an unattempted success")
        kernel[i, ~valid] = 0
    kernel /= kernel.sum(1, keepdims=True)
    return kernel


def fixed_order_actions(order):
    order = tuple(order)
    if len(order) != 4 or sorted(order) != list(range(4)):
        raise ValueError("A fixed order must be a permutation of tasks 0,1,2,3")
    allowed = np.zeros(len(ACTIONS), dtype=bool)
    lookup = {(int(s), int(a)): i for i, (s, a) in enumerate(ACTIONS)}
    for source in range(16):
        prefix = 0
        allowed[lookup[source, prefix]] = True
        for task in order:
            if source & (1 << task):
                prefix |= 1 << task
                allowed[lookup[source, prefix]] = True
    return allowed


def _inputs(p, kernel, costs, capacities, threshold, order):
    p = np.asarray(p, float)
    if p.shape != (16,) or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("Portfolio must contain 16 finite nonnegative probabilities")
    if not np.isclose(p.sum(), 1, atol=1e-10, rtol=0):
        raise ValueError("Portfolio probabilities must sum to one")
    costs = np.ones(4) if costs is None else np.asarray(costs, float)
    if costs.shape != (4,) or not np.isfinite(costs).all() or np.any(costs <= 0):
        raise ValueError("Attempt costs must be four positive finite values")
    if threshold not in (1, 2, 3, 4):
        raise ValueError("Severity threshold must be 1,2,3, or 4")
    if capacities is not None:
        capacities = np.asarray(capacities, float)
        if capacities.shape != (4,) or not np.isfinite(capacities).all() or np.any(capacities < 0):
            raise ValueError("Attempt capacities must be four nonnegative finite values")
    kernel = validate_kernel(kernel)
    allowed = np.ones(len(ACTIONS), dtype=bool) if order is None else fixed_order_actions(order)
    return p, kernel, costs, capacities, allowed


def _certificate(opt, objective, inequality, upper, p, allowed):
    equality_dual = opt.eqlin.marginals
    inequality_dual = opt.ineqlin.marginals
    reduced = objective - MASS.T @ equality_dual - inequality.T @ inequality_dual
    dual_objective = float(p @ equality_dual + upper @ inequality_dual)
    slack = upper - inequality @ opt.x
    return dict(
        primal_dual_gap=float(abs(opt.fun - dual_objective)),
        source_mass_error=float(np.max(abs(MASS @ opt.x - p))),
        inequality_violation=float(max(0., np.max(-slack))),
        negative_flow=float(max(0., -np.min(opt.x))),
        dual_sign_violation=float(max(0., np.max(inequality_dual))),
        reduced_cost_violation=float(max(0., -np.min(reduced[allowed]))),
        flow_complementarity_error=float(np.max(abs(opt.x[allowed] * reduced[allowed]))),
        inequality_complementarity_error=float(np.max(abs(slack * inequality_dual))),
        dual_objective=dual_objective,
        equality_dual=equality_dual,
        inequality_dual=inequality_dual,
        reduced_cost=reduced,
    )


def _result(opt, p, kernel, costs, risk_coefficients, certificate):
    final = opt.x @ kernel
    expected_restored = ((BITS[SOURCE] - kernel @ BITS).T @ opt.x)
    attempted = ATTEMPTS @ opt.x
    return dict(
        risk=float(risk_coefficients @ opt.x),
        final_distribution=final,
        attempted=attempted,
        expected_restored=expected_restored,
        spent=float(costs @ attempted),
        expected_success_cost=float(costs @ expected_restored),
        allocation=opt.x,
        certificate=certificate,
        success=True,
    )


def frontier(p, kernel, budget, costs=None, capacities=None, threshold=2, order=None):
    """Minimize final threshold risk using an attempt budget and capacities."""
    if not np.isfinite(budget) or budget < 0:
        raise ValueError("Attempt budget must be finite and nonnegative")
    p, kernel, costs, capacities, allowed = _inputs(
        p, kernel, costs, capacities, threshold, order)
    risk_coefficients = kernel @ (COUNT >= threshold)
    inequality = np.vstack([costs @ ATTEMPTS] + ([] if capacities is None else list(ATTEMPTS)))
    upper = np.r_[budget, [] if capacities is None else capacities]
    opt = linprog(risk_coefficients, A_ub=inequality, b_ub=upper, A_eq=MASS,
                  b_eq=p, bounds=[(0, None) if x else (0, 0) for x in allowed], method="highs")
    if not opt.success:
        raise RuntimeError("RIFT attempt allocation failed: " + opt.message)
    certificate = _certificate(opt, risk_coefficients, inequality, upper, p, allowed)
    result = _result(opt, p, kernel, costs, risk_coefficients, certificate)
    result.update(baseline=float(p @ (COUNT >= threshold)), budget=float(budget),
                  shadow=-opt.ineqlin.marginals, objective=float(opt.fun))
    return result


def minimum_budget(p, kernel, target_risk, costs=None, capacities=None, threshold=2, order=None):
    """Minimum attempt budget for a risk target; infeasible targets are explicit."""
    if not np.isfinite(target_risk) or not 0 <= target_risk <= 1:
        raise ValueError("Risk target must be finite and in [0,1]")
    p, kernel, costs, capacities, allowed = _inputs(
        p, kernel, costs, capacities, threshold, order)
    risk_coefficients = kernel @ (COUNT >= threshold)
    objective = costs @ ATTEMPTS
    inequality = np.vstack([risk_coefficients] + ([] if capacities is None else list(ATTEMPTS)))
    upper = np.r_[target_risk, [] if capacities is None else capacities]
    opt = linprog(objective, A_ub=inequality, b_ub=upper, A_eq=MASS, b_eq=p,
                  bounds=[(0, None) if x else (0, 0) for x in allowed], method="highs")
    if not opt.success:
        if opt.status == 2:
            return dict(success=False, attainable=False, budget=None,
                        target_risk=float(target_risk), status="infeasible risk target")
        raise RuntimeError("RIFT minimum attempt budget failed: " + opt.message)
    certificate = _certificate(opt, objective, inequality, upper, p, allowed)
    result = _result(opt, p, kernel, costs, risk_coefficients, certificate)
    result.update(attainable=True, budget=float(opt.fun), target_risk=float(target_risk),
                  baseline=float(p @ (COUNT >= threshold)), shadow=-opt.ineqlin.marginals)
    return result


def success_marginals(kernel):
    """Expected successful tasks for each source/action, shape (81,4)."""
    kernel = validate_kernel(kernel)
    return BITS[SOURCE] - kernel @ BITS
