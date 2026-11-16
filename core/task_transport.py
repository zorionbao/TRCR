"""Exact transport of four-task state distributions and observed paired movement.

States 0-15 encode dressing, eating, bed transfer, and toileting in bit order.
Hamming distance counts changed tasks. Multiple optimal couplings may have
the same minimum; the supplied observed coupling records actual paired changes.
Functions accept arrays, return arrays or dictionaries, and write no files.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.special import xlogy

BITS = (np.arange(16)[:, None] >> np.arange(4)) & 1
DISTANCE = np.sum(BITS[:, None, :] != BITS[None, :, :], axis=2).astype(float)
MARGINALS = np.zeros((32, 256))
for _state in range(16):
    MARGINALS[_state, 16 * _state:16 * (_state + 1)] = 1
    MARGINALS[16 + _state, _state::16] = 1


def _distribution(values):
    p = np.asarray(values, dtype=float)
    if p.shape != (16,) or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("A state distribution needs 16 finite nonnegative probabilities")
    if not np.isclose(p.sum(), 1, atol=1e-10, rtol=0):
        raise ValueError("State probabilities must sum to one")
    return p / p.sum()


def entropy_bits(values):
    """Shannon entropy in bits, accepting a validated probability array."""
    p = np.asarray(values, dtype=float)
    return float(-np.sum(xlogy(p, p)) / np.log(2))


def minimum_transport(source, target):
    """Minimum expected Hamming distance with specified state marginals.

    Return one optimal coupling and its primal/dual certificate. The numerical
    optimum describes a distributional requirement, not an identified trajectory.
    """
    p, q = _distribution(source), _distribution(target)
    weights = np.r_[p, q]
    opt = linprog(DISTANCE.ravel(), A_eq=MARGINALS, b_eq=weights,
                  bounds=(0, None), method="highs")
    if not opt.success:
        raise RuntimeError("Task transport failed: " + opt.message)
    coupling = opt.x.reshape(16, 16)
    reduced = DISTANCE.ravel() - MARGINALS.T @ opt.eqlin.marginals
    certificate = {
        "primal_dual_gap": float(abs(opt.fun - weights @ opt.eqlin.marginals)),
        "marginal_error": float(np.max(abs(MARGINALS @ opt.x - weights))),
        "negative_mass": float(max(0, -opt.x.min())),
        "dual_feasibility_error": float(max(0, -reduced.min())),
    }
    if max(certificate.values()) >= 1e-8:
        raise RuntimeError("Task transport failed its numerical certificate")
    return {"minimum": float(opt.fun), "coupling": coupling, "certificate": certificate}


def paired_movement(counts_or_mass):
    """Compare a 16-by-16 observed transition table with minimum transport.

    Rows are reporting states and columns are follow-up states. Nonnegative
    counts or masses are normalized by their positive total. The function also
    returns observed state persistence and information summaries.
    """
    count = np.asarray(counts_or_mass, dtype=float)
    if count.shape != (16, 16) or not np.isfinite(count).all() or np.any(count < 0):
        raise ValueError("Paired transitions need a finite nonnegative 16-by-16 table")
    if count.sum() <= 0:
        raise ValueError("Paired transitions need a positive total")
    observed = count / count.sum()
    p, q = observed.sum(1), observed.sum(0)
    result = minimum_transport(p, q)
    gross = float(np.sum(observed * DISTANCE))
    if result["minimum"] > gross + 1e-8:
        raise RuntimeError("Minimum transport exceeds feasible observed movement")
    excess = gross - result["minimum"]
    task_count = BITS.sum(1)
    h_source, h_target = entropy_bits(p), entropy_bits(q)
    h_joint = entropy_bits(observed)
    result.update(
        observed_coupling=observed, source_distribution=p, target_distribution=q,
        gross_turnover=gross, net_count_change=float((q - p) @ task_count),
        excess_paired_movement=excess,
        share_beyond_minimum=excess / gross if gross > 0 else 0.,
        source_entropy_bits=h_source, target_entropy_bits=h_target,
        conditional_entropy_bits=h_joint - h_source,
        mutual_information_bits=h_source + h_target - h_joint,
        jensen_shannon_bits=entropy_bits((p + q) / 2) - (h_source + h_target) / 2,
        same_state_share=float(np.trace(observed)),
        same_count_changed_tasks_share=float(observed[
            (task_count[:, None] == task_count[None, :]) & (DISTANCE > 0)].sum()),
    )
    return result
