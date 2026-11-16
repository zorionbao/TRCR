"""MOSAIC: marginal-and-association composition of joint disability.

The five-player game changes four marginal prevalences and the complete
association structure independently. Iterative proportional fitting preserves
all higher-order log-linear interactions of its seed distribution. Exact
Shapley averaging removes the arbitrary order of the five replacements.
"""
import math
import numpy as np

BITS = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(float)
COUNT = BITS.sum(1)
PLAYERS = ['Dressing', 'Eating', 'Bed transfer', 'Toileting', 'Joint association']


def simplex(x):
    """Euclidean projection onto the probability simplex."""
    u = np.sort(np.asarray(x))[::-1]
    css = np.cumsum(u) - 1
    k = np.flatnonzero(u - css / np.arange(1, len(u) + 1) > 0)[-1]
    return np.maximum(np.asarray(x) - css[k] / (k + 1), 0)


def smooth(p, rho=.005):
    return (1 - rho) * simplex(p) + rho / 16


def rake_many(seeds, targets, tol=2e-11, max_iter=3000):
    p = np.asarray(seeds, float).copy()
    p /= p.sum(1, keepdims=True)
    for it in range(max_iter):
        for j in range(4):
            yes = BITS[:, j] == 1
            v = p[:, yes].sum(1)
            p[:, yes] *= (targets[:, j] / v)[:, None]
            p[:, ~yes] *= ((1 - targets[:, j]) / (1 - v))[:, None]
        err = np.max(np.abs(p @ BITS - targets))
        if err < tol:
            return p, it + 1, err
    raise RuntimeError(f'IPF did not converge: {err}')


def decompose(p0, p1, rho=.005):
    p0, p1 = smooth(p0, rho), smooth(p1, rho)
    switch = ((np.arange(32)[:, None] >> np.arange(5)) & 1).astype(bool)
    targets = np.where(switch[:, :4], p1 @ BITS, p0 @ BITS)
    seeds = np.where(switch[:, 4, None], p1, p0)
    joint, iterations, error = rake_many(seeds, targets)
    outcomes = np.stack([(COUNT >= 1), (COUNT >= 2), (COUNT >= 3), COUNT], axis=1)
    value = joint @ outcomes
    phi = np.zeros((5, 4))
    for j in range(5):
        for s in range(32):
            if not (s & (1 << j)):
                n = int(s).bit_count()
                w = math.factorial(n) * math.factorial(4 - n) / math.factorial(5)
                phi[j] += w * (value[s | (1 << j)] - value[s])
    assert np.allclose(phi.sum(0), value[-1] - value[0], atol=1e-9)
    return dict(phi=phi, value=value, joint=joint, p0=p0, p1=p1,
                iterations=iterations, marginal_error=error)


def independent(p):
    m = np.asarray(p) @ BITS
    return np.prod(np.where(BITS, m, 1 - m), axis=1)


def information(p):
    p = np.maximum(np.asarray(p), 1e-15)
    ind = np.maximum(independent(p), 1e-15)
    return float(np.sum(p * np.log2(p / ind)))


def influence_operator(p0, p1, rho=.005):
    """Exact local derivative through the 32 calibrated distributions.

    The returned array has shape (player, outcome, arm, state). The operator
    differentiates on the simplex projection's current active face. Multiplier
    draws in the analysis additionally allow that face to change.
    """
    raw = [np.asarray(p0), np.asarray(p1)]
    r = decompose(p0, p1, rho)
    ps = [r['p0'], r['p1']]
    outcomes = np.stack([COUNT>=1, COUNT>=2, COUNT>=3, COUNT], axis=1).astype(float)
    dv = np.zeros((32, 4, 2, 16))
    conditions=[]
    for s in range(32):
        q = r['joint'][s]; marginal = q@BITS
        bc = BITS-marginal; yc = outcomes-q@outcomes
        covariance = bc.T@(q[:,None]*bc)
        conditions.append(np.linalg.cond(covariance))
        gradient = np.linalg.solve(covariance, bc.T@(q[:,None]*yc))
        arm = (s>>4)&1
        seed_gradient = (q/ps[arm])[:,None]*(yc-bc@gradient)
        dv[s,:,arm,:] += seed_gradient.T
        for j in range(4):
            target_arm=(s>>j)&1
            dv[s,:,target_arm,:] += np.outer(gradient[j],BITS[:,j])
    derivative=np.zeros((5,4,2,16))
    for j in range(5):
        for s in range(32):
            if not s&(1<<j):
                n=int(s).bit_count();w=math.factorial(n)*math.factorial(4-n)/math.factorial(5)
                derivative[j]+=w*(dv[s|(1<<j)]-dv[s])
    for arm in range(2):
        active=simplex(raw[arm])>1e-12
        jac=np.zeros((16,16));ix=np.flatnonzero(active)
        jac[np.ix_(ix,ix)]=np.eye(len(ix))-np.ones((len(ix),len(ix)))/len(ix)
        derivative[:,:,arm,:]=derivative[:,:,arm,:]@jac*(1-rho)
    return derivative, float(max(conditions))
