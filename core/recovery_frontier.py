"""Resource-indexed functional task restoration over complete joint states.

Finite linear programs combine task portfolios, restoration costs, task-specific
capacities, threshold outcomes, and dual resource prices.
"""
import numpy as np
from scipy.optimize import linprog
BITS=((np.arange(16)[:,None] >> np.arange(4)) & 1).astype(float)
COUNT=BITS.sum(1)
EDGES=np.array([(s,t) for s in range(16) for t in range(16) if (s&t)==t])
S,T=EDGES.T
RESTORE=(BITS[S]-BITS[T]).T
MASS=(S[None,:]==np.arange(16)[:,None]).astype(float)

def frontier(p,budget,costs=None,capacities=None,threshold=2,single_task=False):
    p=np.asarray(p,float);assert p.shape==(16,) and np.min(p)>=-1e-10 and abs(p.sum()-1)<1e-8
    costs=np.ones(4) if costs is None else np.asarray(costs,float)
    assert np.min(costs)>0 and budget>=0
    a=[costs@RESTORE];b=[budget]
    if capacities is not None:
        a.extend(RESTORE);b.extend(capacities)
    bounds=[(0,0) if single_task and (COUNT[s]-COUNT[t]>1) else (0,None) for s,t in EDGES]
    objective=(COUNT[T]>=threshold).astype(float)
    result=linprog(objective,A_ub=np.array(a),b_ub=np.array(b),A_eq=MASS,b_eq=p,
        bounds=bounds,method='highs')
    assert result.success,result.message
    mass=np.maximum(result.x,0)
    return dict(risk=float(result.fun),baseline=float(p[COUNT>=threshold].sum()),
        spent=float(costs@RESTORE@mass),restored=RESTORE@mass,
        shadow=-result.ineqlin.marginals,coupling=mass,
        constraint_error=float(np.max(np.abs(MASS@mass-p))))

def equal_cost_solution(p,budget,threshold=2):
    risk=float(p[COUNT>=threshold].sum())
    for k in range(threshold,5):
        unit=k-threshold+1; mass=p[COUNT==k].sum();take=min(mass,budget/unit)
        risk-=take;budget-=take*unit
    return max(0,risk)

def numerical_checks(seed=20260923):
    rng=np.random.default_rng(seed);errors=[];violations=[]
    for run in range(120):
        p=rng.dirichlet(np.ones(16));budget=rng.uniform(0,2)
        for threshold in [1,2,3]:
            r=frontier(p,budget,threshold=threshold)
            errors.append(abs(r['risk']-equal_cost_solution(p,budget,threshold)))
            violations.append(r['constraint_error'])
    p=np.ones(16)/16
    no=frontier(p,10,capacities=np.zeros(4))
    full=frontier(p,4)
    return dict(random_distributions=120,thresholds=3,linear_programs=360,
        max_closed_form_error=max(errors),max_mass_balance_error=max(violations),
        zero_capacity_identity=abs(no['risk']-no['baseline'])<1e-10,
        full_capacity_identity=full['risk']<1e-10,
        passed=max(errors)<1e-9 and max(violations)<1e-9 and full['risk']<1e-10)
