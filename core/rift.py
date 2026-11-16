"""Joint-portfolio restoration and attempt-resource frontiers."""
from pathlib import Path
import numpy as np
import pandas as pd
from .clinical import SEEDS
from .recovery_frontier import BITS, COUNT, frontier as deterministic_frontier, equal_cost_solution
from .probabilistic_restoration import independent_kernel, comonotonic_kernel, frontier, minimum_budget

BUDGETS = np.linspace(0, 1.5, 61)
Q_VALUES = (.25, .5, .75, 1.)
Q_TARGET = np.linspace(0, 1, 21)
SCENARIO = 'Equal attempt costs and unrestricted task capacity'
POLICY = 'Joint subset optimization'


def _portfolios(output_dir, input_dir=None):
    source = Path(input_dir)/'joint_states.csv' if input_dir is not None else Path(output_dir)/'stats/mosaic/joint_states.csv'
    states = pd.read_csv(source)
    required = {'cohort', 'state', 'fracture', 'case_count'}
    missing = sorted(required.difference(states.columns))
    if missing:
        raise ValueError('Joint-state input requires fields: '+', '.join(missing))
    portfolios = {}
    for cohort in SEEDS:
        group = states.loc[states.cohort.eq(cohort)].sort_values('state')
        if group.state.tolist() != list(range(16)):
            raise ValueError('A complete 16-state portfolio is required for '+cohort)
        raw_counts = pd.to_numeric(group.case_count, errors='raise').to_numpy(float)
        if (not np.isfinite(raw_counts).all() or np.any(raw_counts < 0)
                or np.any(raw_counts != np.floor(raw_counts))
                or raw_counts.sum() <= 0 or raw_counts.sum() >= np.iinfo(np.int64).max):
            raise ValueError('case_count requires finite, nonnegative integers with a positive total for '+cohort)
        counts = raw_counts.astype(np.int64)
        total = int(counts.sum())
        probability = pd.to_numeric(group.fracture, errors='raise').to_numpy(float)
        if (not np.isfinite(probability).all() or np.any(probability < 0)
                or not np.allclose(probability, counts/total, atol=1e-12, rtol=0)):
            raise ValueError('fracture probabilities must match the case_count proportions for '+cohort)
        portfolios[cohort] = (probability, total)
    return portfolios


def _deterministic(portfolios, outdir):
    rows, uncertainty = [], []
    for index, (cohort, (p, n)) in enumerate(portfolios.items()):
        marginals = p @ BITS
        independent = np.prod(np.where(BITS, marginals, 1-marginals), axis=1)
        for method in ['Joint states', 'Independent tasks', 'One task per person']:
            portfolio = independent if method == 'Independent tasks' else p
            for budget in BUDGETS:
                result = deterministic_frontier(portfolio, float(budget),
                                                single_task=method == 'One task per person')
                rows.append(dict(cohort=cohort, method=method, budget=budget,
                                 risk=result['risk'], reduction=result['baseline']-result['risk'],
                                 spent=result['spent'], budget_shadow=result['shadow'][0]))
        rng = np.random.default_rng(20260923+index)
        draws = rng.multinomial(n, p, size=1000)/n
        for budget in BUDGETS:
            values = [equal_cost_solution(draw, budget) for draw in draws]
            lo, hi = np.quantile(values, [.025, .975])
            uncertainty.append(dict(cohort=cohort, budget=budget, lo=lo, hi=hi,
                                    draws=1000, cases=n))
    pd.DataFrame(rows).to_csv(outdir/'rift_frontiers.csv', index=False)
    pd.DataFrame(uncertainty).to_csv(outdir/'rift_uncertainty.csv', index=False)


def _probabilistic(portfolios, outdir):
    rows, targets = [], []
    builders = {'Independent successes': independent_kernel,
                'Comonotonic successes': comonotonic_kernel}
    for dependence, builder in builders.items():
        probabilities = Q_VALUES if dependence == 'Independent successes' else (.5,)
        for q in probabilities:
            kernel = builder(q)
            for cohort, (p, _) in portfolios.items():
                for budget in BUDGETS:
                    result = frontier(p, kernel, float(budget))
                    rows.append(dict(cohort=cohort, scenario=SCENARIO, dependence=dependence,
                                     q=q, policy=POLICY, budget=budget, baseline=result['baseline'],
                                     risk=result['risk'], spent=result['spent'],
                                     expected_restorations_per_person=float(result['expected_restored'].sum())))
    frame = pd.DataFrame(rows)
    frame.to_csv(outdir/'rift_probabilistic_frontiers.csv', index=False)
    keys = ['cohort', 'scenario', 'q', 'policy', 'budget']
    columns = keys+['risk', 'spent', 'expected_restorations_per_person']
    independent = frame.loc[frame.dependence.eq('Independent successes') & frame.q.eq(.5), columns]
    correlated = frame.loc[frame.dependence.eq('Comonotonic successes'), columns]
    contrasts = independent.merge(correlated, on=keys, validate='one_to_one',
                                  suffixes=('_independent', '_comonotonic'))
    contrasts['independent_minus_comonotonic_risk'] = contrasts.risk_independent-contrasts.risk_comonotonic
    contrasts.to_csv(outdir/'rift_probabilistic_dependence_contrasts.csv', index=False)
    for q in Q_TARGET:
        kernel = independent_kernel(float(q))
        for cohort, (p, _) in portfolios.items():
            baseline = float(p[COUNT >= 2].sum())
            result = minimum_budget(p, kernel, .5*baseline)
            full = frontier(p, kernel, 4.)
            targets.append(dict(cohort=cohort, scenario=SCENARIO, dependence='Independent successes',
                                q=float(q), baseline=baseline, target_risk=.5*baseline,
                                attainable=result['attainable'], minimum_attempt_budget=result['budget'],
                                achieved_risk=result.get('risk', np.nan),
                                minimum_risk_at_all_tasks_attempted=full['risk']))
    pd.DataFrame(targets).to_csv(outdir/'rift_probabilistic_half_risk.csv', index=False)


def run(output_dir: Path, input_dir=None):
    """Calculate restoration-resource inputs for the eight group-4 panel examples."""
    output_dir = Path(output_dir)
    portfolios = _portfolios(output_dir, input_dir)
    outdir = output_dir/'stats/rift'
    outdir.mkdir(parents=True, exist_ok=True)
    _deterministic(portfolios, outdir)
    _probabilistic(portfolios, outdir)
    return outdir
