"""Export the quantitative inputs for 32 reusable panel examples."""
from pathlib import Path
import numpy as np
import pandas as pd
from .clinical import CORE, SEEDS
from .joint_decomposition import COUNT, PLAYERS

COHORTS = tuple(SEEDS)
TASK_LABELS = ['Dressing', 'Eating', 'Bed transfer', 'Toileting']


def _read(output_dir, group, name):
    return pd.read_csv(Path(output_dir)/'stats'/group/(name+'.csv'))


def _panel(frame, label):
    frame = frame.copy()
    frame.insert(0, 'panel', label)
    return frame


def _save(frames, outdir, name):
    pd.concat(frames, ignore_index=True).to_csv(outdir/(name+'.csv'), index=False)


def _figure1(output_dir, outdir):
    effects = _read(output_dir, 'clinical', 'clinical_effects')
    effects = effects.loc[effects.analysis.eq('two_or_more4'),
                          ['cohort', 'analysis', 'n', 'persons', 'cases', 'case_risk',
                           'counterfactual_reference_risk', 'risk_difference', 'lo', 'hi']]
    joint = _read(output_dir, 'mosaic', 'joint_states')
    domains = _read(output_dir, 'clinical', 'clinical_domains')
    information = _read(output_dir, 'mosaic', 'joint_information')
    frames = [_panel(effects, 'Multiple daily activities affected'), _panel(effects, 'b')]
    for index, cohort in enumerate(COHORTS):
        group = joint.loc[joint.cohort.eq(cohort)].sort_values('state')
        frames.append(_panel(group, chr(99+index)))
    rows = []
    for cohort in COHORTS:
        group = joint.loc[joint.cohort.eq(cohort)]
        for arm in ['fracture', 'reference_projected']:
            for threshold in range(1, 5):
                rows.append(dict(cohort=cohort, arm=arm, threshold=threshold,
                                 risk=group.loc[group['count'] >= threshold, arm].sum()*100))
    frames.append(_panel(pd.DataFrame(rows), 'f'))
    for cohort in COHORTS:
        group = domains.loc[domains.cohort.eq(cohort) & domains.analysis.isin(CORE)]
        group = group.set_index('analysis').reindex(CORE).reset_index()
        frames.append(_panel(group[['cohort', 'analysis', 'risk_difference', 'lo', 'hi', 'n', 'cases']], 'g'))
    frames.append(_panel(information[['cohort', 'arm', 'multi_risk', 'independent_multi']], 'h'))
    _save(frames, outdir, 'Figure1_functional_burden')


def _observed_history(frame, lag):
    prefix = f'p{-lag}_' if lag < 0 else f'n{lag}_' if lag > 0 else ''
    observed = frame[prefix+'living_interview'].eq(True)
    if lag > 0:
        observed &= frame[f'obs{lag}'].eq(True)
    return frame.loc[observed], prefix


def _transitions(table, cohort, counts=True):
    group = table.loc[table.cohort.eq(cohort) & table.arm.eq(1)
                      & table.window.eq('Reporting to follow-up')].copy()
    if counts:
        group['start'] = group.from_state.map(lambda state: int(COUNT[int(state)]))
        group['end'] = group.to_state.map(lambda state: int(COUNT[int(state)]))
        matrix = (group.groupby(['start', 'end']).n.sum().unstack(fill_value=0)
                  .reindex(index=range(5), columns=range(5), fill_value=0))
    else:
        matrix = (group.pivot(index='from_state', columns='to_state', values='n')
                  .reindex(index=range(16), columns=range(16), fill_value=0))
    values = matrix.to_numpy(float)
    denominator = values.sum(1)
    return values/np.where(denominator > 0, denominator, 1)[:, None], denominator


def _figure2(output_dir, outdir):
    landmarks = pd.read_csv(Path(output_dir)/'harmonized/clinical_landmarks.csv', low_memory=False)
    phase = _read(output_dir, 'rift', 'turnover_phase')
    transitions = _read(output_dir, 'mosaic', 'joint_transitions')
    frames = []
    for index, cohort in enumerate(COHORTS):
        group = landmarks.loc[landmarks.cohort.eq(cohort) & landmarks.a.eq(1)]
        rows = []
        for lag in [-2, -1, 0, 1, 2]:
            observed, prefix = _observed_history(group, lag)
            for task, label in zip(CORE, TASK_LABELS):
                values = observed[prefix+task]
                rows.append(dict(cohort=cohort, lag=lag, task=label,
                                 risk=values.mean(), n=int(values.notna().sum())))
        frames.append(_panel(pd.DataFrame(rows), chr(97+index)))
        rows = phase.loc[phase.cohort.eq(cohort) & phase.arm.eq(1)
                         & phase.interval.eq('Report to follow-up')]
        frames.append(_panel(rows, cohort+' turnover'))
    rows = []
    for cohort in COHORTS:
        values, denominator = _transitions(transitions, cohort)
        exact, exact_denominator = _transitions(transitions, cohort, counts=False)
        same_state = []
        for count in range(5):
            select = COUNT == count
            same_state.append(np.sum(np.diag(exact)[select]*exact_denominator[select])
                              / max(exact_denominator[select].sum(), 1)*100)
        for count in range(5):
            rows.append(dict(cohort=cohort, count=count,
                             later_multiple=values[count, 2:].sum(),
                             same_state_percent=same_state[count], denominator=denominator[count]))
    frames.append(_panel(pd.DataFrame(rows), 'gh'))
    _save(frames, outdir, 'Figure2_persistence')


def _figure3(output_dir, outdir):
    components = _read(output_dir, 'mosaic', 'mosaic_components')
    coalitions = _read(output_dir, 'mosaic', 'mosaic_coalitions')
    benchmarks = _read(output_dir, 'mosaic', 'mosaic_benchmark_runs')
    summary = _read(output_dir, 'mosaic', 'mosaic_benchmark_summary')
    calibration = _read(output_dir, 'mosaic', 'mosaic_component_calibration')
    frames = []
    for cohort in COHORTS:
        group = components.loc[components.cohort.eq(cohort) & components.outcome.eq('multiple')]
        group = group.set_index('component').reindex(PLAYERS).reset_index()
        frames.append(_panel(group, cohort+' multiple'))
    for cohort in COHORTS:
        with np.load(Path(output_dir)/'stats/mosaic'/f'mosaic_bootstrap_{cohort.lower()}.npz') as bootstrap:
            draws = bootstrap['phi'][:, 4, 1]*100
        frames.append(_panel(pd.DataFrame({'cohort': cohort, 'draw': range(len(draws)),
                                          'association_pp': draws}), 'd'))
    frames.append(_panel(coalitions.loc[coalitions.outcome.eq('multiple')], 'e'))
    frames.append(_panel(benchmarks.loc[benchmarks.scenario.eq('Association change')], 'f'))
    frames.append(_panel(summary, 'g'))
    for scenario in ['Marginal change', 'Association change', 'Combined change']:
        group = calibration.loc[calibration.scenario.eq(scenario)]
        frames.append(_panel(group.set_index('component').reindex(PLAYERS).reset_index(), 'h'))
    _save(frames, outdir, 'Figure3_joint_composition')


def _figure4(output_dir, outdir):
    deterministic = _read(output_dir, 'rift', 'rift_frontiers')
    uncertainty = _read(output_dir, 'rift', 'rift_uncertainty')
    probabilistic = _read(output_dir, 'rift', 'rift_probabilistic_frontiers')
    contrast = _read(output_dir, 'rift', 'rift_probabilistic_dependence_contrasts')
    targets = _read(output_dir, 'rift', 'rift_probabilistic_half_risk')
    scenario = 'Equal attempt costs and unrestricted task capacity'
    policy = 'Joint subset optimization'
    frames = []
    for index, cohort in enumerate(COHORTS):
        left, right = chr(65+2*index), chr(66+2*index)
        frames.append(_panel(deterministic.loc[deterministic.cohort.eq(cohort)], left))
        frames.append(_panel(uncertainty.loc[uncertainty.cohort.eq(cohort)], left+' uncertainty'))
        group = probabilistic.loc[probabilistic.cohort.eq(cohort) & probabilistic.scenario.eq(scenario)
                                  & probabilistic.dependence.eq('Independent successes')
                                  & probabilistic.policy.eq(policy)]
        frames.append(_panel(group, right))
    for cohort in COHORTS:
        group = contrast.loc[contrast.cohort.eq(cohort) & contrast.scenario.eq(scenario)
                             & contrast.policy.eq(policy) & np.isclose(contrast.q, .5)]
        frames.append(_panel(group, 'G'))
        group = targets.loc[targets.cohort.eq(cohort) & targets.scenario.eq(scenario)
                            & targets.dependence.eq('Independent successes') & targets.attainable.eq(True)].sort_values('q')
        frames.append(_panel(group, 'H'))
    _save(frames, outdir, 'Figure4_capacity_opportunity')


def run(output_dir: Path):
    """Write four source tables linking core analyses to the reusable panel examples."""
    output_dir = Path(output_dir)
    outdir = output_dir/'figure_inputs'
    outdir.mkdir(parents=True, exist_ok=True)
    for export in [_figure1, _figure2, _figure3, _figure4]:
        export(output_dir, outdir)
    return outdir
