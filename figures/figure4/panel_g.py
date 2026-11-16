"""Figure 4G: Optimized risk contrasts between task-success dependence laws."""
import numpy as np
from figures.common import COHORTS, COL, subset, title, key, panel_main


def draw(ax, frame):
    data = subset(frame, 'G')
    data = data.loc[data.scenario.eq('Equal attempt costs and unrestricted task capacity')
                    & data.policy.eq('Joint subset optimization') & np.isclose(data.q, .5)]
    for cohort in COHORTS:
        group = data.loc[data.cohort.eq(cohort)].sort_values('budget')
        ax.plot(group.budget, group.independent_minus_comonotonic_risk*100,
                color=COL[cohort], lw=1.25)
    difference = data.independent_minus_comonotonic_risk*100
    lower = min(-.05, float(np.floor(difference.min()*1.08*2)/2))
    upper = max(.05, float(np.ceil(difference.max()*1.08*2)/2))
    ax.set(xlabel='Attempt units per person',
           ylabel='Success-law contrast (percentage points)',
           xlim=(0, 1.5), ylim=(lower, upper))
    ax.axhline(0, color='black', lw=.5, ls=':')
    title(ax, 'Joint success changes the frontier')
    key(ax, COHORTS, [COL[cohort] for cohort in COHORTS])


if __name__ == '__main__':
    panel_main(4, 'G', draw)
