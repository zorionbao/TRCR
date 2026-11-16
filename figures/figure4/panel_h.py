"""Figure 4H: Minimum attempt budgets for attainable half-risk targets."""
import numpy as np
from figures.common import COHORTS, COL, subset, title, key, panel_main


def draw(ax, frame):
    data = subset(frame, 'H')
    data = data.loc[data.scenario.eq('Equal attempt costs and unrestricted task capacity')
                    & data.dependence.eq('Independent successes') & data.attainable.eq(True)]
    markers = {'HRS': '^', 'ELSA': 's', 'CHARLS': 'o'}
    styles = {'HRS': '-.', 'ELSA': '--', 'CHARLS': '-'}
    for cohort in COHORTS:
        group = data.loc[data.cohort.eq(cohort)].sort_values('q')
        ax.plot(group.q, group.minimum_attempt_budget, marker=markers[cohort],
                linestyle=styles[cohort], ms=3, color=COL[cohort], lw=1.2)
    upper = np.ceil(data.minimum_attempt_budget.max()*1.15*5)/5
    ax.set(xlabel='Task success probability', ylabel='Half-risk attempt budget per person',
           xlim=(0, 1), ylim=(0, upper))
    title(ax, 'Resource requirements for a risk target')
    key(ax, COHORTS, [COL[cohort] for cohort in COHORTS],
        markers=[markers[cohort] for cohort in COHORTS],
        styles=[styles[cohort] for cohort in COHORTS])


if __name__ == '__main__':
    panel_main(4, 'H', draw)
